#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, re, json, base64, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Any, Dict, List, Tuple

os.environ.setdefault("CURL_CA_BUNDLE", "")

import torch
from PIL import Image

# ===================== PLANNER (llama.cpp) =====================
from llama_cpp import Llama

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "./models/microsoft_Fara-7B-Q6_K.gguf")
PLANNER_CTX   = int(os.environ.get("PLANNER_CTX", "8192"))
PLANNER_NGPU  = int(os.environ.get("PLANNER_NGPU", "25"))  # OOM'i azaltmak için varsayılanı 0 yaptım

planner = Llama(
    model_path=PLANNER_MODEL,
    n_ctx=PLANNER_CTX,
    n_gpu_layers=PLANNER_NGPU,
    seed=42,
    chat_format="chatml-function-calling",
)

# ===================== VISION (HF ShowUI-2B) ===================
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # ShowUI kartındaki yardımcı

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # HF kartındaki yardımcı fn


def _decode_data_uri_to_pil(data_uri: str) -> Image.Image:
    if data_uri.startswith("data:"):
        b64 = data_uri.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if os.path.exists(data_uri):
        return Image.open(data_uri).convert("RGB")
    raise ValueError("Unsupported image reference")

class _ShowUIHFShim:
    def __init__(self):
        model_root = os.environ.get("SHOWUI_MODEL", os.path.join(os.path.dirname(__file__), "ShowUI-2B"))
        if not os.path.isdir(model_root):
            raise RuntimeError(f"SHOWUI_MODEL path not found: {model_root}")

        use_gpu = torch.cuda.is_available()
        dtype = torch.float16 if use_gpu else torch.float32

        # ENV ile zorlayalım: SHOWUI_FORCE_GPU=1 ise tamamen cuda:0'a yükle
        force_gpu = os.environ.get("SHOWUI_FORCE_GPU", "0") == "1"

        if use_gpu and force_gpu:
            device_map = {"": 0}   # tüm ağı cuda:0’a
        elif use_gpu:
            device_map = "auto"    # VRAM'e göre paylaştır
        else:
            device_map = "cpu"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_root,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_root,
            min_pixels=224 * 28 * 28,
            max_pixels=1008 * 28 * 28,
            local_files_only=True,
            trust_remote_code=True,
        )
    def _messages_to_qwen_format(self, messages: List[Dict[str, Any]]):
        out_msgs, pil_images = [], []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content")
            if isinstance(content, str):
                out_msgs.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue
            parts = []
            for p in content:
                if p.get("type") == "text" or "text" in p:
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") in ("image", "image_url") or "image_url" in p:
                    if "image" in p and isinstance(p["image"], Image.Image):
                        pil = p["image"].convert("RGB")
                    else:
                        url = p.get("image_url", {}).get("url") or p.get("image")
                        pil = _decode_data_uri_to_pil(url)
                    pil_images.append(pil)
                    parts.append({"type": "image", "image": pil})
            if parts:
                out_msgs.append({"role": role, "content": parts})
        return out_msgs, pil_images
        
    @torch.inference_mode()
    def create_chat_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.0, max_new_tokens: int = 64):
        qwen_msgs, _ = self._messages_to_qwen_format(messages)
        text = self.processor.apply_chat_template(qwen_msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(qwen_msgs)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        # deterministik üretim
        gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
        out_text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        # son asistan parçasını al
        if "<|im_start|>" in out_text:
            out_text = out_text.split("<|im_start|>assistant")[-1].strip()
        return {"choices": [{"message": {"content": out_text}}]}

# sandbox_agent.py içe aktaracak:
vision = _ShowUIHFShim()
