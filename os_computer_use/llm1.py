import os
from llama_cpp import Llama

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "./models/Phi-4-mini-instruct.Q8_0.gguf")
PLANNER_CTX = int(os.environ.get("PLANNER_CTX", "8192"))
PLANNER_NGPU = int(os.environ.get("PLANNER_NGPU", "35"))

planner = Llama(
    model_path=PLANNER_MODEL,
    n_ctx=PLANNER_CTX,
    n_gpu_layers=PLANNER_NGPU,
    seed=42,
    # 🔴 Önemli: tool-calling için uygun chat formatı belirt
    chat_format="chatml-function-calling",
)

VISION_MODEL = os.environ.get("VISION_MODEL", "./models/UI-TARS-1.5-7B-q5_k_m.gguf")
VISION_MMPROJ = os.environ.get("VISION_MMPROJ", "./models/UI-TARS-1.5-7B-q8_0.mmproj")  
VISION_CTX = int(os.environ.get("VISION_CTX", "4096"))
VISION_NGPU = int(os.environ.get("VISION_NGPU", "35"))

vision_kwargs = dict(
    model_path=VISION_MODEL,
    n_ctx=VISION_CTX,
    n_gpu_layers=VISION_NGPU,
    seed=123,
)
if VISION_MMPROJ:
    vision_kwargs["mmproj_path"] = VISION_MMPROJ  # Qwen2-VL için projector gerekebilir
vision = Llama(**vision_kwargs)
