#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import base64
import io
import json
import os
import re
import shlex
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from .llm import planner, vision  # planner: llama.cpp, vision: HF ShowUI shim

CLICK_DIR = "./"
RAW_DIR = "/tmp/showui_raw"
LOG_PATH = "/tmp/agent_log.jsonl"

os.makedirs(CLICK_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def _log(ev: Dict[str, Any]):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception:
        pass

def run(cmd: str) -> str:
    try:
        out = os.popen(cmd).read()
    except Exception as e:
        out = str(e)
    _log({"ts": time.time(), "type": "run", "cmd": cmd, "stdout": out[:5000]})
    return out

def take_screenshot_png() -> bytes:
    try:
        import subprocess
        return subprocess.check_output("import -window root png:-", shell=True)
    except Exception:
        import subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".xwd", delete=False)
        tmp.close()
        os.system(f"xwd -root -silent -out {tmp.name}")
        png = subprocess.check_output(f"convert {tmp.name} png:-", shell=True)
        os.unlink(tmp.name)
        return png

def save_overlay(img_b: bytes, x: int, y: int, label: str) -> str:
    img = Image.open(io.BytesIO(img_b)).convert("RGBA")
    d = ImageDraw.Draw(img)
    r = 16
    d.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,200), outline=(255,255,255,220), width=3)
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", label)[:48]
    fn = f"click_{safe}_{x}_{y}_{int(time.time()*1000)}.png"
    path = os.path.join(CLICK_DIR, fn)
    img.save(path)
    return path

# --------------- ShowUI grounding (HF Quick Start format) --------------------

_UI_SYSTEM = (
    "Based on the screenshot of the page, I give a text description and you give its corresponding location. "
    "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate "
    "on the screenshot, scaled from 0 to 1."
)
_MIN_PIXELS = 256 * 28 * 28
_MAX_PIXELS = 1344 * 28 * 28

def _parse_xy(txt: str):
    s = txt.strip()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            return float(obj[0]), float(obj[1])
        if isinstance(obj, dict) and "position" in obj and isinstance(obj["position"], (list, tuple)):
            pos = obj["position"]
            if len(pos) == 2: return float(pos[0]), float(pos[1])
        if isinstance(obj, dict) and {"x","y"} <= set(obj):
            return float(obj["x"]), float(obj["y"])
    except Exception:
        pass
    m = re.search(r"\[\s*([0-1](?:\.\d+)?)\s*,\s*([0-1](?:\.\d+)?)\s*\]", s)
    if m: return float(m.group(1)), float(m.group(2))
    mx = re.search(r'"?x"?\s*:\s*([0-1](?:\.\d+)?)', s); my = re.search(r'"?y"?\s*:\s*([0-1](?:\.\d+)?)', s)
    if mx and my: return float(mx.group(1)), float(my.group(1))
    raise ValueError(f"Cannot parse coordinates from: {s[:160]}")

def showui_ground(query: str):
    png = take_screenshot_png()
    pil = Image.open(io.BytesIO(png)).convert("RGB")
    W, H = pil.size
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": _UI_SYSTEM},
            {"type": "image", "image": pil, "min_pixels": _MIN_PIXELS, "max_pixels": _MAX_PIXELS},
            {"type": "text", "text": query},
        ],
    }]
    resp = vision.create_chat_completion(messages=messages, temperature=0.0, max_new_tokens=64)
    content = (resp["choices"][0]["message"]["content"] or "").strip()

    raw_path = os.path.join(RAW_DIR, f"vlm_{int(time.time()*1000)}.txt")
    try:
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass
    _log({"ts": time.time(), "type": "showui_raw", "query": query, "raw_path": raw_path, "content_head": content[:300]})

    nx, ny = _parse_xy(content)        # 0..1
    x = int(round(nx * W));  y = int(round(ny * H))
    x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
    if x <= 2 and y <= 2:
        raise RuntimeError("Suspicious (0,0) suppressed.")
    overlay_path = save_overlay(png, x, y, query)
    _log({"ts": time.time(), "type": "showui_click", "query": query, "x": x, "y": y, "overlay_png": overlay_path})
    return x, y, overlay_path

def _safe_click(x: int, y: int, reason: str):
    if x is None or y is None:
        return {"ok": False, "reason": "no_coords"}
    if x <= 2 and y <= 2:
        _log({"ts": time.time(), "type": "blocked_corner_click", "x": x, "y": y, "reason": reason})
        return {"ok": False, "reason": "corner_0_0_blocked"}
    run("xdotool keyup ctrl alt shift")
    run(f"xdotool mousemove --sync {x} {y}"); time.sleep(0.06)
    run("xdotool click 1")
    return {"ok": True, "x": x, "y": y}

def tool_ground_and_click(args: Dict[str, Any]) -> Dict[str, Any]:
    q = (args.get("query") or "").strip()
    try:
        x, y, overlay_path = showui_ground(q)
    except Exception as e:
        _log({"ts": time.time(), "type": "ground_and_click_error", "q": q, "error": str(e)})
        print(f"[overlay] saved -> None  (ok=False, reason={e})")
        return {"ok": False, "reason": str(e), "overlay_png": None}
    res = _safe_click(x, y, reason=f"ground:{q}")
    print(f"[ground_and_click] query='{q}' -> ({x},{y}) overlay={overlay_path}  (ok={res.get('ok')}, reason={res.get('reason')})")
    return {"ok": res.get("ok", False), "x": x, "y": y, "overlay_png": overlay_path, "query": q, "reason": res.get("reason")}

def tool_type_text(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args.get("text", "")
    run(f"xdotool type --delay 0 -- {shlex.quote(text)}")
    return {"ok": True, "n": len(text), "text": text}

def tool_press_key(args: Dict[str, Any]) -> Dict[str, Any]:
    key = (args.get("key") or "").strip()
    keymap = {
        "enter": "Return", "return": "Return",
        "esc": "Escape", "escape": "Escape",
        "tab": "Tab",
        "ctrl+l": "ctrl+l", "ctrl+L": "ctrl+l"
    }
    ks = keymap.get(key, key)
    run(f"xdotool key {shlex.quote(ks)}")
    return {"ok": True, "key": key}

def tool_sleep(args: Dict[str, Any]) -> Dict[str, Any]:
    ms = float(args.get("ms", 200))
    t0 = time.time()
    time.sleep(ms/1000.0)
    return {"ok": True, "slept_ms": (time.time()-t0)*1000.0}

def _focus_window(candidates: List[str]) -> bool:
    if not candidates:
        return False
    for cand in candidates:
        run(f"xdotool search --onlyvisible --class {shlex.quote(cand)} windowactivate --sync"); time.sleep(0.12)
        run(f"xdotool search --onlyvisible --name {shlex.quote(cand)} windowactivate --sync"); time.sleep(0.12)
    return True

def tool_focus_window(args: Dict[str, Any]) -> Dict[str, Any]:
    cands = args.get("candidates") or []
    ok = _focus_window(cands)
    return {"ok": ok, "candidates": cands}

def tool_run_shell(args: Dict[str, Any]) -> Dict[str, Any]:
    cmd = args.get("cmd", "")
    out = run(cmd)
    return {"ok": True, "stdout": out[:8000]}

def tool_open_app(args: Dict[str, Any]) -> Dict[str, Any]:
    name = (args.get("name") or "").strip().lower()
    cmd = args.get("cmd")
    focus = args.get("focus") or []
    """
    if name == "firefox" and not cmd:
        cmd = "firefox &"; focus = ["Navigator", "firefox", "Mozilla Firefox", "Firefox"]
        """
    if name in {"files", "nautilus"} and not cmd:
        cmd = "nautilus --new-window &"; focus = ["org.gnome.Nautilus", "Nautilus", "Files"]
    if name in {"terminal", "gnome-terminal"} and not cmd:
        cmd = "gnome-terminal &"; focus = ["Gnome-terminal", "Terminal"]
    if name in {"subl", "sublime", "sublime_text"} and not cmd:
        cmd = "subl &"; focus = ["sublime_text", "Sublime Text"]

    if not cmd:
        return {"ok": False, "error": "no command to open"}

    run(cmd); time.sleep(0.9); _focus_window(focus)
    return {"ok": True, "started": cmd, "focused": focus}

TOOLS = [
    {"type": "function", "function": {
        "name": "ground_and_click",
        "description": "Find a UI element on the current screenshot and left-click it.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "type_text",
        "description": "Type raw text as keyboard input.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    }},
    {"type": "function", "function": {
        "name": "press_key",
        "description": "Press a key (e.g. Return, Escape, Ctrl+l).",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}
    }},
    {"type": "function", "function": {
        "name": "sleep",
        "description": "Sleep for milliseconds.",
        "parameters": {"type": "object", "properties": {"ms": {"type": "number"}}, "required": ["ms"]}
    }},
    {"type": "function", "function": {
        "name": "focus_window",
        "description": "Activate a window by class or title candidates.",
        "parameters": {"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "string"}}}, "required": ["candidates"]}
    }},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command and return stdout.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}
    }},
    {"type": "function", "function": {
        "name": "open_app",
        "description": "Start an app focus it.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "cmd": {"type": "string"},
            "focus": {"type": "array", "items": {"type": "string"}}
        }}
    }},
]

# --------------- Planner + Guardrails ----------------------------------------

AUTO_OPEN_APPS = True

def _route_hint(user_cmd: str) -> Dict[str, Any]:
    s = user_cmd.lower()
    if any(k in s for k in ["files", "file manager", "nautilus", "dosyalar"]):
        return {"app": "files", "focus": ["org.gnome.Nautilus", "Nautilus", "Files"]}
        """
    if any(k in s for k in ["browser", "address bar", "search bar", "youtube", "google", "tab", "mozilla"]):
        return {"app": "firefox", "focus": ["Navigator", "Mozilla Firefox", "Firefox"]}"""
    if any(k in s for k in ["sublime", "subl"]):
        return {"app": "sublime", "focus": ["sublime_text", "Sublime Text"]}
    if "terminal" in s:
        return {"app": "terminal", "focus": ["Gnome-terminal", "Terminal"]}
    return {}

def _extract_click_target(s: str) -> Optional[str]:
    m = re.search(r"\bclick\b\s+(.*)", s, re.I)
    if not m: return None
    tail = m.group(1).strip()
    words = tail.split()
    if len(words) > 8: tail = " ".join(words[:8])
    return tail

SYSTEM_PROMPT = (
    "You are a planner for a computer-use agent on Linux (X11).\n"
    "You MUST solve the user's instruction by calling tools only. Never output plain text.\n"
    "Tools: open_app, focus_window, ground_and_click, type_text, press_key, sleep, run_shell.\n"
    "For clicking UI elements, call ground_and_click with a concise target description.\n"
    "For typing a URL: focus address bar (by click or Ctrl+l), then type_text, then press_key('Return').\n"
)

def _planner(messages: List[Dict[str, Any]], add_tools=True) -> Dict[str, Any]:
    if add_tools:
        return planner.create_chat_completion(messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.2)
    else:
        return planner.create_chat_completion(messages=messages, temperature=0.2)

def _expand_and_execute(toolcalls: List[Dict[str, Any]]):
    for tc in toolcalls:
        name = tc["function"]["name"] if "function" in tc else tc.get("name")
        args_raw = tc["function"].get("arguments", "{}") if "function" in tc else tc.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            args = json.loads(args_raw.strip())
        res = {"ok": False}
        try:
            res = call_tool(name, args)
        finally:
            _log({"ts": time.time(), "type": "tool_result", "name": name, "args": args, "result": res})
            if name == "ground_and_click":
                print(f"[overlay] saved -> {res.get('overlay_png')}  (ok={res.get('ok')}, reason={res.get('reason')})")

def _append_guardrails(user_cmd: str, plan_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    steps = list(plan_calls)
    s = user_cmd.lower()
    hint = _route_hint(user_cmd)

    if hint.get("app"):
        if AUTO_OPEN_APPS:
            steps.insert(0, {"type": "function", "function": {"name": "open_app",
                        "arguments": json.dumps({"name": hint["app"]})}})
        steps.insert(1, {"type": "function", "function": {"name": "focus_window",
                        "arguments": json.dumps({"candidates": hint.get("focus", [])})}})

    if "click" in s:
        target = _extract_click_target(user_cmd) or "target"
        steps.append({"type": "function", "function": {"name": "ground_and_click",
                        "arguments": json.dumps({"query": target})}})
        steps.append({"type": "function", "function": {"name": "sleep",
                        "arguments": json.dumps({"ms": 180})}})

    if any(k in s for k in ["youtube", "address bar", "search bar", "url bar"]):
        steps.append({"type": "function", "function": {"name": "ground_and_click",
                        "arguments": json.dumps({"query": "Address Bar in Firefox"})}})
        steps.append({"type": "function", "function": {"name": "sleep",
                        "arguments": json.dumps({"ms": 150})}})
        m = re.search(r"(https?://\S+|[a-z0-9\.\-]+\.com\b)", s)
        if m:
            url = m.group(1)
            steps.append({"type": "function", "function": {"name": "type_text",
                            "arguments": json.dumps({"text": url})}})
            steps.append({"type": "function", "function": {"name": "press_key",
                            "arguments": json.dumps({"key": "Return"})}})
            steps.append({"type": "function", "function": {"name": "sleep",
                            "arguments": json.dumps({"ms": 350})}})
        else:
            steps.append({"type": "function", "function": {"name": "press_key",
                            "arguments": json.dumps({"key": "Ctrl+l"})}})

    return steps

def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "ground_and_click":
        return tool_ground_and_click(args)
    if name == "type_text":
        return tool_type_text(args)
    if name == "press_key":
        return tool_press_key(args)
    if name == "sleep":
        return tool_sleep(args)
    if name == "focus_window":
        return tool_focus_window(args)
    if name == "run_shell":
        return tool_run_shell(args)
    if name == "open_app":
        return tool_open_app(args)
    return {"ok": False, "error": f"unknown tool: {name}"}

# --------------- Entry --------------------------------------------------------

def run_planner(instruction: str) -> str:
    _log({"ts": time.time(), "type": "user_cmd", "cmd": instruction})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    out = _planner(messages, add_tools=True)
    msg = out["choices"][0]["message"]

    steps = []
    if "tool_calls" in msg and msg["tool_calls"]:
        steps = msg["tool_calls"]
    else:
        content = (msg.get("content") or "").strip()
        if content:
            pat = re.compile(r'([a-zA-Z_]\w*)\((?P<q>[\'"])(.*?)\1\)')
            for m in pat.finditer(content):
                func = m.group(1); arg = m.group(3)
                steps.append({"type": "function", "function": {
                    "name": func,
                    "arguments": json.dumps({"query" if func=="ground_and_click" else "text": arg})
                }})

    steps2 = _append_guardrails(instruction, steps)

    print("\n===== PLAN (model tool_calls) =====")
    try:
        print(json.dumps(steps, indent=2))
    except Exception:
        print(steps)
    print("===== END PLAN =====\n")

    _expand_and_execute(steps2)
    return "Done."
