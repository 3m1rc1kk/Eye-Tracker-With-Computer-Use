
import os
import sys
import time


SYSTEM_PYTHON = "/usr/bin/python3"

if sys.executable != SYSTEM_PYTHON:
    print(
        f"[AT-SPI helper] reexec: {sys.executable} -> {SYSTEM_PYTHON}",
        file=sys.stderr,
    )
    os.execv(SYSTEM_PYTHON, [SYSTEM_PYTHON] + sys.argv)

print(f"[AT-SPI helper] python: {sys.executable}", file=sys.stderr)

try:
    import pyatspi
except Exception as e:
    print("[AT-SPI helper] pyatspi import failed:", repr(e), file=sys.stderr)
    raise SystemExit(1)

FLAG_PATH = "/tmp/eye_keyboard_request"
_last_set = 0.0
MIN_INTERVAL = 0.5  


def write_flag():

    global _last_set
    now = time.time()
    if now - _last_set < MIN_INTERVAL:
        return
    _last_set = now

    try:
        with open(FLAG_PATH, "w") as f:
            f.write("keyboard\n")
        print(f"[AT-SPI helper] flag set -> {FLAG_PATH}")
    except Exception as e:
        print("[AT-SPI helper] flag write error:", e)


def focus_event_listener(event):

    try:
        acc = event.source
        if acc is None:
            return

        state = acc.getState()
        if not state.contains(pyatspi.STATE_FOCUSED):
            return

        editable = state.contains(pyatspi.STATE_EDITABLE)
        role_name = acc.getRoleName().lower()

        if editable and role_name in ("text", "entry", "password text"):
            name = acc.name or ""
            print(
                f"[AT-SPI helper] editable focus on '{name}' "
                f"(role={role_name}) -> request keyboard"
            )
            write_flag()

    except Exception as e:
        print("[AT-SPI helper] listener error:", e)


def main():
    print("[AT-SPI helper] starting...")
    pyatspi.Registry.registerEventListener(
        focus_event_listener, "object:state-changed:focused"
    )
    print("[AT-SPI helper] listener registered, entering main loop...")
    pyatspi.Registry.start()
    print("[AT-SPI helper] main loop ended")


if __name__ == "__main__":
    main()
