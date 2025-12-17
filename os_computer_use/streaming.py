# os_computer_use/streaming.py
import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class _Proc:
    proc: subprocess.Popen


class _Commands:
    def run(self, cmd: str, background: bool = False) -> _Proc:
        if background:
            # process group ile başlat; stop_stream'te grupça SIGTERM göndereceğiz
            p = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            return _Proc(proc=p)
        else:
            subprocess.check_call(cmd, shell=True)
            return _Proc(proc=None)


class Sandbox:
    """
    Basit ekran yayını helper'ı.
    start_stream() -> HTTP (mpegts, listen=1) + output.mp4; stop_stream() -> ffmpeg'i kapatır.
    """
    def __init__(self) -> None:
        self.commands = _Commands()
        self.process: Optional[_Proc] = None

    def get_host(self, port: int) -> str:
        return f"127.0.0.1:{port}"

    def start_stream(self) -> str:
        display = os.environ.get("DISPLAY", ":0")
        log_path = "ffmpeg_stream.log"

        # Önemli:
        #  - x11grab ile DISPLAY'ten yakala
        #  - tee: aynı anda MP4 dosyası + HTTP (mpegts)
        #  - listen=1 -> HTTP server
        #  - onfail=ignore -> HTTP çıkışı başarısız olsa bile dosyaya yazmaya devam et
        #  - -map 0:v:0 -> tee için explicit stream seçimi şart
        cmd = (
            f'ffmpeg -nostdin -loglevel error '
            f'-f x11grab -framerate 30 -i "{display}" '
            f'-an -map 0:v:0 '
            f'-c:v libx264 -preset ultrafast -tune zerolatency '
            f'-f tee "[f=mp4]output.mp4|[onfail=ignore:f=mpegts]http://127.0.0.1:8080?listen=1" '
            f'> {log_path} 2>&1'
        )
        proc = self.commands.run(cmd, background=True)
        self.process = proc
        return f"http://{self.get_host(8080)}"

    def stop_stream(self) -> None:
        if self.process and self.process.proc and self.process.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.proc.pid), signal.SIGTERM)
            except Exception:
                pass
        self.process = None


class DisplayClient:
    """
    Kayıt sonrası output.mp4 dosyasını kontrol eder (tee zaten mp4 üretti).
    """
    def save_stream(self) -> None:
        if not os.path.exists("output.mp4"):
            print("No output.mp4 found; nothing to save.")
            return
        print("Stream saved successfully as mp4.")
