import asyncio
import os
import signal
import subprocess
from dataclasses import dataclass

@dataclass
class ProcessHandle:
    pid: int
    proc: subprocess.Popen

    async def wait(self):
        if self.proc.poll() is None:
            await asyncio.get_event_loop().run_in_executor(None, self.proc.wait)

class Commands:
    def run(self, cmd: str, background: bool=False):
        if background:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            return ProcessHandle(pid=proc.pid, proc=proc)
        else:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            return out.decode("utf-8", "ignore")

class LocalSandbox:
    """Drop-in replacement for e2b_desktop.Sandbox used by streaming.py.
    Provides .commands, .kill(), etc., but runs on the local machine.
    """
    def __init__(self):
        self.commands = Commands()
        self.process = None

    def kill(self):
        if self.process and self.process.proc and (self.process.proc.poll() is None):
            try:
                os.killpg(os.getpgid(self.process.proc.pid), signal.SIGTERM)
            except Exception:
                pass
            self.process = None