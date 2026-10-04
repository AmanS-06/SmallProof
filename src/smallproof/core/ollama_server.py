"""Runs the Ollama server only for the length of a job, and always stops it.

    with OllamaServer.from_config(config):
        ...  # SLM calls

- Models are stored in the project (OLLAMA_MODELS is set for this process only).
- OLLAMA_KEEP_ALIVE unloads the model from the GPU soon after the last call.
- A watchdog kills the server after max_server_minutes, whatever happens.
- A server that was already running is used as is and left alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from smallproof.core.config import resolve_path


def _tie_to_this_process(pid: int):
    """Windows: put the server in a job object that the OS kills when this Python process ends,
    even if it is force-killed (so the watchdog thread never ran). Returns the job handle, which
    must stay referenced while the server should live."""
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("r", "w", "o", "rb", "wb", "ob")]

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))  # 9: extended limits
    process = kernel32.OpenProcess(0x1F0FFF, False, pid)  # PROCESS_ALL_ACCESS
    kernel32.AssignProcessToJobObject(job, process)
    kernel32.CloseHandle(process)
    return job


class OllamaServer:
    def __init__(self, exe: str | Path, models_dir: str | Path, host: str = "127.0.0.1:11434",
                 keep_alive: str = "1m", max_server_minutes: float = 25, log_file: str | Path | None = None) -> None:
        self.exe, self.models_dir, self.host = str(exe), str(models_dir), host
        self.keep_alive, self.max_seconds = keep_alive, max_server_minutes * 60
        self.log_file = Path(log_file) if log_file else None
        self.process: subprocess.Popen | None = None
        self._watchdog: threading.Timer | None = None

    @classmethod
    def from_config(cls, config: dict) -> OllamaServer:
        settings = config["runtime"]["ollama"]
        host = config["generator"]["host"].removeprefix("http://")
        return cls(resolve_path(config, settings["exe"]), resolve_path(config, settings["models_dir"]), host,
                   settings.get("keep_alive", "1m"), settings.get("max_server_minutes", 25),
                   resolve_path(config, config["paths"]["data_dir"]) / "logs" / "ollama_serve.log")

    def is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://{self.host}/api/version", timeout=1):
                return True
        except OSError:
            return False

    def start(self) -> None:
        if self.is_running():
            return  # someone else's server: use it, do not stop it later
        env = {**os.environ, "OLLAMA_MODELS": self.models_dir, "OLLAMA_HOST": self.host,
               "OLLAMA_KEEP_ALIVE": self.keep_alive, "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NUM_PARALLEL": "1"}
        output = subprocess.DEVNULL
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            output = self.log_file.open("a", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen([self.exe, "serve"], env=env, stdout=output, stderr=subprocess.STDOUT,
                                        creationflags=flags)
        if sys.platform == "win32":  # the server and its runner die with this process, however it ends
            self._job = _tie_to_this_process(self.process.pid)
        self._watchdog = threading.Timer(self.max_seconds, self.stop)
        self._watchdog.daemon = True
        self._watchdog.start()
        deadline = time.monotonic() + 30
        while not self.is_running():
            if time.monotonic() > deadline or self.process.poll() is not None:
                self.stop()
                raise RuntimeError(f"Ollama server did not start. See {self.log_file}")
            time.sleep(0.5)

    def stop(self) -> None:
        """Stop the server and its model runner (the whole process tree)."""
        if self._watchdog:
            self._watchdog.cancel()
        if self.process and self.process.poll() is None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True)
            else:
                self.process.terminate()
            self.process.wait(timeout=15)
        self.process = None

    def __enter__(self) -> OllamaServer:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


class OnDemandOllama:
    """For long-running apps (API, demo): start the server on the first request, stop it after idle_minutes.

    The server's own watchdog still stops it after max_server_minutes; the next request starts it again.
    """

    def __init__(self, config: dict, idle_minutes: float = 5) -> None:
        self.config, self.idle_seconds = config, idle_minutes * 60
        self.server: OllamaServer | None = None
        self.last_used = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True).start()

    def ensure(self) -> None:
        with self._lock:
            if self.server is None or not self.server.is_running():
                self.server = OllamaServer.from_config(self.config)
                self.server.start()
            self.last_used = time.monotonic()

    def _reaper(self) -> None:
        while True:
            time.sleep(30)
            with self._lock:
                if self.server and time.monotonic() - self.last_used > self.idle_seconds:
                    self.server.stop()
                    self.server = None
