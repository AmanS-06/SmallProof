"""Keeps long jobs from overheating the laptop.

CPU temperature cannot be read on this Windows laptop without admin rights,
so the CPU is protected by limits instead of readings:
- a thread cap for torch (runtime.torch_threads, applied in core.config)
- a duty cycle: after work_seconds of work, rest for rest_seconds
- a hard time limit per job (max_job_minutes); jobs stop cleanly and resume
The GPU is protected by its own sensor (NVML): above max_gpu_c the job pauses
until the GPU is back under resume_gpu_c.

Usage: call guard.checkpoint() between units of work (a document, a batch, a
question). It returns False when the time limit is reached, and the job stops.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class ThermalGuard:
    def __init__(self, max_gpu_c: int = 80, resume_gpu_c: int = 70, work_seconds: float = 60,
                 rest_seconds: float = 30, max_job_minutes: float = 25, log: Callable[[str], None] = print) -> None:
        self.max_gpu_c, self.resume_gpu_c = max_gpu_c, resume_gpu_c
        self.work_seconds, self.rest_seconds = work_seconds, rest_seconds
        self.max_job_seconds = max_job_minutes * 60
        self.log = log
        self.started = self.work_started = time.monotonic()
        self.rested_seconds = 0.0
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml, self._handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # no NVIDIA GPU or driver: only the CPU limits apply
            self._nvml = None

    @classmethod
    def from_config(cls, config: dict, log: Callable[[str], None] = print) -> ThermalGuard:
        return cls(**config.get("runtime", {}).get("thermal", {}), log=log)

    def gpu_temp(self) -> int | None:
        if self._nvml is None:
            return None
        return self._nvml.nvmlDeviceGetTemperature(self._handle, self._nvml.NVML_TEMPERATURE_GPU)

    def _rest(self, seconds: float) -> None:
        time.sleep(seconds)
        self.rested_seconds += seconds

    def checkpoint(self) -> bool:
        """Rest if due, wait for the GPU to cool if needed. False means: stop, the time limit is reached."""
        now = time.monotonic()
        if now - self.started >= self.max_job_seconds:
            self.log(f"Time limit of {self.max_job_seconds / 60:.0f} min reached: stopping. Run again to resume.")
            return False
        if now - self.work_started >= self.work_seconds:
            self._rest(self.rest_seconds)
            self.work_started = time.monotonic()
        temp = self.gpu_temp()
        if temp is not None and temp >= self.max_gpu_c:
            self.log(f"GPU at {temp} C: pausing until it is below {self.resume_gpu_c} C")
            while temp is not None and temp > self.resume_gpu_c:
                self._rest(5)
                temp = self.gpu_temp()
            self.work_started = time.monotonic()
        return True

    def close(self) -> None:
        if self._nvml is not None:
            self._nvml.nvmlShutdown()
            self._nvml = None

    def __enter__(self) -> ThermalGuard:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
