"""NVIDIA GPU sampling through NVML, with an ``nvidia-smi`` fallback for
per-process memory when the driver does not report it."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable

MiB = 2**20
SMI_ARGV = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits"]


@dataclass
class GpuProcess:
    pid: int
    used_bytes: int | None
    name: str | None = None


@dataclass
class GpuSample:
    index: int
    name: str
    mem_total: int
    mem_used: int
    mem_reserved: int | None
    util_gpu: int | None
    util_mem: int | None
    temp_c: int | None
    power_w: float | None
    power_limit_w: float | None
    sm_clock_mhz: int | None
    mem_clock_mhz: int | None
    processes: list[GpuProcess] = field(default_factory=list)
    proc_source: str = "none"  # "nvml" | "nvidia-smi" | "none"


def _run_smi(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=1.0,
                          check=True).stdout


class GpuCollector:
    """Samples every NVIDIA device. ``available`` is False (with ``reason``)
    when NVML cannot be initialised; ``sample()`` then returns ``[]``."""

    def __init__(self, nvml=None, smi_runner: Callable[[list[str]], str] | None = None):
        self.available = False
        self.reason: str | None = None
        self._smi = smi_runner or _run_smi
        if nvml is None:
            try:
                import pynvml as nvml  # type: ignore[no-redef]
            except ImportError as e:  # pragma: no cover - depends on install
                self.reason = f"nvidia-ml-py not installed: {e}"
                self._nv = None
                return
        self._nv = nvml
        try:
            nvml.nvmlInit()
            self._handles = [nvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(nvml.nvmlDeviceGetCount())]
            self.available = True
        except Exception as e:  # NVMLError and friends
            self.reason = str(e) or type(e).__name__

    def _opt(self, fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    def _processes(self, h) -> tuple[list[GpuProcess], str]:
        nv = self._nv
        merged: dict[int, int | None] = {}
        for getter in (nv.nvmlDeviceGetComputeRunningProcesses,
                       nv.nvmlDeviceGetGraphicsRunningProcesses):
            for p in self._opt(getter, h) or []:
                used = getattr(p, "usedGpuMemory", None)
                prev = merged.get(p.pid)
                if used is None:
                    merged.setdefault(p.pid, prev)
                else:
                    merged[p.pid] = max(used, prev or 0)
        procs = [GpuProcess(pid, used) for pid, used in merged.items()]
        if not procs or all(p.used_bytes is not None for p in procs):
            return procs, "nvml" if procs else "none"
        # nvidia-smi lists every GPU's processes: keep only this device's rows,
        # otherwise a PID on two GPUs gets the last row's allocation on both
        uuid = self._opt(nv.nvmlDeviceGetUUID, h)
        if isinstance(uuid, bytes):
            uuid = uuid.decode()
        if not uuid:
            return procs, "none"
        try:
            smi = {}
            for line in self._smi(SMI_ARGV).splitlines():
                parts = [x.strip() for x in line.split(",")]
                if (len(parts) == 3 and parts[0] == uuid
                        and parts[1].isdigit() and parts[2].isdigit()):
                    smi[int(parts[1])] = int(parts[2]) * MiB
        except Exception:
            return procs, "none"
        for p in procs:
            if p.used_bytes is None and p.pid in smi:
                p.used_bytes = smi[p.pid]
        return procs, "nvidia-smi"

    def sample(self) -> list[GpuSample]:
        if not self.available:
            return []
        try:
            return self._sample()
        except Exception as e:  # GPU lost, driver/library mismatch after an upgrade
            self.available = False
            self.reason = f"NVML stopped responding: {e}"
            return []

    def _sample(self) -> list[GpuSample]:
        nv = self._nv
        out = []
        for i, h in enumerate(self._handles):
            name = nv.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            util = self._opt(nv.nvmlDeviceGetUtilizationRates, h)
            power = self._opt(nv.nvmlDeviceGetPowerUsage, h)
            limit = self._opt(nv.nvmlDeviceGetEnforcedPowerLimit, h)
            procs, source = self._processes(h)
            out.append(GpuSample(
                index=i, name=name, mem_total=int(mem.total), mem_used=int(mem.used),
                mem_reserved=getattr(mem, "reserved", None),
                util_gpu=util.gpu if util else None,
                util_mem=util.memory if util else None,
                temp_c=self._opt(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU),
                power_w=power / 1000.0 if power is not None else None,
                power_limit_w=limit / 1000.0 if limit is not None else None,
                sm_clock_mhz=self._opt(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM),
                mem_clock_mhz=self._opt(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM),
                processes=procs, proc_source=source))
        return out

    def close(self) -> None:
        if self.available:
            self._opt(self._nv.nvmlShutdown)
            self.available = False
