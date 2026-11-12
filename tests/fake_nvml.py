"""Minimal stand-in for the subset of ``pynvml`` used by the GPU collector."""

from types import SimpleNamespace

MiB = 2**20


class NVMLError(Exception):
    pass


class NVMLError_NotSupported(NVMLError):
    pass


class FakeNVML:
    NVMLError = NVMLError
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2

    def __init__(self, devices=None, compute=None, graphics=None, fail_init=None,
                 unsupported=()):
        self.devices = devices if devices is not None else [dict(
            name="NVIDIA GeForce RTX 3060", total=12288 * MiB, used=9000 * MiB,
            util=57, util_mem=31, temp=64, power=142_000, limit=170_000,
            sm=1777, mem=7300)]
        self.compute = compute or {}
        self.graphics = graphics or {}
        self.fail_init = fail_init
        self.unsupported = set(unsupported)
        self.shutdown_called = False

    def _maybe(self, key):
        if key in self.unsupported:
            raise NVMLError_NotSupported("Not Supported")

    def nvmlInit(self):
        if self.fail_init:
            raise NVMLError(self.fail_init)

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlDeviceGetCount(self):
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetUUID(self, h):
        return self.devices[h].get("uuid", f"GPU-fake-{h}")

    def nvmlDeviceGetName(self, h):
        return self.devices[h]["name"]

    def nvmlDeviceGetMemoryInfo(self, h):
        d = self.devices[h]
        return SimpleNamespace(total=d["total"], used=d["used"], free=d["total"] - d["used"])

    def nvmlDeviceGetUtilizationRates(self, h):
        self._maybe("util")
        d = self.devices[h]
        return SimpleNamespace(gpu=d["util"], memory=d["util_mem"])

    def nvmlDeviceGetTemperature(self, h, sensor):
        self._maybe("temp")
        return self.devices[h]["temp"]

    def nvmlDeviceGetPowerUsage(self, h):
        self._maybe("power")
        return self.devices[h]["power"]

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        self._maybe("power")
        return self.devices[h]["limit"]

    def nvmlDeviceGetClockInfo(self, h, kind):
        self._maybe("clock")
        return self.devices[h]["sm" if kind == self.NVML_CLOCK_SM else "mem"]

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [SimpleNamespace(pid=p, usedGpuMemory=m) for p, m in self.compute.get(h, [])]

    def nvmlDeviceGetGraphicsRunningProcesses(self, h):
        return [SimpleNamespace(pid=p, usedGpuMemory=m) for p, m in self.graphics.get(h, [])]
