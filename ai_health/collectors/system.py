"""System CPU / RAM / swap sampling (psutil, non-blocking)."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass
class SystemSample:
    cpu_percent: float
    per_cpu: list[float]
    load1: float
    ram_total: int
    ram_used: int  # total - available (what is really taken, cache excluded)
    ram_available: int
    swap_total: int
    swap_used: int


def system_sample(ps=psutil) -> SystemSample:
    vm = ps.virtual_memory()
    sw = ps.swap_memory()
    return SystemSample(
        cpu_percent=float(ps.cpu_percent(interval=None)),
        per_cpu=[float(x) for x in ps.cpu_percent(interval=None, percpu=True)],
        load1=float(ps.getloadavg()[0]),
        ram_total=int(vm.total),
        ram_used=int(vm.total - vm.available),
        ram_available=int(vm.available),
        swap_total=int(sw.total),
        swap_used=int(sw.used),
    )
