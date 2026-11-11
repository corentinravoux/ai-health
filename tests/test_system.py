from types import SimpleNamespace

from ai_health.collectors.system import system_sample


class StubPs:
    def cpu_percent(self, interval=None, percpu=False):
        assert interval is None  # never block the sampler
        return [10.0, 30.0] if percpu else 20.0

    def virtual_memory(self):
        return SimpleNamespace(total=16 * 2**30, available=6 * 2**30)

    def swap_memory(self):
        return SimpleNamespace(total=18 * 2**30, used=2**30)

    def getloadavg(self):
        return (1.5, 1.0, 0.5)


def test_system_sample_shapes():
    s = system_sample(StubPs())
    assert s.cpu_percent == 20.0
    assert s.per_cpu == [10.0, 30.0]
    assert s.load1 == 1.5
    assert s.ram_total == 16 * 2**30
    assert s.ram_available == 6 * 2**30
    assert s.ram_used == 10 * 2**30  # total - available
    assert s.swap_total == 18 * 2**30 and s.swap_used == 2**30
