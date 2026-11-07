from fake_nvml import FakeNVML, MiB

from ai_health.collectors.gpu import GpuCollector


def test_sample_reads_memory_and_util():
    g = GpuCollector(nvml=FakeNVML())
    [s] = g.sample()
    assert g.available and g.reason is None
    assert s.name == "NVIDIA GeForce RTX 3060"
    assert s.mem_total == 12288 * MiB
    assert s.mem_used == 9000 * MiB
    assert s.util_gpu == 57 and s.util_mem == 31
    assert s.temp_c == 64
    assert s.power_w == 142.0 and s.power_limit_w == 170.0
    assert s.sm_clock_mhz == 1777 and s.mem_clock_mhz == 7300


def test_processes_compute_and_graphics_merged():
    nv = FakeNVML(compute={0: [(10, 5000 * MiB), (11, 100 * MiB)]},
                  graphics={0: [(11, 300 * MiB), (12, 50 * MiB)]})
    [s] = GpuCollector(nvml=nv).sample()
    by_pid = {p.pid: p.used_bytes for p in s.processes}
    assert by_pid == {10: 5000 * MiB, 11: 300 * MiB, 12: 50 * MiB}
    assert s.proc_source == "nvml"


def test_nvml_init_failure_returns_unavailable():
    g = GpuCollector(nvml=FakeNVML(fail_init="Driver Not Loaded"))
    assert g.available is False
    assert "Driver Not Loaded" in g.reason
    assert g.sample() == []


def test_falls_back_to_nvidia_smi():
    nv = FakeNVML(compute={0: [(1234, None)]})
    calls = []

    def smi(argv):
        calls.append(argv)
        return "1234, 2048\n"

    [s] = GpuCollector(nvml=nv, smi_runner=smi).sample()
    assert s.processes[0].pid == 1234
    assert s.processes[0].used_bytes == 2048 * MiB
    assert s.proc_source == "nvidia-smi"
    assert "--query-compute-apps=pid,used_memory" in calls[0]


def test_smi_failure_leaves_unknown():
    nv = FakeNVML(compute={0: [(1234, None)]})

    def smi(argv):
        raise OSError("nvidia-smi missing")

    [s] = GpuCollector(nvml=nv, smi_runner=smi).sample()
    assert s.processes[0].used_bytes is None
    assert s.proc_source == "none"


def test_optional_fields_none_on_not_supported():
    nv = FakeNVML(unsupported={"power", "temp", "clock", "util"})
    [s] = GpuCollector(nvml=nv).sample()
    assert s.power_w is None and s.power_limit_w is None
    assert s.temp_c is None and s.sm_clock_mhz is None and s.util_gpu is None
    assert s.mem_used == 9000 * MiB


def test_close_shuts_down():
    nv = FakeNVML()
    g = GpuCollector(nvml=nv)
    g.close()
    assert nv.shutdown_called
