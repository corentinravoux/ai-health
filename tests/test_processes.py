import psutil
from fake_procs import FakeProc

from ai_health.collectors.processes import find_llama_processes, is_llama_process

BIN = "/home/u/llama.cpp/build/bin/"


def test_is_llama_process():
    assert is_llama_process("llama-server", [BIN + "llama-server", "-m", "x"])
    assert is_llama_process("llama-cli", ["llama-cli"])
    assert is_llama_process("llama", [BIN + "llama"])
    assert is_llama_process("llama-bench", ["./llama-bench"])
    assert not is_llama_process("python3", ["python3", "my_llama_script.py"])
    assert not is_llama_process("ollama", ["ollama", "serve"])
    assert not is_llama_process("bash", [])


def test_detects_server_and_cli():
    procs = [FakeProc(10, 1, "llama-server", [BIN + "llama-server", "-m", "/m/a.gguf", "-ngl", "20"]),
             FakeProc(11, 1, "llama-cli", [BIN + "llama-cli", "-m", "/m/b.gguf"]),
             FakeProc(12, 1, "python3", ["python3", "my_llama_script.py"])]
    found = {p.pid: p for p in find_llama_processes(procs)}
    assert set(found) == {10, 11}
    assert found[10].role == "server" and found[10].args.ngl == 20
    assert found[11].role == "cli"
    assert found[10].rss == 100 * 2**20 and found[10].threads == 8


def test_router_child_role():
    procs = [FakeProc(20, 1, "llama-server", [BIN + "llama-server", "--models-dir", "/m"]),
             FakeProc(21, 20, "llama-server", [BIN + "llama-server", "-m", "/m/a.gguf", "--port", "40001"])]
    found = {p.pid: p for p in find_llama_processes(procs)}
    assert found[20].role == "router"
    assert found[21].role == "router-child"
    assert found[20].children == [21]


def test_env_flags_merged():
    procs = [FakeProc(30, 1, "llama-server", [BIN + "llama-server", "-m", "x"],
                      environ={"LLAMA_ARG_N_GPU_LAYERS": "7", "HOME": "/home/u"})]
    [p] = find_llama_processes(procs)
    assert p.args.ngl == 7 and p.args.sources["ngl"] == "env"


def test_access_denied_is_skipped():
    procs = [FakeProc(40, 1, "llama-server", [BIN + "llama-server", "-m", "x", "-ngl", "3"],
                      deny_environ=True, has_full_info=False)]
    [p] = find_llama_processes(procs)
    assert p.args.ngl == 3
    assert p.swap is None


def test_swap_from_memory_full_info():
    [p] = find_llama_processes([FakeProc(50, 1, "llama-server", [BIN + "llama-server"], swap=2**20)])
    assert p.swap == 2**20


def test_vanished_process_is_skipped():
    class Gone(FakeProc):
        def cmdline(self):
            raise psutil.NoSuchProcess(self.pid)

    procs = [Gone(60, 1, "llama-server", []), FakeProc(61, 1, "llama-cli", ["llama-cli"])]
    assert [p.pid for p in find_llama_processes(procs)] == [61]


def test_name_gate_skips_cmdline_of_unrelated_processes():
    class NoCmdline(FakeProc):
        def cmdline(self):
            raise AssertionError("cmdline read for an unrelated process")

    procs = [NoCmdline(70, 1, "bash", ["bash"]), FakeProc(71, 1, "llama-server", [BIN + "llama-server"])]
    assert [p.pid for p in find_llama_processes(procs)] == [71]


def test_port_from_listening_socket_when_not_given():
    # llama.cpp's default port changed across versions (8080 -> 9931): trust the socket
    procs = [FakeProc(80, 1, "llama-server", [BIN + "llama-server", "-m", "x"], listen=[("127.0.0.1", 9931)])]
    [p] = find_llama_processes(procs)
    assert p.args.port == 9931 and p.args.sources["port"] == "socket"
    assert p.listen_ports == [9931]


def test_explicit_port_wins_over_socket():
    procs = [FakeProc(81, 1, "llama-server", [BIN + "llama-server", "-m", "x", "--port", "7000"],
                      listen=[("127.0.0.1", 7000), ("127.0.0.1", 7001)])]
    [p] = find_llama_processes(procs)
    assert p.args.port == 7000 and p.args.sources["port"] == "cli"
