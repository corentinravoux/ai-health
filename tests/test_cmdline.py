import pytest

from ai_health.llama.cmdline import key_flags_summary, parse_llama_args

SRV = "/home/u/llama.cpp/build/bin/llama-server"


def p(*args, env=None):
    return parse_llama_args([SRV, *args], env)


def test_basic_server_flags():
    a = p("-m", "/m/q.gguf", "-ngl", "20", "-c", "8192", "--port", "9000", "--metrics")
    assert a.kind == "server"
    assert a.model == "/m/q.gguf"
    assert a.ngl == 20 and a.ctx == 8192 and a.port == 9000 and a.metrics
    assert a.sources["ngl"] == "cli"
    assert a.router is False


def test_defaults():
    a = p("-m", "x.gguf")
    assert a.ngl == "auto" and a.sources["ngl"] == "default"
    assert a.fit is True and a.host == "127.0.0.1" and a.port == 8080
    assert a.cache_type_k == "f16" and a.cache_type_v == "f16"
    assert a.flash_attn == "auto" and a.kv_offload is True
    assert a.metrics is False and a.slots is True
    assert a.overrides == [] and a.n_cpu_moe == 0 and a.cpu_moe is False


@pytest.mark.parametrize("args,value", [
    (["-ngl", "all"], "all"), (["--gpu-layers", "auto"], "auto"),
    (["--n-gpu-layers=33"], 33), (["-ngl", "-1"], "auto"),
])
def test_ngl_forms(args, value):
    assert p("-m", "x", *args).ngl == value


@pytest.mark.parametrize("args", [["--n-cpu-moe", "6"], ["-ncmoe=6"], ["-ncmoe", "6"]])
def test_n_cpu_moe_forms(args):
    assert p("-m", "x", *args).n_cpu_moe == 6


def test_override_tensor_pairs_in_order():
    a = p("-m", "x", "-ot", r"blk\.(1|2)\.ffn=CPU,exps=CUDA0", "--override-tensor", "output=CPU")
    assert a.overrides == [(r"blk\.(1|2)\.ffn", "CPU"), ("exps", "CUDA0"), ("output", "CPU")]


def test_misc_flags():
    a = p("-m", "x", "--cpu-moe", "--no-slots", "-nkvo", "--fit", "off", "-fa", "on",
          "-ctk", "q8_0", "-ctv", "q4_0", "-t", "6", "-np", "2", "-a", "mine",
          "-ncffn", "3", "-b", "1024", "-ub", "256", "--host", "0.0.0.0")
    assert a.cpu_moe and not a.slots and not a.kv_offload and not a.fit
    assert a.flash_attn == "on" and a.cache_type_k == "q8_0" and a.cache_type_v == "q4_0"
    assert a.threads == 6 and a.parallel == 2 and a.alias == "mine" and a.n_cpu_ffn == 3
    assert a.batch == 1024 and a.ubatch == 256 and a.host == "0.0.0.0"


def test_api_key_never_stored():
    a = p("-m", "x", "--api-key", "s3cret")
    assert a.api_key_set
    assert "s3cret" not in repr(a)
    assert "s3cret" not in a.raw


def test_api_key_from_env():
    assert p("-m", "x", env={"LLAMA_API_KEY": "k"}).api_key_set


def test_router_when_no_model():
    assert p("--models-dir", "/m").router is True
    assert p("-hf", "org/repo").router is False


def test_env_used_when_no_cli():
    a = p("-m", "x", env={"LLAMA_ARG_N_GPU_LAYERS": "12", "LLAMA_ARG_ENDPOINT_METRICS": "1",
                          "LLAMA_ARG_CTX_SIZE": "4096", "LLAMA_ARG_N_CPU_MOE": "2"})
    assert a.ngl == 12 and a.sources["ngl"] == "env"
    assert a.metrics and a.ctx == 4096 and a.n_cpu_moe == 2


def test_cli_beats_env():
    a = p("-m", "x", "-ngl", "5", env={"LLAMA_ARG_N_GPU_LAYERS": "12"})
    assert a.ngl == 5 and a.sources["ngl"] == "cli"


def test_model_from_env():
    a = p(env={"LLAMA_ARG_MODEL": "/m/e.gguf"})
    assert a.model == "/m/e.gguf" and a.router is False


@pytest.mark.parametrize("argv0,kind", [
    ("llama-server", "server"), ("/x/llama-cli", "cli"), ("llama", "cli"),
    ("./llama-bench", "bench"), ("llama-perplexity", "other"),
])
def test_kind(argv0, kind):
    assert parse_llama_args([argv0]).kind == kind


def test_unknown_flags_ignored():
    a = p("-m", "x", "--jinja", "--temp", "0.7", "-ngl", "3")
    assert a.ngl == 3


def test_key_flags_summary():
    a = p("-m", "x", "-c", "8192", "--n-cpu-moe", "4")
    flags = dict(key_flags_summary(a))
    assert flags["ngl"] == "auto (fit)"
    assert flags["ctx"] == "8192"
    assert flags["n-cpu-moe"] == "4"
