import pytest

from ai_health.explorer.model import open_model
from ai_health.explorer.placement import (
    args_from_flags,
    compute_placement,
    estimate_vram,
)

GiB = 2**30


@pytest.fixture
def dense(make_gguf):
    return open_model(make_gguf(n_layer=4))


@pytest.fixture
def moe(make_gguf):
    return open_model(make_gguf(n_layer=4, n_expert=4, n_expert_used=2))


def dev(p, name):
    return p.device[name]


def test_ngl_partial(dense):
    p = compute_placement(dense, args_from_flags("-ngl 2"))
    assert [layer["device"] for layer in p.layers] == ["CPU", "CPU", "CPU", "GPU"]
    # i_gpu_start = n_layer + 1 - ngl = 3: layer 3 + output on GPU
    assert dev(p, "output.weight") == "GPU" and dev(p, "output_norm.weight") == "GPU"
    assert dev(p, "token_embd.weight") == "CPU"
    assert p.ngl_effective == 2 and p.estimated is False


def test_ngl_three_matches_llama_cpp_rule(dense):
    p = compute_placement(dense, args_from_flags("-ngl 3"))
    assert [layer["device"] for layer in p.layers] == ["CPU", "CPU", "GPU", "GPU"]


def test_ngl_zero_all_cpu_including_output(dense):
    p = compute_placement(dense, args_from_flags("-ngl 0"))
    assert set(p.device.values()) == {"CPU"}
    assert p.gpu_weight_bytes == 0


def test_ngl_all(dense):
    p = compute_placement(dense, args_from_flags("-ngl all"))
    assert all(layer["device"] == "GPU" for layer in p.layers)
    assert dev(p, "token_embd.weight") == "CPU"  # input layer always CPU
    assert p.ngl_effective == 5
    assert p.gpu_weight_bytes + p.cpu_weight_bytes == sum(t.n_bytes for t in dense.tensors)


def test_n_cpu_moe(moe):
    p = compute_placement(moe, args_from_flags("-ngl 99 --n-cpu-moe 2"))
    assert dev(p, "blk.0.ffn_up_exps.weight") == "CPU"
    assert dev(p, "blk.1.ffn_down_exps.weight") == "CPU"
    assert dev(p, "blk.2.ffn_up_exps.weight") == "GPU"
    assert dev(p, "blk.0.attn_q.weight") == "GPU"
    assert dev(p, "blk.0.ffn_gate_inp.weight") == "GPU"
    assert [layer["device"] for layer in p.layers] == ["MIXED", "MIXED", "GPU", "GPU"]
    assert any("n-cpu-moe" in r for r in p.rules)


def test_cpu_moe_all_experts(moe):
    p = compute_placement(moe, args_from_flags("-ngl 99 --cpu-moe"))
    exps = [n for n in p.device if "_exps" in n]
    assert exps and all(p.device[n] == "CPU" for n in exps)


def test_n_cpu_ffn_dense(dense):
    p = compute_placement(dense, args_from_flags("-ngl 99 --n-cpu-ffn 1"))
    assert dev(p, "blk.0.ffn_up.weight") == "CPU" and dev(p, "blk.1.ffn_up.weight") == "GPU"


def test_override_first_match_wins(dense):
    p = compute_placement(dense, args_from_flags(r'-ngl 0 -ot "blk\.3\.=CUDA0,blk\.=CPU"'))
    assert dev(p, "blk.3.attn_q.weight") == "GPU"
    assert dev(p, "blk.2.attn_q.weight") == "CPU"


def test_override_to_cpu_on_gpu_layer(dense):
    p = compute_placement(dense, args_from_flags(r"-ngl 99 -ot attn_output=CPU"))
    assert dev(p, "blk.1.attn_output.weight") == "CPU"
    assert dev(p, "blk.1.attn_q.weight") == "GPU"


def test_auto_without_vram_is_all_gpu_estimated(dense):
    p = compute_placement(dense, args_from_flags(""))
    assert all(layer["device"] == "GPU" for layer in p.layers)
    assert p.estimated is True


def test_auto_fit_off_is_all_gpu_exact(dense):
    p = compute_placement(dense, args_from_flags("--fit off"))
    assert all(layer["device"] == "GPU" for layer in p.layers) and p.estimated is False


def test_auto_with_small_vram_moves_layers(dense):
    a = args_from_flags("-c 2048")
    full = estimate_vram(dense, a, compute_placement(dense, args_from_flags("-ngl all -c 2048")), None)
    # leave room for roughly half the model above the 1 GiB margin
    budget = GiB + full.compute_buffer + full.total // 2 - full.compute_buffer
    p = compute_placement(dense, a, vram_free=budget)
    assert p.estimated is True
    assert p.ngl_effective is not None and p.ngl_effective < 5
    assert any("fit" in r for r in p.rules)


def test_auto_moe_moves_experts_first(moe):
    a = args_from_flags("-c 2048")
    full = estimate_vram(moe, a, compute_placement(moe, args_from_flags("-ngl all -c 2048")), None)
    exp_bytes = sum(t.n_bytes for t in moe.tensors if t.component == "moe_experts" and t.layer == 0)
    p = compute_placement(moe, a, vram_free=GiB + full.total - exp_bytes)
    assert p.device["blk.0.ffn_up_exps.weight"] == "CPU"
    assert p.device["blk.0.attn_q.weight"] == "GPU"
    assert p.ngl_effective == 5


def test_kv_estimate_matches_formula(dense):
    a = args_from_flags("-ngl all -c 4096 -ctk q8_0 -ctv f16")
    e = estimate_vram(dense, a, compute_placement(dense, a), 12 * GiB)
    # per attention layer: ctx * n_head_kv * head_dim * (1.0625 + 2)
    assert e.kv_cache == round(4 * 4096 * 2 * 16 * (1.0625 + 2))
    assert e.total == e.weights + e.kv_cache + e.compute_buffer
    assert e.capacity == 12 * GiB and e.fits is True


def test_kv_uses_ctx_train_when_unset(dense):
    a = args_from_flags("-ngl all")
    e = estimate_vram(dense, a, compute_placement(dense, a), None)
    assert e.kv_cache == 4 * 2048 * 2 * 16 * 4
    assert e.fits is None


def test_kv_on_cpu_when_no_kv_offload(dense):
    a = args_from_flags("-ngl all -c 4096 -nkvo")
    e = estimate_vram(dense, a, compute_placement(dense, a), None)
    assert e.kv_cache == 0
    assert any("KV cache in RAM" in n for n in e.notes)


def test_kv_only_for_gpu_layers(dense):
    a = args_from_flags("-ngl 2 -c 1024")
    e = estimate_vram(dense, a, compute_placement(dense, a), None)
    assert e.kv_cache == 1 * 1024 * 2 * 16 * 4


def test_fits_flag(dense):
    a = args_from_flags("-ngl all")
    p = compute_placement(dense, a)
    assert estimate_vram(dense, a, p, 1024).fits is False


def test_args_from_flags_accepts_full_command():
    a = args_from_flags("~/llama.cpp/build/bin/llama-server -m x.gguf -ngl 7")
    assert a.ngl == 7 and a.model == "x.gguf"
    assert args_from_flags("-ngl 3").ngl == 3


def test_kv_skips_linear_attention_layers(make_gguf):
    m = open_model(make_gguf(n_layer=4, linear_attn_layers=(0, 1, 2)))
    a = args_from_flags("-ngl all -c 1024")
    e = estimate_vram(m, a, compute_placement(m, a), None)
    assert e.kv_cache == 1 * 1024 * 2 * 16 * 4
