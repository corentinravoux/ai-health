import pytest

from ai_health.explorer.architecture import build_architecture
from ai_health.explorer.model import open_model


def arch_of(make_gguf, **kw):
    return build_architecture(open_model(make_gguf(**kw)))


def test_dense_block_groups(make_gguf):
    a = arch_of(make_gguf, n_layer=3)
    assert [g["id"] for g in a["block"]["groups"]] == ["attn_norm", "attention", "ffn_norm", "ffn"]
    att = a["block"]["groups"][1]
    assert att["roles"] == ["attn_q", "attn_k", "attn_v", "attn_output"]
    assert att["detail"] == {"n_head": 4, "n_head_kv": 2, "head_dim": 16, "gqa_ratio": 2}
    ffn = a["block"]["groups"][3]
    assert ffn["label"] == "SwiGLU FFN" and ffn["roles"] == ["ffn_gate", "ffn_up", "ffn_down"]
    assert a["block"]["kind"] == "transformer" and a["block"]["hybrid"] is False
    assert a["arch"] == "llama" and a["name"] == "demo"
    assert a["hparams"]["n_layer"] == 3


def test_moe_block_has_router_and_experts(make_gguf):
    a = arch_of(make_gguf, n_layer=2, n_expert=4, n_expert_used=2)
    ids = [g["id"] for g in a["block"]["groups"]]
    assert "moe_router" in ids and "moe_experts" in ids and "ffn" not in ids
    exp = next(g for g in a["block"]["groups"] if g["id"] == "moe_experts")
    assert exp["detail"] == {"n_expert": 4, "n_expert_used": 2}
    assert all(layer["kind"] == "moe" for layer in a["layers"])


def test_layers_count_and_bytes_sum(make_gguf):
    m = open_model(make_gguf(n_layer=3))
    a = build_architecture(m)
    assert [layer["index"] for layer in a["layers"]] == [0, 1, 2]
    total = sum(t.n_bytes for t in m.tensors)
    parts = (sum(layer["bytes"] for layer in a["layers"]) + sum(s["bytes"] for s in a["stem"])
             + sum(h["bytes"] for h in a["head"]) + sum(e["bytes"] for e in a["extras"]))
    assert parts == total == a["summary"]["bytes_total"]
    g = a["layers"][1]["groups"]["attention"]
    assert "blk.1.attn_q.weight" in g["tensors"] and g["bytes"] == 4 * (64 * 64 * 2 + 64 * 32 * 2)


def test_stem_and_head(make_gguf):
    a = arch_of(make_gguf, n_layer=1)
    assert a["stem"][0]["id"] == "token_embd" and a["stem"][0]["tensors"] == ["token_embd.weight"]
    head_ids = [h["id"] for h in a["head"]]
    assert head_ids == ["output_norm", "output"]
    assert a["head"][1]["tied"] is False


def test_tied_output(make_gguf):
    a = arch_of(make_gguf, n_layer=1, tied=True)
    out = a["head"][-1]
    assert out["id"] == "output" and out["tied"] is True
    assert out["tensors"] == ["token_embd.weight"] and out["bytes"] == 0


def test_heterogeneous_layers_flagged(make_gguf):
    a = arch_of(make_gguf, n_layer=4, ssm_layers=(0, 2))
    kinds = [layer["mixer"] for layer in a["layers"]]
    assert kinds == ["ssm", "attention", "ssm", "attention"]
    assert a["layers"][0]["kind"] == "ssm"
    assert a["block"]["hybrid"] is True
    assert "ssm" in [g["id"] for g in a["block"]["groups"]]


def test_summary_shares_sum_to_one(make_gguf):
    s = arch_of(make_gguf, n_layer=2, n_expert=4, n_expert_used=2)["summary"]
    assert sum(c["share"] for c in s["by_component"]) == pytest.approx(1.0)
    comps = {c["component"] for c in s["by_component"]}
    assert {"embedding", "attention", "moe_experts", "moe_router", "norm", "output"} <= comps
    assert s["params_total"] == sum(c["params"] for c in s["by_component"])
    assert s["by_qtype"][0]["qtype"] == "F32"


def test_kv_bytes_per_token(make_gguf):
    s = arch_of(make_gguf, n_layer=4, ssm_layers=(0,))["summary"]
    assert s["kv_bytes_per_token_f16"] == 2 * 3 * 2 * 16 * 2
    assert s["n_layer_attn"] == 3
