import pytest

from ai_health.explorer.classify import classify_tensor


@pytest.mark.parametrize("name,layer,component,role", [
    ("token_embd.weight", None, "embedding", "token_embd"),
    ("position_embd.weight", None, "embedding", "position_embd"),
    ("blk.3.attn_q.weight", 3, "attention", "attn_q"),
    ("blk.3.attn_k.weight", 3, "attention", "attn_k"),
    ("blk.3.attn_v.weight", 3, "attention", "attn_v"),
    ("blk.3.attn_output.weight", 3, "attention", "attn_output"),
    ("blk.12.attn_qkv.weight", 12, "attention", "attn_qkv"),
    ("blk.3.attn_q.bias", 3, "attention", "attn_q"),
    ("blk.0.attn_norm.weight", 0, "norm", "attn_norm"),
    ("blk.0.attn_q_norm.weight", 0, "norm", "attn_q_norm"),
    ("blk.0.post_ffw_norm.weight", 0, "norm", "post_ffw_norm"),
    ("blk.1.ffn_gate.weight", 1, "ffn", "ffn_gate"),
    ("blk.1.ffn_down.weight", 1, "ffn", "ffn_down"),
    ("blk.1.ffn_up_exps.weight", 1, "moe_experts", "ffn_up_exps"),
    ("blk.1.ffn_gate_up_exps.weight", 1, "moe_experts", "ffn_gate_up_exps"),
    ("blk.1.ffn_down_chexps.weight", 1, "moe_experts", "ffn_down_chexps"),
    ("blk.1.ffn_gate_inp.weight", 1, "moe_router", "ffn_gate_inp"),
    ("blk.1.exp_probs_b.bias", 1, "moe_router", "exp_probs_b"),
    ("blk.1.ffn_up_shexp.weight", 1, "moe_shared", "ffn_up_shexp"),
    ("blk.2.ssm_in.weight", 2, "ssm", "ssm_in"),
    ("blk.2.ssm_conv1d.weight", 2, "ssm", "ssm_conv1d"),
    ("blk.2.time_mix_key.weight", 2, "ssm", "time_mix_key"),
    ("blk.2.shortconv.in_proj.weight", 2, "ssm", "shortconv.in_proj"),
    ("output_norm.weight", None, "norm", "output_norm"),
    ("output.weight", None, "output", "output"),
    ("v.blk.0.attn_q.weight", 0, "vision", "attn_q"),
    ("mm.input_projection.weight", None, "vision", "input_projection"),
    ("rope_freqs.weight", None, "other", "rope_freqs"),
    ("blk.5.nextn.eh_proj.weight", 5, "other", "nextn.eh_proj"),
    ("something_weird", None, "other", "something_weird"),
])
def test_classify(name, layer, component, role):
    c = classify_tensor(name)
    assert (c.layer, c.component, c.role) == (layer, component, role)
