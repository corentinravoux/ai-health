import base64

import gguf
import numpy as np
import pytest

from ai_health.explorer.model import open_model, read_tensor_raw
from ai_health.explorer.weights import ViewCache, tensor_view


def f32(m, name):
    raw, _ = read_tensor_raw(m, name)
    return np.asarray(raw, dtype=np.float32)


def test_view_f32_stats_match_numpy(make_gguf):
    m = open_model(make_gguf(n_layer=1))
    v = tensor_view(m, "blk.0.attn_q.weight")
    x = f32(m, "blk.0.attn_q.weight")
    assert v.available and v.reason is None
    for k, ref in [("mean", x.mean()), ("std", x.std()), ("min", x.min()), ("max", x.max()),
                   ("absmax", np.abs(x).max()), ("rms", np.sqrt((x ** 2).mean()))]:
        assert v.stats[k] == pytest.approx(float(ref), abs=1e-6)
    assert 0 <= v.stats["near_zero_frac"] <= 1 and 0 <= v.stats["outlier_frac"] <= 1
    assert sum(v.hist_counts) == x.size and len(v.hist_counts) == 128
    assert len(v.hist_edges) == 129
    assert v.shape == [64, 64] and v.qtype == "F32"


def test_view_q8_0_dequantizes(make_gguf):
    ref = open_model(make_gguf(n_layer=1, qtype="F32"))
    q = open_model(make_gguf(n_layer=1, qtype="Q8_0"))
    a = tensor_view(ref, "blk.0.ffn_up.weight")
    b = tensor_view(q, "blk.0.ffn_up.weight")
    assert b.available and b.qtype == "Q8_0"
    assert abs(a.stats["std"] - b.stats["std"]) < 0.05 * a.stats["std"]
    assert abs(a.stats["absmax"] - b.stats["absmax"]) < 0.05


def test_heatmap_downsampled_keeps_aspect(make_gguf):
    # ggml shape [n_embd=2048, vocab=1024] -> 1024 rows x 2048 cols
    m = open_model(make_gguf(n_layer=1, n_embd=2048, n_head=16, n_head_kv=4, n_ff=64, vocab=1024))
    v = tensor_view(m, "token_embd.weight")
    assert (v.heat_w, v.heat_h) == (512, 256)
    assert len(base64.b64decode(v.heatmap_b64)) == 512 * 256


def test_small_tensor_not_upscaled(make_gguf):
    m = open_model(make_gguf(n_layer=1))
    v = tensor_view(m, "blk.0.attn_k.weight")  # 32 rows x 64 cols
    assert (v.heat_w, v.heat_h) == (64, 32)


def test_1d_tensor_rendered_as_strip(make_gguf):
    m = open_model(make_gguf(n_layer=1))
    v = tensor_view(m, "blk.0.attn_norm.weight")
    assert v.heat_h == 1 and v.heat_w == 64


def test_3d_expert_tensor_flattened_by_expert(make_gguf):
    m = open_model(make_gguf(n_layer=1, n_expert=4, n_expert_used=2))
    v = tensor_view(m, "blk.0.ffn_up_exps.weight")  # 4 experts x 128 rows x 64 cols
    assert v.rows_total == 512 and v.heat_h <= 512
    assert v.heat_h == 512 and v.heat_w == 64


def test_large_tensor_is_row_sampled(make_gguf):
    m = open_model(make_gguf(n_layer=1, vocab=20000))
    v = tensor_view(m, "token_embd.weight", max_rows=8192)
    assert v.sampled and v.rows_used == 8192 and v.rows_total == 20000
    assert sum(v.hist_counts) == 8192 * 64


def test_reduce_mean_is_symmetric(make_gguf):
    m = open_model(make_gguf(n_layer=1))
    v = tensor_view(m, "blk.0.attn_q.weight", reduce="mean")
    assert v.reduce == "mean" and v.heat_min == pytest.approx(-v.heat_max)


def test_bad_reduce_rejected(make_gguf):
    m = open_model(make_gguf(n_layer=1))
    with pytest.raises(ValueError):
        tensor_view(m, "blk.0.attn_q.weight", reduce="median")


def test_unsupported_qtype_returns_reason(make_gguf, monkeypatch):
    m = open_model(make_gguf(n_layer=1, qtype="Q4_0"))

    def boom(data, qtype):
        raise NotImplementedError(f"Dequantization for {qtype.name} is not yet implemented")

    monkeypatch.setattr(gguf.quants, "dequantize", boom)
    v = tensor_view(m, "blk.0.attn_q.weight")
    assert v.available is False and "Q4_0" in v.reason
    assert v.stats == {} and v.heatmap_b64 is None and v.hist_counts == []


def test_cache_hits(make_gguf, monkeypatch):
    m = open_model(make_gguf(n_layer=1))
    cache = ViewCache()
    a = cache.get(m, "blk.0.attn_q.weight", "absmax")
    monkeypatch.setattr("ai_health.explorer.weights.tensor_view",
                        lambda *a, **k: pytest.fail("cache miss"))
    assert cache.get(m, "blk.0.attn_q.weight", "absmax") is a


def test_element_cap_limits_rows(make_gguf):
    m = open_model(make_gguf(n_layer=1, vocab=20000))  # token_embd: 20000 rows x 64
    v = tensor_view(m, "token_embd.weight", max_rows=8192, max_elems=64 * 1000)
    assert v.sampled and v.rows_used == 1000


def test_service_serialises_and_dedupes_views(make_gguf, monkeypatch):
    import threading
    import time

    from ai_health.explorer import weights
    from ai_health.explorer.service import ExplorerService

    svc = ExplorerService(search_dirs=[])
    svc.open(str(make_gguf(n_layer=1)))
    calls, active, peak = [], [0], [0]
    real = weights.tensor_view

    def slow(*a, **k):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        calls.append(a[1])
        time.sleep(0.05)
        active[0] -= 1
        return real(*a, **k)

    monkeypatch.setattr(weights, "tensor_view", slow)
    names = ["blk.0.attn_q.weight"] * 3 + ["blk.0.attn_k.weight", "blk.0.ffn_up.weight"]
    ts = [threading.Thread(target=svc.view, args=(n, "absmax")) for n in names]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 1                      # one dequantization at a time
    assert calls.count("blk.0.attn_q.weight") == 1   # duplicates served from the first
