import os
import subprocess
import sys

import numpy as np
import pytest

from ai_health.explorer.model import open_model, read_tensor_raw


def test_open_dense(make_gguf):
    m = open_model(make_gguf(n_layer=3, n_embd=64, n_head=4, n_head_kv=2, n_ff=128, vocab=256))
    h = m.hparams
    assert h.arch == "llama" and h.n_layer == 3 and h.n_embd == 64
    assert h.n_head == 4 and h.n_head_kv == 2 and h.head_dim == 16
    assert h.n_ff == 128 and h.vocab == 256 and h.ctx_train == 2048
    assert h.n_expert == 0 and h.rope_base == 10000.0
    assert m.name == "demo"
    assert len(m.tensors) == 30
    assert sum(t.n_params for t in m.tensors) == sum(int(np.prod(t.shape)) for t in m.tensors)
    q = m.tensor("blk.1.attn_q.weight")
    assert q.layer == 1 and q.component == "attention" and q.role == "attn_q"
    assert q.shape == [64, 64] and q.qtype == "F32" and q.n_bytes == 64 * 64 * 4
    k = m.tensor("blk.1.attn_k.weight")
    assert k.shape == [64, 32]  # ggml order: [n_embd, n_head_kv*head_dim]
    assert m.file_bytes == os.path.getsize(m.path)


def test_long_per_layer_arrays_still_give_hparams(make_gguf):
    # 80 layers: the metadata panel summarizes the array, the estimate must not
    m = open_model(make_gguf(n_layer=80, n_embd=16, n_head=4, n_head_kv=2, n_ff=16,
                             vocab=32, kv_heads_per_layer=True))
    assert m.metadata["llama.attention.head_count_kv"] == "[array len=80]"
    assert m.hparams.n_head_kv == 2


def test_tensor_lookup_missing(make_gguf):
    m = open_model(make_gguf(n_layer=1))
    with pytest.raises(KeyError):
        m.tensor("nope")


def test_open_moe(make_gguf):
    m = open_model(make_gguf(n_layer=2, n_expert=4, n_expert_used=2))
    assert m.hparams.n_expert == 4 and m.hparams.n_expert_used == 2
    assert m.tensor("blk.0.ffn_up_exps.weight").shape == [64, 128, 4]


def test_open_quantized_bytes(make_gguf):
    m = open_model(make_gguf(n_layer=1, qtype="Q8_0"))
    q = m.tensor("blk.0.attn_q.weight")
    assert q.qtype == "Q8_0"
    assert q.n_bytes == 64 * 64 // 32 * 34


def test_open_split_reads_all_shards(make_gguf):
    m = open_model(make_gguf(n_layer=2, split=2))
    assert len(m.shards) == 2
    assert len(m.tensors) == 2 * 9 + 3
    assert {t.shard for t in m.tensors} == {0, 1}
    assert m.file_bytes == sum(os.path.getsize(p) for p in m.shards)


def test_split_opened_from_second_shard(make_gguf):
    first = make_gguf(n_layer=2, split=2)
    second = first.with_name(first.name.replace("00001-of", "00002-of"))
    assert len(open_model(second).tensors) == 21


def test_metadata_large_arrays_summarised(make_gguf):
    m = open_model(make_gguf(n_layer=1, vocab=256))
    assert m.metadata["tokenizer.ggml.tokens"] == "[array len=256]"
    assert m.metadata["general.architecture"] == "llama"
    assert m.metadata["llama.block_count"] == 1


def test_read_tensor_raw(make_gguf):
    m = open_model(make_gguf(n_layer=1))
    raw, qtype = read_tensor_raw(m, "blk.0.attn_norm.weight")
    assert qtype == "F32" and raw.size == 64


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_model(tmp_path / "nope.gguf")


def test_not_a_gguf_raises_value_error(tmp_path):
    p = tmp_path / "fake.gguf"
    p.write_bytes(b"NOTGGUF" + b"\0" * 100)
    with pytest.raises(ValueError, match="GGUF"):
        open_model(p)


def test_open_does_not_read_tensor_data(make_gguf):
    path = make_gguf(n_layer=4, n_embd=1024, n_ff=4096, vocab=4096)  # ~200 MB F32
    assert os.path.getsize(path) > 150e6
    code = ("import resource, sys; from ai_health.explorer.model import open_model; "
            "import numpy, gguf; base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
            "m = open_model(sys.argv[1]); sum(t.n_bytes for t in m.tensors); "
            "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base)")
    out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True,
                         text=True, check=True).stdout
    assert int(out) / 1024 < 50  # MB grown while opening: header only


def test_open_releases_parsed_metadata(make_gguf):
    # gguf-py keeps every array element (248k tokenizer entries) as numpy objects:
    # ~370 MB for a real model. Metadata is summarised, then the parsed fields dropped.
    m = open_model(make_gguf(n_layer=1, vocab=256))
    assert all(not r.fields for r in m.readers)
    assert m.metadata["tokenizer.ggml.tokens"] == "[array len=256]"
    raw, _ = read_tensor_raw(m, "blk.0.attn_q.weight")
    assert raw.size > 0
