import json
from pathlib import Path

import gguf

FAKES = Path(__file__).parent / "fakes"


def test_make_gguf_roundtrip(make_gguf):
    reader = gguf.GGUFReader(make_gguf(n_layer=3))
    assert len(reader.tensors) == 3 * 9 + 3
    field = reader.fields["llama.block_count"]
    assert int(field.parts[field.data[0]][0]) == 3


def test_make_gguf_moe(make_gguf):
    reader = gguf.GGUFReader(make_gguf(n_layer=2, n_expert=4, n_expert_used=2))
    names = {t.name: t for t in reader.tensors}
    assert "blk.0.ffn_up_exps.weight" in names
    assert 4 in [int(x) for x in names["blk.0.ffn_up_exps.weight"].shape]
    assert "blk.0.ffn_gate_inp.weight" in names


def test_make_gguf_quantized(make_gguf):
    reader = gguf.GGUFReader(make_gguf(n_layer=1, qtype="Q8_0"))
    names = {t.name: t for t in reader.tensors}
    assert names["blk.0.attn_q.weight"].tensor_type == gguf.GGMLQuantizationType.Q8_0
    # norms stay F32 like real models
    assert names["blk.0.attn_norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32


def test_make_gguf_split(make_gguf):
    first = make_gguf(n_layer=2, split=2)
    assert first.name.endswith("-00001-of-00002.gguf")
    assert first.with_name(first.name.replace("00001-of", "00002-of")).exists()


def test_canned_payloads_parse():
    assert "llamacpp:tokens_predicted_total" in (FAKES / "metrics.txt").read_text()
    slots = json.loads((FAKES / "slots_busy.json").read_text())
    assert slots[0]["is_processing"] is True
    assert slots[0]["next_token"]["n_decoded"] == 40
    json.loads((FAKES / "props.json").read_text())
