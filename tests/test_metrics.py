import json

from ai_health.llama.metrics import ThroughputTracker, parse_prometheus


def test_parse_prometheus_strips_prefix_and_comments(fakes_dir):
    m = parse_prometheus((fakes_dir / "metrics.txt").read_text())
    assert m["tokens_predicted_total"] == 1200
    assert m["prompt_seconds_total"] == 5
    assert m["requests_processing"] == 1
    assert not any(k.startswith("#") or ":" in k for k in m)


def test_parse_prometheus_labels_and_garbage():
    m = parse_prometheus('x{position="1"} 3\nbroken line\nllamacpp:y 2.5e1\n')
    assert m["y"] == 25.0
    assert m['x{position="1"}'] == 3


def M(pred, pred_s, prompt=0, prompt_s=0.0, proc=1):
    return {"tokens_predicted_total": pred, "tokens_predicted_seconds_total": pred_s,
            "prompt_tokens_total": prompt, "prompt_seconds_total": prompt_s,
            "requests_processing": proc, "requests_deferred": 0}


def test_tps_from_metrics_diff():
    tr = ThroughputTracker()
    first = tr.update(0.0, M(100, 3.0, 1000, 1.0), None)
    assert first.gen_tps is None and first.source == "metrics"
    t = tr.update(1.0, M(140, 4.0, 1500, 1.5), None)
    assert t.gen_tps == 40.0
    assert t.prompt_tps == 1000.0
    assert t.busy is True and t.requests_processing == 1


def test_idle_keeps_last_rate_but_busy_false():
    tr = ThroughputTracker()
    tr.update(0.0, M(100, 3.0), None)
    tr.update(1.0, M(140, 4.0), None)
    t = tr.update(2.0, M(140, 4.0, proc=0), None)
    assert t.busy is False
    assert t.gen_tps == 40.0


def test_counter_reset_is_not_negative():
    tr = ThroughputTracker()
    tr.update(0.0, M(1000, 30.0), None)
    t = tr.update(1.0, M(10, 0.3), None)
    assert t.gen_tps is None or t.gen_tps >= 0
    t2 = tr.update(2.0, M(50, 1.3), None)
    assert t2.gen_tps == 40.0


def slots(n, busy=True):
    return [{"id": 0, "n_ctx": 8192, "is_processing": busy, "next_token": {"n_decoded": n}}]


def test_tps_from_slots_when_no_metrics():
    tr = ThroughputTracker()
    tr.update(0.0, None, slots(10))
    t = tr.update(2.0, None, slots(50))
    assert t.gen_tps == 20.0 and t.source == "slots" and t.busy is True
    # new request: n_decoded restarts, counts from 0
    t = tr.update(3.0, None, slots(15))
    assert t.gen_tps == 15.0
    assert t.prompt_tps is None


def test_slots_idle_is_not_busy(fakes_dir):
    tr = ThroughputTracker()
    t = tr.update(0.0, None, json.loads((fakes_dir / "slots_idle.json").read_text()))
    assert t.busy is False


def test_no_data_source_none():
    t = ThroughputTracker().update(0.0, None, None)
    assert t.source == "none" and t.gen_tps is None and t.busy is None


def test_tokens_last_min():
    tr = ThroughputTracker()
    tr.update(0.0, M(0, 0.0), None)
    tr.update(1.0, M(40, 1.0), None)
    tr.update(30.0, M(100, 2.5), None)
    t = tr.update(70.0, M(130, 3.3), None)
    # window (10, 70]: +60 at t=30 and +30 at t=70
    assert t.tokens_last_min == 90
