import json

from ai_health.llama.probe import ServerProbe


class FakeFetch:
    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append((url, timeout))
        for key, resp in self.routes.items():
            if url.split("?")[0].endswith(key):
                if isinstance(resp, Exception):
                    raise resp
                code, body = resp
                return code, body.encode() if isinstance(body, str) else body
        return 404, b"not found"


def test_metrics_ok_parses(fakes_dir):
    f = FakeFetch({"/metrics": (200, (fakes_dir / "metrics.txt").read_text())})
    r = ServerProbe("127.0.0.1", 8080, fetch=f).metrics()
    assert r.ok and r.status == "ok"
    assert r.data["tokens_predicted_total"] == 1200
    assert f.urls[0] == ("http://127.0.0.1:8080/metrics", 0.3)


def test_metrics_501_is_disabled():
    body = json.dumps({"error": {"code": 501, "message": "This server does not support metrics endpoint.",
                                 "type": "not_supported_error"}})
    r = ServerProbe("127.0.0.1", 8080, fetch=FakeFetch({"/metrics": (501, body)})).metrics()
    assert not r.ok and r.status == "disabled"


def test_probe_401_reports_auth():
    r = ServerProbe("127.0.0.1", 8080, fetch=FakeFetch({"/slots": (401, "{}")})).slots()
    assert r.status == "auth" and r.data is None


def test_unreachable_on_connection_error():
    r = ServerProbe("127.0.0.1", 8080,
                    fetch=FakeFetch({"/health": ConnectionRefusedError("refused")})).health()
    assert r.status == "unreachable" and "refused" in r.detail


def test_timeout_is_unreachable():
    r = ServerProbe("127.0.0.1", 8080, fetch=FakeFetch({"/health": TimeoutError("timed out")})).health()
    assert r.status == "unreachable"


def test_health_503_is_loading():
    r = ServerProbe("127.0.0.1", 8080, fetch=FakeFetch({"/health": (503, '{"error":{"message":"Loading model"}}')})).health()
    assert r.status == "loading"


def test_slots_and_props_json(fakes_dir):
    f = FakeFetch({"/slots": (200, (fakes_dir / "slots_busy.json").read_text()),
                   "/props": (200, (fakes_dir / "props.json").read_text())})
    p = ServerProbe("127.0.0.1", 8080, fetch=f)
    assert p.slots().data[0]["n_ctx"] == 8192
    assert p.props().data["total_slots"] == 1


def test_bad_json_is_error():
    r = ServerProbe("127.0.0.1", 8080, fetch=FakeFetch({"/slots": (200, "<html>")})).slots()
    assert r.status == "error"


def test_router_adds_model_query():
    f = FakeFetch({"/metrics": (200, "")})
    ServerProbe("127.0.0.1", 8080, fetch=f).metrics(model="org/m:Q4")
    assert f.urls[0][0] == "http://127.0.0.1:8080/metrics?model=org%2Fm%3AQ4"


def test_wildcard_host_probed_on_loopback():
    f = FakeFetch({"/health": (200, '{"status":"ok"}')})
    ServerProbe("0.0.0.0", 9000, fetch=f).health()
    assert f.urls[0][0].startswith("http://127.0.0.1:9000/")


def test_slots_that_are_not_a_list_of_objects_are_errors():
    for body in ('{"error": "slots disabled"}', "[null]", '[1, 2]'):
        r = ServerProbe("127.0.0.1", 8080, fetch=FakeFetch({"/slots": (200, body)})).slots()
        assert r.ok is False and r.status == "error", body
