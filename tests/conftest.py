from pathlib import Path

import pytest

from ai_health.devtools.make_demo_gguf import make_gguf as _make_gguf

FAKES = Path(__file__).parent / "fakes"


@pytest.fixture
def make_gguf(tmp_path):
    counter = {"n": 0}

    def factory(**kwargs):
        counter["n"] += 1
        return _make_gguf(tmp_path / f"model{counter['n']}.gguf", **kwargs)

    return factory


@pytest.fixture
def fakes_dir():
    return FAKES
