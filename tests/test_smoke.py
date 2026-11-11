import re

import ai_health


def test_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", ai_health.__version__)
