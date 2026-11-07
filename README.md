# ai_health

[![tests](https://github.com/corentinravoux/ai_health/actions/workflows/tests.yml/badge.svg)](https://github.com/corentinravoux/ai_health/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Local LLM health monitor and GGUF model explorer (desktop app)

## What it does

TODO: two or three sentences and, if useful, a pipeline sketch.

## Install

```bash
git clone https://github.com/corentinravoux/ai_health
cd ai_health
pip install -e ".[dev]"      # with test dependencies
```

## Quick start

```python
import ai_health
```

## Tests

```bash
pytest
```

## Release

```bash
uvx bump-my-version bump patch   # or minor / major: commits + tags v<version>
git push --follow-tags
```

## License

MIT — see [`LICENSE`](LICENSE).
