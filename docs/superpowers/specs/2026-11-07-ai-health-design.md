# ai-health — design

Date: 2026-11-07 · Author: Corentin Ravoux (with Claude)

## Intent

A local desktop app for watching a local LLM run on this PC, and for
understanding the model itself. The owner is starting out with local LLMs on
a constrained machine (RTX 3060 12 GB, 16 GB DDR4, Ryzen 5 3600, llama.cpp
CUDA build in `~/Software/llama.cpp`). What they keep wanting to know is:
**how fast is it generating (tokens/s), how full is VRAM, how much RAM is it
eating, and which processes hold what.** The second purpose is learning:
an on-demand explorer that draws the model's architecture as an AI diagram
and gives a complete view of its weights.

## Goals

1. **Monitor mode (live, always on while the app is open)**
   - Tokens/s: generation and prompt-processing throughput, live and over time.
   - VRAM: total used/free, and per-process VRAM (who holds what).
   - RAM and swap: system usage and per-process RSS of LLM processes.
   - GPU: utilization, temperature, power, clocks.
   - CPU: overall and per-LLM-process usage and thread count.
   - Processes: discover running llama.cpp processes (`llama-server`,
     `llama-cli`, `llama-bench`, router-mode child instances) and show the
     process tree, PID, the model file, and the launch flags that matter
     (`-ngl`, `--n-cpu-moe`, `-ot/--override-tensor`, `-c`, `-t`,
     `--cache-type-k/v`, `-fa`, `--port`), parsed from `/proc/<pid>/cmdline`.
   - Server status per llama-server: slots (busy/idle, context used vs
     `n_ctx`), requests processing/deferred, from `/slots`, `/props`,
     `/health`, `/metrics`.
   - History: rolling window (default 15 min) in memory with charts.
   - Warnings that matter for this machine: VRAM near full, swap in use by
     the LLM process, layers/experts placed on CPU, `/metrics` disabled
     (with the flag to add).

2. **Explorer mode (on demand, offline, for understanding)**
   - Starts only when the user opens a GGUF file (file picker, drag-drop,
     or "explore the model this server is running").
   - Reads the GGUF with `gguf-py` (header, metadata, tensor table) without
     loading the whole file in RAM (memory-mapped reader).
   - **Architecture diagram**: a dedicated, AI-style diagram of the model:
     token embedding → N repeated transformer blocks → final norm → output
     head. Each block expands to its sub-modules (attention: q/k/v/o, norms,
     rope; FFN: gate/up/down; MoE: router + experts + shared expert; SSM/
     hybrid layers when present). Blocks are drawn once with "× N" plus a
     scrollable layer strip, so a 64-layer model stays readable.
   - **Complete weight view**: every tensor listed and reachable from the
     diagram, with name, shape, quant type, bytes, parameter count, share
     of model size. Clicking a tensor dequantizes it (on demand, cached) and
     shows a downsampled heatmap, a value histogram, and stats
     (mean, std, min/max, abs-max, sparsity near zero, outlier ratio).
   - Model-level summaries: parameter count by component (embeddings,
     attention, FFN/experts, output), bytes by quant type, metadata table
     (context length, heads, KV heads, head dim, rope, vocab, chat template).
   - **Placement overlay (links the two modes)**: given a llama-server's
     flags (or flags typed by hand), colour each layer/tensor by where it
     lives: GPU or CPU. Estimate VRAM: weights on GPU + KV cache for the
     chosen `-c` and cache types + a compute-buffer allowance, against the
     12 GB card. This answers "what spills out of VRAM and why".
   - External viewers: buttons to open the same file in Netron
     (`netron` Python package, experimental GGUF support) and in Google
     Model Explorer with the `model-explorer-gguf` adapter, when installed.
     These are optional integrations; the app works without them.

3. **Desktop app**: a native window (pywebview, GTK/WebKit2 backend, which
   is already present system-wide), a launcher script and a `.desktop`
   entry so it starts from the Ubuntu app menu. A `--browser` flag serves
   the same UI in a normal browser for development and screenshots.

4. **A front that looks deliberately designed**, not a template dashboard
   (see Visual direction).

## Non-goals

- Not a launcher/manager for llama-server (no start/stop/model download in
  v1). It observes. *(Builder may add "copy suggested flags" text.)*
- No Ollama, vLLM, LM Studio support in v1. Collectors are pluggable so
  they can be added later.
- No live per-token activation tracing (Spectra-Vis style) — explorer is
  static weights only.
- No remote/multi-machine monitoring, no auth, no Prometheus/Grafana stack.
- No PyPI publishing.

## Anti-goals (would make it a failure even if it works)

- The monitor itself costs noticeable GPU/CPU/RAM. It must stay light:
  1 Hz sampling, no GPU usage by the app (WebKit compositing aside), well
  under 2 % CPU, explorer memory freed when closed.
- Explorer loads a whole 8 GB model into RAM on a 16 GB machine and pushes
  the LLM into swap. Tensors are read lazily via mmap, one at a time, and
  heatmaps are downsampled before reaching the UI.
- Numbers that look precise but are wrong (e.g. tokens/s computed from a
  stale counter). Every derived number says where it comes from; unknown is
  shown as unknown.
- Generic "AI dashboard" look: purple gradients, glassmorphism cards,
  emoji icons.

## Constraints

- Linux only (Ubuntu 26.04), NVIDIA only (NVML). Python 3.14 system.
- No Node, Rust, pip or uv installed today. Frontend must need **no build
  step**: plain HTML/CSS/ES modules, with third-party JS vendored into the
  repo (works offline).
- Python deps: `nvidia-ml-py`, `psutil`, `numpy`, `pywebview`, `gguf`
  (prefer the local `~/Software/llama.cpp/gguf-py` checkout so it tracks the
  built llama.cpp; fall back to the PyPI `gguf`). PyGObject comes from the
  system (`/usr/lib/python3/dist-packages/gi`), so the env must see system
  site-packages.
- Python environment (owner's decision): the single env of this PC,
  `~/Software/venv` (Python 3.14, created with uv `--system-site-packages`
  for system PyGObject). Used for all work on this PC; never `claude_env`,
  no per-project venv. `uv` is at `~/.local/bin/uv`.
- No real GGUF model is downloaded on this machine yet. Tests must build
  tiny synthetic GGUF files with `gguf.GGUFWriter`; never require a real
  model.
- llama.cpp facts verified in the local checkout
  (`tools/server/README.md`): `/metrics` only with `--metrics`, metric names
  `llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`,
  `llamacpp:tokens_predicted_total`,
  `llamacpp:tokens_predicted_seconds_total`,
  `llamacpp:requests_processing`, `llamacpp:requests_deferred`;
  `/slots` on by default (`--no-slots` disables); router mode needs
  `?model=` on `/metrics` and `/props`.
- Repo conventions for an application (owner's rule): README, MIT license,
  CI tests. No docs site required.

## Decided how (owner's decisions)

- Name `ai-health`, located at `~/Software/ai-health`.
- Desktop app, not a terminal tool.
- Two modes: live monitor; on-demand explorer with AI architecture diagram
  and complete weight view.
- Builds on the tools found by the web search: NVML, psutil, llama-server
  endpoints, gguf-py, pywebview, Netron/Model Explorer as external viewers.
- A large plan and a carefully designed front before implementation; a
  final front-end polish pass once the interface works.

## My calls (owner can override)

- **Architecture**: Python backend = small stdlib HTTP server on
  `127.0.0.1:<random port>` serving static files + JSON endpoints + a
  Server-Sent Events stream for live samples. pywebview opens a window on
  that URL. Same code path for `--browser`. No web framework.
- **Sampler**: one background thread at 1 Hz collecting GPU, system,
  processes, servers into a ring buffer; SSE pushes each sample. Server
  HTTP polls time out fast (300 ms) so a hung server never stalls sampling.
- **Tokens/s method**, in order of preference: (1) diff of
  `tokens_predicted_total` / `tokens_predicted_seconds_total` between
  samples (true decode speed while busy); (2) diff of `/slots` token
  counters over wall time when `/metrics` is off; (3) unknown. Displayed
  rate is "while generating", plus a separate "tokens in last minute".
  Prompt-processing speed is computed the same way from the prompt
  counters.
- **Per-process VRAM** via `nvmlDeviceGetComputeRunningProcesses`.
- **Placement model**: llama.cpp offloads the *last* `ngl` repeating layers
  plus the output layer when `ngl > n_layer`; `--n-cpu-moe N` keeps the
  expert tensors of the first N layers on CPU; `-ot regex=CPU` patterns are
  applied by regex on tensor names. `-ngl auto/-1/999` and llama.cpp's
  `--fit` behaviour mean "all that fit": shown as "auto (estimated)". The
  builder must confirm these rules against `~/Software/llama.cpp` source
  (`common/arg.cpp`, `src/llama-model.cpp`) and note any it could not.
- **KV estimate**: `2 × n_layer_attn × n_ctx × n_head_kv × head_dim ×
  bytes(cache type)` (SWA / hybrid models flagged as approximate).
- **Heatmap**: dequantize to float32, reduce to ≤ 512×512 by block
  abs-max or mean (toggle), send as base64 uint8 with a scale; histogram
  128 bins computed in numpy. Cap per-tensor work (e.g. token embedding
  of 150k×4096 is sampled by rows) and show "sampled" when it is.
- **Frontend libraries (vendored)**: uPlot for time-series (tiny, fast,
  canvas); the architecture diagram is hand-built SVG (not a graph library)
  so it can look like a designed AI diagram; heatmaps on `<canvas>`.
- **Persistence**: none except a small JSON settings file
  (`~/.config/ai-health/settings.json`): window size, history length,
  last opened model, theme.
- **Packaging**: `pyproject.toml` with a console entry point `ai-health`,
  `scripts/install-desktop-entry.sh` writing
  `~/.local/share/applications/ai-health.desktop`.
- **Testing**: pytest. Collectors tested against fakes (fake NVML module,
  canned `/metrics` text and `/slots` JSON, fake process tables); GGUF
  explorer tested on synthetic GGUFs (dense llama-like and MoE-like) with
  F32/F16/Q8_0/Q4_0 tensors; placement rules tested table-driven; API
  tested through the HTTP server. Frontend checked by screenshots in
  `--browser` mode (headless Chromium/WebKit if available, otherwise
  manual).

- **Placement when rules can't be confirmed**: ship the overlay anyway,
  labelled "estimated" with the rule used; never block it. Rules verified
  against the source are marked "per llama.cpp source" in a tooltip.
- **VRAM bar**: shows *every* GPU memory holder (browser, compositor, …)
  with llama.cpp processes highlighted, plus an "unattributed" segment =
  total used − sum of per-process. If NVML gives no per-process figures,
  fall back to `nvidia-smi --query-compute-apps`, then "unknown".
- **Vendored assets**: download once with curl, commit to
  `ai_health/static/vendor/` with their licence files (uPlot MIT, fonts
  OFL). Tests do not depend on them.
- **Screenshots**: Firefox is installed; use Playwright-for-Python with
  Firefox if the env allows it, else `firefox --headless --screenshot`.
  No Chromium install.

## Visual direction (my call, refined in the front-end pass)

Feel: a precision instrument panel for a machine you care about — the
calm of a lab oscilloscope, not a crypto dashboard.

- **Dark first** (graphite `#0e1013`-ish background, not pure black), with
  a light theme toggle. One signal colour per quantity, used everywhere
  that quantity appears: VRAM = warm amber, RAM = cool cyan, tokens/s =
  phosphor green, GPU compute = soft violet, CPU = neutral sand. Red only
  for real danger (VRAM > 95 %, swap in use).
- **Typography**: a characterful grotesk for UI (e.g. "Space Grotesk" or
  "Inter Tight", vendored as woff2) + a monospace with tabular numerals
  for every number ("JetBrains Mono"/"IBM Plex Mono"). Numbers are the
  hero: big tabular readouts that don't jitter width.
- **Monitor layout**: left rail (mode switch: Monitor / Explorer,
  settings); top strip of four large readouts (tok/s, VRAM, RAM, GPU %)
  with sparkline underneath each; a VRAM "fill bar" segmented by process;
  main area with stacked time-series sharing one time axis; right column
  with the process tree and server slot cards.
- **Explorer layout**: centre = vertical architecture diagram (embedding at
  bottom, head at top, like the classic transformer figure, but drawn
  in the app's own style: thin strokes, rounded modules, tensor chips sized
  by bytes, GPU/CPU placement as a fill pattern); left = tensor tree +
  search; right = inspector (heatmap, histogram, stats); bottom = layer
  strip (one cell per layer, colour = placement or chosen stat, click to
  focus).
- Motion: minimal — smooth chart streaming, diagram expand/collapse
  transitions; respects `prefers-reduced-motion`.

The builder should use the frontend-design / design-taste skills for the
polish pass, with screenshots before and after.

## Left to the builder

- Exact module split, endpoint names, SSE payload schema.
- Chart details, colours' exact values, fonts (within the direction above).
- How to classify tensor names into components (use `gguf.constants`
  `MODEL_TENSOR` names / `tensor_mapping.py` as the source of truth).
- Handling of split GGUFs (`-00001-of-0000N`), multimodal projector files.
- Error states and empty states (no GPU, no server running, no model).

## Owner's answers to the review questions

- Python env: the single PC env `~/Software/venv` (see Constraints).
- Verification: download one small real GGUF (Q4_K_M, ≤ 2.5 GB) to
  `~/models` and check monitor + explorer end-to-end against a real
  `llama-server --metrics`. Acceptance = monitor shows correct tok/s, VRAM,
  RAM, process and flags for that server; explorer opens that file and
  its placement overlay matches the server's `-ngl`.
- Repo: public GitHub repo `corentinravoux/ai-health`, GitHub Actions CI
  running the pytest suite on fakes only (no GPU, no GTK; pywebview import
  is lazy so tests don't need it).
