// Architecture diagram: a hand-built SVG of the model, data flowing upwards:
// tokens -> embedding -> [block × N] -> final norm -> output head -> logits.
// The block shows the selected layer's real tensors as chips (width ∝ log
// bytes); a placement overlay fills chips amber (GPU) or hatched sand (CPU).
import { store } from "../store.js";
import { h, hideTip, s, showTip } from "../ui.js";
import { bytes, num, shape } from "../format.js";

const W = 760;
const RAIL = 120;
const MX = 190;            // module left edge
const MW = W - MX - 40;    // module width
const ROLE_LABEL = {
  attn_q: "Q", attn_k: "K", attn_v: "V", attn_qkv: "QKV", attn_output: "O",
  attn_q_a: "Q↓", attn_q_b: "Q↑", attn_kv_a_mqa: "KV↓", attn_kv_b: "KV↑", attn_k_b: "K↑", attn_v_b: "V↑",
  attn_q_norm: "q-norm", attn_k_norm: "k-norm", attn_sinks: "sinks",
  ffn_gate: "gate", ffn_up: "up", ffn_down: "down",
  ffn_gate_inp: "router", exp_probs_b: "bias",
  ffn_gate_exps: "gate ×E", ffn_up_exps: "up ×E", ffn_down_exps: "down ×E", ffn_gate_up_exps: "gate·up ×E",
  ffn_gate_shexp: "gate", ffn_up_shexp: "up", ffn_down_shexp: "down", ffn_gate_inp_shexp: "gate",
};

export function mount(root) {
  const svg = s("svg", { role: "img", "aria-label": "Model architecture diagram" });
  const legend = h("div", { class: "diagram-legend", hidden: true },
    h("span", {}, h("i", { style: { background: "var(--gpu-place)" } }), "On GPU (VRAM)"),
    h("span", {}, h("i", { style: { background: "repeating-linear-gradient(135deg, var(--cpu-place) 0 2px, transparent 2px 4px)", outline: "1px solid var(--cpu-place)" } }), "On CPU (RAM)"));
  const fitBtn = h("button", { class: "btn", title: "Fit to view" }, "Fit");
  const zin = h("button", { class: "btn", title: "Zoom in", "aria-label": "Zoom in" }, "+");
  const zout = h("button", { class: "btn", title: "Zoom out", "aria-label": "Zoom out" }, "−");
  root.append(legend, h("div", { class: "diagram-tools" }, zout, zin, fitBtn), svg);

  let full = { x: 0, y: 0, w: W, h: 800 };
  let vb = { ...full };
  let userMoved = false;  // after a zoom or pan, resizes keep the user's view
  const applyVB = () => svg.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.w} ${vb.h}`);
  const fit = () => {
    const r = svg.getBoundingClientRect();
    const aspect = r.width && r.height ? r.width / r.height : 1;
    // fit the width; tall models scroll (drag or wheel) instead of shrinking
    const w = full.w + 20;
    const hgt = w / aspect;
    const y0 = hgt >= full.h ? full.y - (hgt - full.h) / 2 : full.y - 6;
    vb = { x: full.x - 10, y: y0, w, h: hgt };
    userMoved = false;
    applyVB();
  };
  const keepView = () => {  // same centre and zoom, new aspect ratio
    const r = svg.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const cx = vb.x + vb.w / 2, cy = vb.y + vb.h / 2;
    const hgt = vb.w / (r.width / r.height);
    vb = { x: cx - vb.w / 2, y: cy - hgt / 2, w: vb.w, h: hgt };
    applyVB();
  };
  const zoom = (f, cx = vb.x + vb.w / 2, cy = vb.y + vb.h / 2) => {
    const w = Math.min(Math.max(vb.w * f, 120), full.w * 4);
    const k = w / vb.w;
    vb = { x: cx - (cx - vb.x) * k, y: cy - (cy - vb.y) * k, w, h: vb.h * k };
    userMoved = true;
    applyVB();
  };
  fitBtn.onclick = fit;
  zin.onclick = () => zoom(0.8);
  zout.onclick = () => zoom(1.25);
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (!e.ctrlKey && !e.metaKey) {  // plain wheel scrolls, ctrl+wheel zooms
      vb.y += (e.deltaY / svg.getBoundingClientRect().height) * vb.h;
      vb.y = Math.max(full.y - vb.h * 0.5, Math.min(vb.y, full.y + full.h - vb.h * 0.5));
      userMoved = true;
      applyVB();
      return;
    }
    const r = svg.getBoundingClientRect();
    const cx = vb.x + ((e.clientX - r.left) / r.width) * vb.w;
    const cy = vb.y + ((e.clientY - r.top) / r.height) * vb.h;
    zoom(e.deltaY > 0 ? 1.12 : 1 / 1.12, cx, cy);
  }, { passive: false });
  let drag = null;
  svg.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".chipg")) return;
    drag = { x: e.clientX, y: e.clientY, vb: { ...vb } };
    svg.setPointerCapture(e.pointerId);
    svg.classList.add("panning");
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = svg.getBoundingClientRect();
    vb.x = drag.vb.x - ((e.clientX - drag.x) / r.width) * vb.w;
    vb.y = drag.vb.y - ((e.clientY - drag.y) / r.height) * vb.h;
    userMoved = true;
    applyVB();
  });
  svg.addEventListener("pointerup", () => { drag = null; svg.classList.remove("panning"); });
  new ResizeObserver(() => (userMoved ? keepView() : fit())).observe(root);

  let lastKey = "";
  const render = () => {
    const arch = store.get("model");
    if (!arch) return;
    const place = store.get("placement");
    const L = Math.max(0, Math.min(store.get("layer") ?? 0, arch.layers.length - 1));
    const sel = store.get("selectedTensor");
    const idx = store.get("tensorIndex") || new Map();
    const device = place?.placement?.device || null;
    legend.hidden = !device;
    const out = draw(arch, L, idx, device, sel, place?.placement?.output_device);
    svg.replaceChildren(...out.nodes);
    full = { x: 0, y: 0, w: W, h: out.height };
    const key = `${arch.path}`;
    if (key !== lastKey) { lastKey = key; fit(); } else applyVB();
  };
  for (const k of ["model", "placement", "layer", "selectedTensor", "tensorIndex"]) store.on(k, render, { immediate: false });
  render();

  svg.addEventListener("mousemove", (e) => {
    const c = e.target.closest(".chipg");
    if (!c) return hideTip();
    const t = (store.get("tensorIndex") || new Map()).get(c.dataset.name);
    const dev = c.dataset.dev;
    showTip(e.clientX, e.clientY, [
      h("div", {}, h("b", {}, c.dataset.name)),
      t ? h("div", { class: "num" }, `${shape(t.shape)}  ${t.qtype}  ${bytes(t.n_bytes)}`) : "",
      dev ? h("div", {}, `Placed on ${dev}`) : "",
      h("div", { class: "faint" }, "Click to inspect the weights"),
    ]);
  });
  svg.addEventListener("mouseleave", hideTip);
  svg.addEventListener("click", (e) => {
    const c = e.target.closest(".chipg");
    if (c) store.set("selectedTensor", c.dataset.name);
  });
}

// ---------------------------------------------------------------------------
function draw(arch, L, idx, device, sel, outputDevice = null) {
  const nodes = [defs()];
  const layer = arch.layers[L] || { groups: {} };
  const hp = arch.hparams;
  const groupTensors = (gid) => (layer.groups[gid]?.tensors || []);
  const sizes = [...idx.values()].map((t) => t.n_bytes).filter((b) => b > 0);
  const maxB = Math.max(1, ...sizes);

  const chipW = (name, min = 46, max = 150) => {
    const t = idx.get(name);
    if (!t) return min;
    const f = Math.log2(1 + t.n_bytes) / Math.log2(1 + maxB);
    return Math.round(min + (max - min) * f * f);
  };
  const chip = (name, label, x, y, w, hgt = 34, devOverride = null) => {
    const t = idx.get(name);
    const dev = devOverride || (device ? device[name] : null);
    const g = s("g", { class: `chipg${sel === name ? " selected" : ""}`, "data-name": name, "data-dev": dev, tabindex: 0 },
      s("rect", { class: "body", x, y, width: w, height: hgt, rx: 5 }),
      s("text", { x: x + 8, y: y + 15 }, label),
      t ? s("text", { class: "q", x: x + 8, y: y + hgt - 7 }, `${t.qtype} ${bytes(t.n_bytes, 1)}`) : "");
    return g;
  };
  const pill = (label, x, y, name) => {
    const w = Math.max(70, label.length * 7 + 18);
    const g = s("g", { class: `d-pill chipg${sel === name ? " selected" : ""}`, "data-name": name || "", "data-dev": device && name ? device[name] : null },
      s("rect", { class: "body", x, y: y - 11, width: w, height: 22, rx: 11 }),
      s("text", { x: x + w / 2, y: y + 4, "text-anchor": "middle" }, label));
    return { g, w };
  };
  const junction = (y) => s("g", { class: "d-junction" }, s("circle", { cx: RAIL, cy: y, r: 9 }),
    s("path", { d: `M${RAIL - 4.5} ${y}h9M${RAIL} ${y - 4.5}v9` }));
  const wire = (d, cls = "d-wire") => s("path", { class: cls, d });
  const text = (x, y, str, cls = "d-label", anchor = "start") => s("text", { class: cls, x, y, "text-anchor": anchor }, str);
  const nameFor = (gid, role) => groupTensors(gid).find((n) => n.includes(`.${role}.`) || n.endsWith(`.${role}`));

  // Layout is computed top (logits) to bottom (tokens).
  let y = 28;
  const top = [];
  top.push(text(RAIL, y, `Next-token logits over ${num(hp.vocab)} tokens`, "d-strong", "middle"));
  y += 22;
  top.push(wire(`M${RAIL} ${y + 18}V${y}`), s("path", { class: "d-wire", d: `M${RAIL - 4} ${y + 6}L${RAIL} ${y}L${RAIL + 4} ${y + 6}` }));
  y += 22;
  const headOut = arch.head.find((x) => x.id === "output");
  if (headOut) {
    const n = headOut.tensors[0];
    const w = chipW(n, 120, 260);
    top.push(chip(n, headOut.tied ? "Output head (tied to embedding)" : "Output head", RAIL - w / 2, y, w, 38,
      headOut.tied && device ? outputDevice : null));
    top.push(text(RAIL + w / 2 + 10, y + 24, headOut.tied ? "reuses token_embd" : "d_model → vocab", "d-label"));
    y += 38;
  }
  y += 24;
  const headNorm = arch.head.find((x) => x.id === "output_norm");
  if (headNorm) {
    const p = pill("Final norm", RAIL - 45, y, headNorm.tensors[0]);
    top.push(p.g);
    y += 30;
  }
  const railTop = y - 30;
  y += 20;

  // -- block frame -----------------------------------------------------------
  const block = [];
  if (!arch.layers.length) {
    // projector / adapter files have no repeating blocks
    block.push(s("rect", { class: "d-frame", x: 30, y, width: W - 50, height: 60, rx: 16 }),
      text(48, y + 35, "No repeating layers in this file (for example a multimodal projector)", "d-label"));
    y += 86;
  } else {
  const frameTop = y;
  y += 46;
  const present = (gid) => (layer.groups[gid]?.tensors || []).length > 0;
  const seqTop = ["ffn_post_norm", "moe", "ffn_norm", "attn_post_norm", "mixer", "attn_norm"];
  const junctions = [];
  for (const step of seqTop) {
    if (step === "moe") {
      junctions.push(y); y += 26;
      const isMoe = present("moe_experts");
      const m = isMoe ? moeModule() : present("ffn") ? ffnModule() : null;
      if (m) { block.push(...m.draw(y)); m.y = y; y += m.height + 18; block.push(wire(`M${MX + 40} ${m.y}V${junctions.at(-1)}H${RAIL + 9}`)); }
    } else if (step === "mixer") {
      junctions.push(y); y += 26;
      const m = layer.mixer === "ssm"
        ? genericModule(["attention", "ssm"], present("attention") ? "Linear attention (state-space)" : "State-space mixer",
          "fixed-size recurrent state instead of a KV cache")
        : present("attention") ? attnModule() : null;
      if (m) { block.push(...m.draw(y)); m.y = y; y += m.height + 18; block.push(wire(`M${MX + 40} ${m.y}V${junctions.at(-1)}H${RAIL + 9}`)); }
    } else if (present(step)) {
      const name = groupTensors(step)[0];
      const label = { attn_norm: arch.block.groups.find((g) => g.id === "attn_norm")?.label || "Norm",
        ffn_norm: arch.block.groups.find((g) => g.id === "ffn_norm")?.label || "Norm",
        attn_post_norm: "Post-attention norm", ffn_post_norm: "Post-FFN norm" }[step];
      const p = pill(label, MX, y, name);
      block.push(p.g, wire(`M${RAIL} ${y}H${MX}`));
      // wire from pill up into the module above
      block.push(wire(`M${MX + 40} ${y - 11}V${y - 18}`));
      y += 40;
    }
  }
  if (present("other")) {
    const m = genericModule("other", "Other tensors in this layer", "");
    block.push(...m.draw(y)); y += m.height + 18;
  }
  const frameBottom = y + 6;
  block.unshift(s("rect", { class: "d-frame", x: 30, y: frameTop, width: W - 50, height: frameBottom - frameTop, rx: 16 }));
  block.push(text(48, frameTop + 26, `Transformer block`, "d-strong"));
  block.push(text(W - 40, frameTop + 26, `layer ${L} of ${arch.layers.length}, repeated × ${arch.layers.length}`, "d-label", "end"));
  if (arch.block.hybrid) block.push(text(48, frameTop + 42, `hybrid model: this layer mixes with ${layer.mixer}`, "d-module-sub"));
  for (const jy of junctions) block.push(junction(jy));
  y = frameBottom + 26;
  }

  // -- stem ----------------------------------------------------------------
  const stem = [];
  const emb = arch.stem.find((x) => x.id === "token_embd");
  if (emb) {
    const n = emb.tensors[0];
    const w = chipW(n, 120, 260);
    stem.push(chip(n, "Token embedding", RAIL - w / 2, y, w, 38));
    stem.push(text(RAIL + w / 2 + 10, y + 24, `${num(hp.vocab)} tokens → ${num(hp.n_embd)} dims`, "d-label"));
    y += 38;
  }
  const railBottom = y - 38;
  y += 30;
  stem.push(text(RAIL, y, "Input tokens", "d-strong", "middle"));
  y += 14;

  const rail = [wire(`M${RAIL} ${railBottom}V${railTop + 11}`, "d-rail")];
  // frame first, then the residual rail, then everything sitting on top of it
  return { nodes: [...nodes, block[0], ...rail, ...block.slice(1), ...top, ...stem], height: y + 10 };

  // ---- module builders (closures over layer/hp) --------------------------
  function frame(yy, hgt, title, sub) {
    return [s("rect", { class: "d-module", x: MX, y: yy, width: MW, height: hgt, rx: 10 }),
      text(MX + 14, yy + 20, title, "d-module-title"), sub ? text(MX + 14, yy + 35, sub, "d-module-sub") : ""];
  }
  function rowOfChips(names, labels, x0, yy, gap = 10) {
    const out = [];
    let x = x0;
    names.forEach((n, i) => {
      if (!n) return;
      const w = chipW(n);
      out.push(chip(n, labels[i], x, yy, w));
      out.push({ x, w, n });
      x += w + gap;
    });
    return out;
  }
  function attnModule() {
    const roles = ["attn_q", "attn_k", "attn_v", "attn_qkv", "attn_q_a", "attn_q_b", "attn_kv_a_mqa", "attn_kv_b"]
      .filter((r) => nameFor("attention", r));
    const nH = hp.n_head || 0, nKV = hp.n_head_kv || nH;
    const sub = nH ? `${nH} query heads, ${nKV} KV heads, head dim ${hp.head_dim}` + (nKV && nH / nKV > 1 ? `, grouped-query ${nH / nKV}:1` : "") : "";
    const height = 214;
    return { height, draw(yy) {
      const out = frame(yy, height, "Self-attention", sub);
      const chipsY = yy + height - 46;
      const placed = rowOfChips(roles.map((r) => nameFor("attention", r)), roles.map((r) => ROLE_LABEL[r] || r), MX + 14, chipsY);
      const chips = placed.filter((p) => p instanceof Element);
      const boxes = placed.filter((p) => !(p instanceof Element));
      // heads strip
      const hy = yy + 92, hx0 = MX + 14, hw = MW - 28;
      const drawn = Math.min(nH, 64);
      const bw = drawn ? Math.min(22, (hw - drawn * 2) / drawn) : 0;
      const groupSize = nKV ? nH / nKV : 1;
      for (let i = 0; i < drawn; i++) {
        const head = Math.floor((i * nH) / drawn);
        const kv = Math.floor(head / groupSize);
        out.push(s("rect", { class: `d-head${kv % 2 ? "" : " kv0"}`, x: hx0 + i * (bw + 2) + Math.floor(head / groupSize) * 4, y: hy, width: bw, height: 26, rx: 2 }));
      }
      out.push(text(hx0, hy - 8, nH > drawn ? `attention heads (${drawn} of ${nH} drawn), shaded by shared KV head` : "attention heads, shaded by shared KV head", "d-module-sub"));
      for (const b of boxes) out.push(wire(`M${b.x + b.w / 2} ${chipsY}V${hy + 30}`, "d-wire thin"));
      // q/k norms
      const qn = ["attn_q_norm", "attn_k_norm"].map((r) => nameFor("attention", r)).filter(Boolean);
      let nx = MX + MW - 14;
      for (const n of qn.reverse()) {
        const w = 62; nx -= w;
        out.push(chip(n, ROLE_LABEL[n.split(".")[2]] || "norm", nx, chipsY, w));
        nx -= 8;
      }
      // output projection at top right
      const o = nameFor("attention", "attn_output");
      if (o) {
        const w = chipW(o);
        out.push(chip(o, "O", MX + MW - 14 - w, yy + 12, w));
        out.push(wire(`M${MX + MW - 14 - w / 2} ${yy + 46}V${hy}`, "d-wire thin"));
      }
      const sinks = nameFor("attention", "attn_sinks");
      if (sinks) out.push(chip(sinks, "sinks", MX + MW - 170, yy + 12, 56));
      return [...out, ...chips];
    } };
  }
  function ffnModule() {
    const gate = nameFor("ffn", "ffn_gate"), up = nameFor("ffn", "ffn_up"), down = nameFor("ffn", "ffn_down");
    const label = gate ? "SwiGLU feed-forward" : "Feed-forward (MLP)";
    const sub = hp.n_ff ? `${num(hp.n_embd)} → ${num(hp.n_ff)} → ${num(hp.n_embd)}` : "";
    const height = 178;
    return { height, draw(yy) {
      const out = frame(yy, height, label, sub);
      const cy = yy + height - 46;
      const placed = rowOfChips([gate, up].filter(Boolean), gate ? ["gate", "up"] : ["up"], MX + 14, cy);
      const boxes = placed.filter((p) => !(p instanceof Element));
      const nodeX = MX + 14 + (boxes.length ? (boxes.at(-1).x + boxes.at(-1).w - MX - 14) / 2 : 40);
      const nodeY = yy + 84;
      for (const b of boxes) out.push(wire(`M${b.x + b.w / 2} ${cy}L${nodeX} ${nodeY + 10}`, "d-wire thin"));
      out.push(s("circle", { cx: nodeX, cy: nodeY, r: 10, class: "d-module", style: "fill: var(--panel-3)" }));
      out.push(text(nodeX, nodeY + 4, gate ? "⊙" : "σ", "d-strong", "middle"));
      out.push(text(nodeX + 18, nodeY + 4, gate ? "SiLU(gate) × up" : "activation", "d-module-sub"));
      if (down) {
        const w = chipW(down);
        out.push(chip(down, "down", MX + MW - 14 - w, yy + 12, w));
        out.push(wire(`M${nodeX} ${nodeY - 10}V${yy + 29}H${MX + MW - 14 - w}`, "d-wire thin"));
      }
      return [...out, ...placed.filter((p) => p instanceof Element)];
    } };
  }
  function moeModule() {
    const E = hp.n_expert || 0, K = hp.n_expert_used || 0;
    const router = nameFor("moe_router", "ffn_gate_inp");
    const exps = ["ffn_gate_exps", "ffn_up_exps", "ffn_gate_up_exps", "ffn_down_exps"].map((r) => nameFor("moe_experts", r)).filter(Boolean);
    const shared = ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"].map((r) => nameFor("moe_shared", r)).filter(Boolean);
    const cols = Math.min(32, Math.max(8, Math.ceil(Math.sqrt(E * 3))));
    const rows = Math.ceil(E / cols);
    const cell = Math.min(14, (MW - 200) / cols - 2);
    const gridH = rows * (cell + 2);
    const height = 120 + gridH + 50;
    return { height, draw(yy) {
      const out = frame(yy, height, "Mixture of experts", `${E} experts, ${K} active per token${shared.length ? ", plus a shared expert" : ""}`);
      const gx = MX + 120, gy = yy + 56;
      for (let i = 0; i < E; i++) {
        const r = Math.floor(i / cols), c = i % cols;
        // illustrative top-k: spread the active experts across the grid
        const active = K && i % Math.max(1, Math.floor(E / K)) === 0 && Math.floor(i / Math.max(1, Math.floor(E / K))) < K;
        out.push(s("rect", { class: `d-expert${active ? " topk" : ""}`, x: gx + c * (cell + 2), y: gy + r * (cell + 2), width: cell, height: cell, rx: 2 }));
      }
      out.push(text(gx, gy + gridH + 14, `filled: the ${K} experts a token is routed to (illustrative)`, "d-module-sub"));
      if (router) {
        out.push(chip(router, "router", MX + 14, gy, 90));
        out.push(wire(`M${MX + 104} ${gy + 17}H${gx - 4}`, "d-wire thin"));
      }
      const placed = rowOfChips(exps, exps.map((n) => ROLE_LABEL[n.split(".")[2]] || n), MX + 14, yy + height - 46);
      out.push(text(MX + 14, yy + height - 52, "stacked expert weights", "d-module-sub"));
      let sx = MX + MW - 14;
      for (const n of [...shared].reverse()) {
        const w = chipW(n, 46, 90); sx -= w;
        out.push(chip(n, ROLE_LABEL[n.split(".")[2]] || "shared", sx, gy, w)); sx -= 8;
      }
      if (shared.length) out.push(text(MX + MW - 14, gy - 6, "shared expert", "d-module-sub", "end"));
      return [...out, ...placed.filter((p) => p instanceof Element)];
    } };
  }
  function genericModule(gids, title, sub) {
    const names = [].concat(gids).flatMap((g) => groupTensors(g));
    const perRow = 5;
    const rowsN = Math.max(1, Math.ceil(names.length / perRow));
    const height = 52 + rowsN * 44;
    return { height, draw(yy) {
      const out = frame(yy, height, title, sub);
      names.forEach((n, i) => {
        const r = Math.floor(i / perRow), c = i % perRow;
        const w = (MW - 28 - (perRow - 1) * 8) / perRow;
        const role = n.replace(/^blk\.\d+\./, "").replace(/\.(weight|bias)$/, "");
        out.push(chip(n, role.length > 14 ? role.slice(0, 13) + "…" : role, MX + 14 + c * (w + 8), yy + 46 + r * 44, w));
      });
      return out;
    } };
  }
}

function defs() {
  return s("defs", {},
    s("pattern", { id: "hatch-cpu", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
      s("rect", { width: 6, height: 6, fill: "var(--panel-2)" }),
      s("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "var(--cpu-place)", "stroke-width": 2.2, "stroke-opacity": 0.55 })));
}
