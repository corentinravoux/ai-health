// Weight inspector for the selected tensor: heatmap, histogram, stats.
import { api } from "../api.js";
import { store } from "../store.js";
import { cssVar, h, hideTip, showTip } from "../ui.js";
import { bytes, num, params, sci, shape } from "../format.js";

// Perceptual, colour-blind-safe maps (anchors from matplotlib magma and ColorBrewer RdBu).
const MAGMA = ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55064", "#fb8761", "#fec287", "#fcfdbf"];
const RDBU = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"];

function lut(anchors) {
  const rgb = anchors.map((c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)));
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const f = (i / 255) * (rgb.length - 1);
    const a = Math.floor(f), b = Math.min(a + 1, rgb.length - 1), t = f - a;
    for (let k = 0; k < 3; k++) out[i * 3 + k] = rgb[a][k] + (rgb[b][k] - rgb[a][k]) * t;
  }
  return out;
}
const LUTS = { absmax: lut(MAGMA), mean: lut(RDBU) };

export function mount(root) {
  let reduce = "absmax", logY = true, current = null, reqId = 0;

  const empty = () => root.replaceChildren(h("div", { class: "insp-empty" },
    h("h2", {}, "Weight inspector"),
    h("p", {}, "Click a tensor in the diagram or the list to see its values: a heatmap of the matrix, the distribution of weights, and summary statistics."),
    h("p", { class: "faint" }, "Large tensors are row-sampled so a 150k-token embedding stays fast.")));

  const render = (v, t) => {
    const total = store.get("model")?.summary?.bytes_total || 1;
    const segR = h("div", { class: "seg", role: "group", "aria-label": "Heatmap reduction" },
      ...[["absmax", "Magnitude"], ["mean", "Signed mean"]].map(([k, label]) => h("button", { "aria-pressed": String(k === reduce),
        title: k === "absmax" ? "Largest |value| in each cell" : "Mean value in each cell (blue negative, red positive)",
        onclick: () => { reduce = k; load(current); } }, label)));
    const meta = h("dl", { class: "insp-meta" },
      h("dt", {}, "Shape"), h("dd", { class: "num" }, shape(t?.shape || v.shape)),
      h("dt", {}, "Type"), h("dd", {}, v.qtype),
      h("dt", {}, "Size"), h("dd", {}, `${bytes(t?.n_bytes)}, ${((100 * (t?.n_bytes || 0)) / total).toFixed(2)}% of the model`),
      h("dt", {}, "Parameters"), h("dd", {}, params(t?.n_params)),
      h("dt", {}, "Role"), h("dd", {}, t ? `${t.component.replace("_", " ")}, ${t.role}${t.layer !== null ? `, layer ${t.layer}` : ""}` : "—"));
    const head = [h("h2", {}, v.name), meta];
    if (!v.available) {
      root.replaceChildren(h("div", { class: "insp" }, ...head, h("div", { class: "insp-note" }, v.reason || "No values to show.")));
      return;
    }
    // heatmap
    const canvas = h("canvas", { width: v.heat_w, height: v.heat_h, "aria-label": "Weight heatmap" });
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(v.heat_w, v.heat_h);
    const raw = atob(v.heatmap_b64);
    const L = LUTS[v.reduce];
    for (let i = 0; i < raw.length; i++) {
      const c = raw.charCodeAt(i) * 3;
      img.data[i * 4] = L[c]; img.data[i * 4 + 1] = L[c + 1]; img.data[i * 4 + 2] = L[c + 2]; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    // keep the matrix aspect; tall tensors get a narrower, capped-height canvas
    const aspect = v.heat_w / v.heat_h;
    if (v.heat_h < 24) canvas.style.height = "24px";
    else if (aspect < 0.8) { canvas.style.height = "420px"; canvas.style.width = `${Math.max(48, 420 * aspect)}px`; canvas.style.margin = "0 auto"; }
    const scale = h("canvas", { width: 256, height: 1 });
    const sctx = scale.getContext("2d");
    const simg = sctx.createImageData(256, 1);
    for (let i = 0; i < 256; i++) { simg.data[i * 4] = L[i * 3]; simg.data[i * 4 + 1] = L[i * 3 + 1]; simg.data[i * 4 + 2] = L[i * 3 + 2]; simg.data[i * 4 + 3] = 255; }
    sctx.putImageData(simg, 0, 0);
    const rows = v.rows_total, cols = v.shape[0];
    canvas.addEventListener("mousemove", (e) => {
      const r = canvas.getBoundingClientRect();
      const cx = Math.floor(((e.clientX - r.left) / r.width) * v.heat_w);
      const cy = Math.floor(((e.clientY - r.top) / r.height) * v.heat_h);
      const val = raw.charCodeAt(cy * v.heat_w + cx) / 255 * (v.heat_max - v.heat_min) + v.heat_min;
      const r0 = Math.floor((cy / v.heat_h) * rows), r1 = Math.floor(((cy + 1) / v.heat_h) * rows) - 1;
      const c0 = Math.floor((cx / v.heat_w) * cols), c1 = Math.floor(((cx + 1) / v.heat_w) * cols) - 1;
      showTip(e.clientX, e.clientY, [h("div", { class: "num" }, `rows ${r0}–${Math.max(r0, r1)}, cols ${c0}–${Math.max(c0, c1)}`),
        h("div", {}, `${v.reduce === "absmax" ? "max |w|" : "mean w"} ≈ `, h("span", { class: "num" }, sci(val)))]);
    });
    canvas.addEventListener("mouseleave", hideTip);

    // histogram
    const hist = h("canvas", { "aria-label": "Weight histogram" });
    const logBtn = h("button", { class: "btn", style: { padding: "2px 8px", fontSize: "11.5px" }, onclick: () => { logY = !logY; drawHist(); logBtn.textContent = logY ? "Log scale" : "Linear scale"; } }, logY ? "Log scale" : "Linear scale");
    const drawHist = () => {
      const dpr = devicePixelRatio || 1;
      const w = hist.clientWidth || 340, hh = 110;
      hist.width = w * dpr; hist.height = hh * dpr;
      const c = hist.getContext("2d");
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.clearRect(0, 0, w, hh);
      const counts = v.hist_counts.map((x) => (logY ? Math.log10(1 + x) : x));
      const mx = Math.max(...counts) || 1;
      const bw = w / counts.length;
      c.fillStyle = cssVar("--fg-dim");
      counts.forEach((x, i) => { const bh = (x / mx) * (hh - 16); c.fillRect(i * bw, hh - 14 - bh, Math.max(1, bw - 0.5), bh); });
      // zero line
      const e0 = v.hist_edges[0], e1 = v.hist_edges.at(-1);
      if (e0 < 0 && e1 > 0) { const zx = ((0 - e0) / (e1 - e0)) * w; c.fillStyle = cssVar("--ram"); c.fillRect(zx, 0, 1, hh - 14); }
      c.fillStyle = cssVar("--fg-faint");
      c.font = "300 10px Martian Mono";
      c.fillText(sci(e0), 0, hh - 2);
      const t1 = sci(e1); c.fillText(t1, w - c.measureText(t1).width, hh - 2);
    };
    const st = v.stats;
    const stats = h("div", { class: "stats" }, ...[
      ["mean", sci(st.mean)], ["std", sci(st.std)], ["min", sci(st.min)], ["max", sci(st.max)],
      ["max |w|", sci(st.absmax)], ["rms", sci(st.rms)],
      ["near zero", `${(100 * st.near_zero_frac).toFixed(2)}%`], ["outliers > 6σ", `${(100 * st.outlier_frac).toFixed(3)}%`],
    ].map(([k, val]) => h("div", {}, h("span", {}, k), h("span", { class: "num" }, val))));
    root.replaceChildren(h("div", { class: "insp" }, ...head,
      h("div", {}, h("div", { class: "strip-head" }, h("h3", {}, "Heatmap"), h("span", { class: "grow" }), segR),
        h("div", { class: "heat-wrap" }, canvas,
          h("div", { class: "heat-scale" }, h("span", {}, sci(v.heat_min)), scale, h("span", {}, sci(v.heat_max)))),
        h("div", { class: "insp-note", style: { marginTop: "6px" } },
          `${num(v.heat_w)} × ${num(v.heat_h)} cells`, v.sampled ? `, sampled ${num(v.rows_used)} of ${num(v.rows_total)} rows` : "",
          v.shape.length > 2 ? ", experts stacked top to bottom" : "")),
      h("div", { class: "hist" }, h("div", { class: "strip-head" }, h("h3", {}, "Distribution"), h("span", { class: "grow" }), logBtn), hist),
      stats));
    requestAnimationFrame(drawHist);
  };

  const load = async (name) => {
    current = name;
    if (!name) return empty();
    const id = ++reqId;
    const t = (store.get("tensorIndex") || new Map()).get(name);
    root.replaceChildren(h("div", { class: "insp" }, h("h2", {}, name),
      h("div", { class: "loading" }, h("span", { class: "spinner" }), t && t.n_bytes > 64 * 2 ** 20 ? `Dequantizing ${bytes(t.n_bytes)}…` : "Reading weights…")));
    try {
      const v = await api.get(`/api/model/tensor?name=${encodeURIComponent(name)}&reduce=${reduce}`);
      if (id === reqId) render(v, t);
    } catch (e) {
      if (id === reqId) root.replaceChildren(h("div", { class: "insp" }, h("h2", {}, name), h("div", { class: "insp-note" }, e.message)));
    }
  };
  store.on("selectedTensor", load);
  store.on("model", () => { if (!store.get("selectedTensor")) empty(); });
  store.on("theme", () => current && load(current), { immediate: false });
}
