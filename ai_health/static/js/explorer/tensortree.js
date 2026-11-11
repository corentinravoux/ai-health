// Every tensor, grouped Stem / Layer i / Head / Other, searchable. Layers are
// <details> that render their rows only when opened (models have ~1000 tensors).
import { store } from "../store.js";
import { h } from "../ui.js";
import { bytes, shape } from "../format.js";

export function mount(root) {
  const search = h("input", { type: "search", placeholder: "Filter tensors (text or /regex/)", "aria-label": "Filter tensors" });
  const count = h("div", { class: "count" });
  const tree = h("div", { class: "tree" });
  root.append(h("div", { class: "tree-head" }, search, count), tree);
  let tensors = [];

  const row = (t) => h("button", { class: "trow", "data-name": t.name, "aria-current": String(store.get("selectedTensor") === t.name), title: t.name },
    h("span", { class: "tn" }, t.layer !== null && t.component !== "vision" ? t.role : t.name),
    h("span", { class: "qt" }, t.qtype),
    h("span", { class: "ts" }, `${shape(t.shape)}, ${bytes(t.n_bytes)}`));

  const matcher = () => {
    const q = search.value.trim();
    if (!q) return null;
    if (q.length > 2 && q.startsWith("/") && q.endsWith("/")) {
      try { const re = new RegExp(q.slice(1, -1), "i"); return (t) => re.test(t.name); } catch { return () => false; }
    }
    const lq = q.toLowerCase();
    return (t) => t.name.toLowerCase().includes(lq);
  };

  const render = () => {
    const m = matcher();
    const shown = m ? tensors.filter(m) : tensors;
    count.textContent = m ? `${shown.length} of ${tensors.length} tensors` : `${tensors.length} tensors`;
    const groups = new Map();
    for (const t of shown) {
      const key = t.layer !== null && t.component !== "vision" ? `Layer ${t.layer}`
        : t.component === "embedding" ? "Input" : (t.component === "output" || t.role === "output_norm") ? "Output" : "Other";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(t);
    }
    const L = store.get("layer");
    const sel = store.get("selectedTensor");
    tree.replaceChildren(...[...groups.entries()].map(([key, ts]) => {
      const open = !!m || key === `Layer ${L}` || key === "Input" || key === "Output" || ts.some((t) => t.name === sel);
      const d = h("details", { open, "data-layer": key.startsWith("Layer") ? key.slice(6) : null },
        h("summary", {}, h("span", {}, key), h("span", { class: "grow" }), h("span", { class: "num" }, bytes(ts.reduce((a, t) => a + t.n_bytes, 0)))));
      const fill = () => { if (d.childElementCount === 1) d.append(...ts.map(row)); };
      if (open) fill();
      d.addEventListener("toggle", () => { if (d.open) fill(); });
      return d;
    }));
  };

  tree.addEventListener("click", (e) => {
    const r = e.target.closest(".trow");
    if (!r) return;
    store.set("selectedTensor", r.dataset.name);
    const t = tensors.find((x) => x.name === r.dataset.name);
    if (t && t.layer !== null) store.set("layer", t.layer);
  });
  let timer = null;
  search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(render, 120); });
  store.on("tensors", (ts) => { tensors = ts || []; render(); });
  store.on("selectedTensor", (name) => {
    for (const r of tree.querySelectorAll(".trow")) r.setAttribute("aria-current", String(r.dataset.name === name));
    const r = tree.querySelector(`.trow[data-name="${CSS.escape(name || "")}"]`);
    if (r) r.scrollIntoView({ block: "nearest" });
    else if (name) render();
  }, { immediate: false });
  store.on("layer", () => { if (!search.value) render(); }, { immediate: false });
}
