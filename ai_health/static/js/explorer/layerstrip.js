// One cell per layer under the diagram; colour = placement or size; click to focus.
import { store } from "../store.js";
import { h, hideTip, showTip } from "../ui.js";
import { bytes } from "../format.js";

export function mount(root) {
  let mode = "placement";
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Colour layers by" });
  const modes = [["placement", "Placement"], ["bytes", "Size"]];
  for (const [m, label] of modes) {
    const b = h("button", { "aria-pressed": String(m === mode), onclick: () => { mode = m; for (const x of seg.children) x.setAttribute("aria-pressed", String(x === b)); render(); } }, label);
    seg.append(b);
  }
  const info = h("span", { class: "dim" });
  const cells = h("div", { class: "strip-cells", role: "listbox", "aria-label": "Layers" });
  const axis = h("div", { class: "strip-axis" });
  root.append(h("div", { class: "strip-head" }, h("h3", {}, "Layers"), info, h("span", { class: "grow" }), seg), cells, axis);

  const render = () => {
    const arch = store.get("model");
    if (!arch) return;
    const place = store.get("placement")?.placement;
    const L = store.get("layer") ?? 0;
    const maxB = Math.max(1, ...arch.layers.map((l) => l.bytes));
    const byIdx = new Map((place?.layers || []).map((l) => [l.index, l]));
    cells.replaceChildren(...arch.layers.map((layer) => {
      const p = byIdx.get(layer.index);
      const c = h("button", { class: "strip-cell", role: "option", "aria-current": String(layer.index === L),
        "aria-label": `Layer ${layer.index}`, "data-i": layer.index, "data-mixer": layer.mixer });
      if (mode === "placement" && p) c.dataset.dev = p.device;
      else c.style.background = `color-mix(in srgb, var(--fg-dim) ${Math.round(25 + 75 * layer.bytes / maxB)}%, transparent)`;
      return c;
    }));
    const n = arch.layers.length;
    root.hidden = n === 0;  // e.g. a multimodal projector (mmproj) file has no repeating layers
    if (n === 0) return;
    axis.replaceChildren(h("span", {}, "0"), h("span", {}, String(Math.floor(n / 2))), h("span", {}, String(n - 1)));
    const counts = { GPU: 0, CPU: 0, MIXED: 0 };
    for (const l of place?.layers || []) counts[l.device]++;
    info.textContent = place ? `${counts.GPU} on GPU, ${counts.CPU} on CPU, ${counts.MIXED} split` : `${n} layers, click one to show its tensors`;
  };
  cells.addEventListener("click", (e) => {
    const c = e.target.closest(".strip-cell");
    if (c) store.set("layer", Number(c.dataset.i));
  });
  cells.addEventListener("keydown", (e) => {
    const L = store.get("layer") ?? 0;
    const n = store.get("model")?.layers.length || 0;
    if (e.key === "ArrowRight" && L < n - 1) store.set("layer", L + 1);
    if (e.key === "ArrowLeft" && L > 0) store.set("layer", L - 1);
  });
  cells.addEventListener("mousemove", (e) => {
    const c = e.target.closest(".strip-cell");
    if (!c) return hideTip();
    const arch = store.get("model");
    const layer = arch.layers[Number(c.dataset.i)];
    const p = (store.get("placement")?.placement?.layers || []).find((l) => l.index === layer.index);
    showTip(e.clientX, e.clientY, [h("div", {}, h("b", {}, `Layer ${layer.index}`), ` ${layer.mixer}, ${layer.ffn}`),
      h("div", { class: "num" }, bytes(layer.bytes)),
      p ? h("div", {}, `${p.device}: ${bytes(p.gpu_bytes)} GPU, ${bytes(p.cpu_bytes)} CPU`) : ""]);
  });
  cells.addEventListener("mouseleave", hideTip);
  for (const k of ["model", "placement", "layer"]) store.on(k, render, { immediate: false });
  render();
}
