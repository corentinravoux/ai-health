// VRAM map: the card's memory drawn as 128 MiB cells, coloured by owner.
import { store } from "../store.js";
import { h, hideTip, showTip } from "../ui.js";
import { bytes, gib, pct } from "../format.js";
import { gpu0 } from "./data.js";

const CELL = 128 * 2 ** 20;
const LLAMA_SHADES = ["var(--vram)", "color-mix(in srgb, var(--vram) 70%, var(--fg))",
  "color-mix(in srgb, var(--vram) 60%, var(--bg))", "color-mix(in srgb, var(--vram) 45%, var(--fg))"];
const OTHER_SHADES = ["var(--other)", "color-mix(in srgb, var(--other) 75%, var(--bg))",
  "color-mix(in srgb, var(--other) 75%, var(--fg))", "color-mix(in srgb, var(--other) 55%, var(--bg))"];

export function mount(root) {
  const title = h("h2", {}, "VRAM map");
  const used = h("span", { class: "num" });
  const note = h("span", { class: "dim" }, "Each cell is 128 MiB");
  const grid = h("div", { class: "vram-grid", role: "img" });
  const legend = h("div", { class: "vram-legend" });
  root.append(h("div", { class: "vram-head" }, title, used, h("span", { class: "grow" }), note), grid, legend);
  let cells = [];

  const render = (sample) => {
    const g = gpu0(sample);
    if (!g) {
      grid.replaceChildren(h("div", { class: "dim" }, sample.gpu?.reason ? `GPU unavailable: ${sample.gpu.reason}` : "GPU unavailable"));
      cells = []; legend.replaceChildren(); used.textContent = "";
      return;
    }
    const n = Math.max(1, Math.round(g.mem_total / CELL));
    const cols = n > 64 ? Math.ceil(n / 4) : Math.ceil(n / 3);
    grid.style.setProperty("--cols", cols);
    if (cells.length !== n) {
      cells = Array.from({ length: n }, (_, i) => h("div", { class: "cell", "data-i": i }));
      grid.replaceChildren(...cells);
    }
    const holders = [];
    let li = 0, oi = 0;
    for (const hd of sample.vram_holders || []) {
      holders.push({ ...hd, kind: hd.is_llama ? "llama" : "other",
        color: hd.is_llama ? LLAMA_SHADES[li++ % LLAMA_SHADES.length] : OTHER_SHADES[oi++ % OTHER_SHADES.length] });
    }
    holders.sort((a, b) => (a.kind === b.kind ? b.bytes - a.bytes : a.kind === "llama" ? -1 : 1));
    if (sample.vram_unattributed > 0) {
      holders.push({ name: "Driver and unattributed", bytes: sample.vram_unattributed, kind: "unattributed", color: null });
    }
    // assign cells by cumulative rounding so totals stay honest
    let acc = 0, cursor = 0;
    const owner = new Array(n).fill(null);
    for (const hd of holders) {
      acc += hd.bytes;
      const end = Math.min(n, Math.round(acc / CELL));
      for (; cursor < end; cursor++) owner[cursor] = hd;
    }
    cells.forEach((c, i) => {
      const o = owner[i];
      c.dataset.kind = o ? o.kind : "free";
      c.style.background = o && o.color ? o.color : "";
      c._owner = o;
    });
    const frac = g.mem_used / g.mem_total;
    grid.classList.toggle("danger", frac > 0.95);
    grid.setAttribute("aria-label", `VRAM ${pct(frac * 100)} used`);
    used.textContent = `${gib(g.mem_used, 2)} of ${gib(g.mem_total, 2)} GiB`;
    legend.replaceChildren(...holders.map((hd) => h("span", {},
      h("i", { style: hd.color ? { background: hd.color } : { background: "repeating-linear-gradient(135deg, var(--other) 0 2px, transparent 2px 4px)" } }),
      h("b", {}, hd.name), hd.pid ? h("span", { class: "faint" }, `pid ${hd.pid}`) : "",
      h("span", { class: "num" }, bytes(hd.bytes)))),
      h("span", {}, h("i", { style: { outline: "1px solid var(--line)", background: "var(--free)" } }), "Free",
        h("span", { class: "num" }, bytes(g.mem_total - g.mem_used))));
  };

  grid.addEventListener("mousemove", (e) => {
    const c = e.target.closest(".cell");
    if (!c) return hideTip();
    const o = c._owner;
    showTip(e.clientX, e.clientY, o
      ? [h("div", {}, h("b", {}, o.name), o.pid ? ` pid ${o.pid}` : ""), h("div", { class: "num" }, bytes(o.bytes))]
      : "Free");
  });
  grid.addEventListener("mouseleave", hideTip);
  store.on("sample", render);
}
