// Stacked time-series (uPlot) sharing one time axis and cursor.
import { store } from "../store.js";
import { cssVar, h } from "../ui.js";
import { series } from "./data.js";

const PANELS = [
  { key: "tps", title: "Tokens per second", height: 150,
    lines: [{ label: "generation", field: "gen", color: "--tps", width: 1.8, fill: true }] },
  { key: "vram", title: "VRAM (GiB)", height: 110,
    lines: [{ label: "used", field: "vram", color: "--vram", width: 1.8, fill: true },
            { label: "capacity", field: "vramTotal", color: "--fg-faint", width: 1, dash: [2, 4] }] },
  { key: "ram", title: "RAM and swap (GiB)", height: 110,
    lines: [{ label: "RAM", field: "ram", color: "--ram", width: 1.8, fill: true },
            { label: "swap", field: "swap", color: "--danger", width: 1.2 }] },
  { key: "load", title: "Utilisation (%)", height: 110, max: 100,
    lines: [{ label: "GPU", field: "gpuUtil", color: "--gpu", width: 1.6 },
            { label: "CPU", field: "cpu", color: "--cpu", width: 1.2 }] },
];

export function mount(root) {
  let windowS = 300;
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Time window" });
  for (const [label, s] of [["1 min", 60], ["5 min", 300], ["15 min", 900]]) {
    const b = h("button", { "aria-pressed": String(s === windowS), onclick: () => {
      windowS = s;
      for (const x of seg.children) x.setAttribute("aria-pressed", String(x === b));
      redraw(true);
    } }, label);
    seg.append(b);
  }
  const body = h("div", { class: "charts-body" });
  root.append(h("div", { class: "charts-head" }, h("h2", {}, "History"), h("span", { class: "grow" }), seg), body);

  const data = { t: [] };
  for (const p of PANELS) for (const l of p.lines) data[l.field] = [];
  let plots = [];
  const sync = uPlot.sync("ai-health");

  const append = (sample) => {
    const s = series(sample);
    data.t.push(s.t);
    for (const p of PANELS) for (const l of p.lines) data[l.field].push(s[l.field]);
    const cutoff = s.t - 900;
    while (data.t.length && data.t[0] < cutoff) {
      data.t.shift();
      for (const p of PANELS) for (const l of p.lines) data[l.field].shift();
    }
  };

  const build = () => {
    for (const p of plots) p.plot.destroy();
    body.replaceChildren();
    plots = [];
    const width = Math.max(300, body.clientWidth || root.clientWidth - 24);
    const axisColor = cssVar("--fg-faint");
    const grid = cssVar("--line");
    for (const p of PANELS) {
      const wrap = h("div", { class: "chart" });
      const title = h("div", { class: "chart-title" }, p.title,
        ...p.lines.map((l) => h("span", {}, h("i", { style: { background: cssVar(l.color) } }), l.label)));
      wrap.append(title);
      body.append(wrap);
      const opts = {
        width, height: p.height,
        cursor: { sync: { key: sync.key }, points: { size: 5 } },
        legend: { show: false },
        padding: [22, 8, 0, 0],
        scales: { x: { time: true }, y: { range: (u, mn, mx) => [0, p.max ?? Math.max(1, mx * 1.15)] },
                  p: { range: (u, mn, mx) => [0, Math.max(1, mx * 1.15)] } },
        axes: [
          { stroke: axisColor, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, size: 28,
            font: "11px Instrument Sans", space: 90,
            values: (u, vals) => vals.map((v) => new Date(v * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })) },
          { stroke: axisColor, grid: { stroke: grid, width: 1 }, ticks: { show: false }, size: 50,
            font: "300 10.5px Martian Mono", space: 28 },
        ],
        series: [{}, ...p.lines.map((l) => {
          const c = cssVar(l.color);
          return { label: l.label, stroke: c, width: l.width, dash: l.dash, scale: l.scale || "y",
                   fill: l.fill ? c + "22" : undefined, spanGaps: false, points: { show: false } };
        })],
      };
      const plot = new uPlot(opts, [data.t, ...p.lines.map((l) => data[l.field])], wrap);
      plots.push({ plot, panel: p });
    }
    redraw(true);
  };

  let pending = false;
  const redraw = (force = false) => {
    if (pending && !force) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const now = data.t.length ? data.t[data.t.length - 1] : Date.now() / 1000;
      for (const { plot, panel } of plots) {
        plot.batch(() => {
          plot.setData([data.t, ...panel.lines.map((l) => data[l.field])], false);
          plot.setScale("x", { min: now - windowS, max: now });
        });
      }
    });
  };

  store.on("history", (hist) => {
    for (const k of Object.keys(data)) data[k].length = 0;
    for (const s of hist || []) append(s);
    if (!plots.length) build(); else redraw(true);
  });
  store.on("sample", (s) => { append(s); if (!plots.length) build(); else redraw(); });
  store.on("theme", () => plots.length && build(), { immediate: false });
  new ResizeObserver(() => {
    const w = body.clientWidth;
    for (const { plot } of plots) if (Math.abs(plot.width - w) > 4) plot.setSize({ width: w, height: plot.height });
  }).observe(body);
  store.on("mode", (m) => { if (m === "monitor" && plots.length) requestAnimationFrame(() => {
    const w = body.clientWidth;
    for (const { plot } of plots) plot.setSize({ width: w, height: plot.height });
  }); }, { immediate: false });
}
