// The four big readouts: tokens/s, VRAM, RAM, GPU — each with a 120 s sparkline.
import { store } from "../store.js";
import { cssVar, h } from "../ui.js";
import { DASH, gib, pct, rate } from "../format.js";
import { gpu0, series, throughput } from "./data.js";
import { drawSpark } from "./spark.js";

const SPARK_N = 120;

function tile(id, label, sigVar, hero = false) {
  const big = h("span", { class: "big" }, DASH);
  const unit = h("span", { class: "unit" });
  const sub = h("div", { class: "readout-sub" });
  const src = h("span", { class: "src faint" });
  const canvas = h("canvas", { class: "spark", "aria-hidden": "true" });
  const el = h("div", { class: `panel readout${hero ? " hero" : ""}`, id, style: { "--sig": `var(${sigVar})` } },
    h("div", { class: "readout-label" }, h("i"), label, src),
    h("div", { class: "readout-value" }, big, unit),
    sub, canvas);
  return { el, big, unit, sub, src, canvas, sigVar };
}

export function mount(root) {
  const tiles = {
    tps: tile("r-tps", "Tokens per second", "--tps", true),
    vram: tile("r-vram", "VRAM", "--vram"),
    ram: tile("r-ram", "RAM", "--ram"),
    gpu: tile("r-gpu", "GPU", "--gpu"),
  };
  root.append(...Object.values(tiles).map((t) => t.el));
  const spark = { tps: [], vram: [], ram: [], gpu: [] };

  const push = (sample) => {
    const s = series(sample);
    spark.tps.push(s.gen); spark.vram.push(s.vram); spark.ram.push(s.ram); spark.gpu.push(s.gpuUtil);
    for (const k of Object.keys(spark)) if (spark[k].length > SPARK_N) spark[k].shift();
  };

  const render = (sample) => {
    const tp = throughput(sample);
    const g = gpu0(sample);
    const sys = sample.system || {};
    const t = tiles;

    // tokens/s
    t.tps.el.dataset.state = tp.busy ? "busy" : "idle";
    if (tp.n === 0) {
      t.tps.big.textContent = DASH; t.tps.unit.textContent = "";
      t.tps.sub.textContent = "No llama-server found";
      t.tps.src.textContent = "";
    } else {
      t.tps.big.textContent = rate(tp.busy ? tp.gen : tp.last);
      t.tps.unit.textContent = "tok/s";
      t.tps.src.textContent = tp.source === "metrics" ? "exact, from /metrics" : tp.source === "slots" ? "live, from /slots" : "no data";
      t.tps.sub.replaceChildren(
        tp.busy ? "Generating" : "Idle, last rate shown",
        tp.prompt != null ? h("span", {}, ", prompt ", h("span", { class: "num" }, rate(tp.prompt)), " tok/s") : "",
        tp.tokens != null ? h("span", {}, ", ", h("span", { class: "num" }, tp.tokens), " tokens in the last minute") : "");
    }

    // VRAM
    if (!g) {
      t.vram.big.textContent = DASH; t.vram.unit.textContent = "";
      t.vram.sub.textContent = sample.gpu?.reason ? `GPU unavailable: ${sample.gpu.reason}` : "GPU unavailable";
      t.gpu.big.textContent = DASH; t.gpu.unit.textContent = ""; t.gpu.sub.textContent = "";
    } else {
      const frac = g.mem_used / g.mem_total;
      t.vram.el.dataset.state = frac > 0.95 ? "danger" : "";
      t.vram.big.textContent = gib(g.mem_used, 2);
      t.vram.unit.textContent = `of ${gib(g.mem_total, 0)} GiB`;
      t.vram.sub.replaceChildren(h("span", { class: "num" }, pct(frac * 100)), " used, ",
        h("span", { class: "num" }, gib(g.mem_total - g.mem_used, 2)), " GiB free");
      t.gpu.big.textContent = g.util_gpu ?? DASH;
      t.gpu.unit.textContent = g.util_gpu != null ? "% busy" : "";
      t.gpu.sub.replaceChildren(
        g.temp_c != null ? h("span", {}, h("span", { class: "num" }, g.temp_c), " °C") : "",
        g.power_w != null ? h("span", {}, ", ", h("span", { class: "num" }, Math.round(g.power_w)), " of ",
          h("span", { class: "num" }, Math.round(g.power_limit_w || 0)), " W") : "",
        g.sm_clock_mhz != null ? h("span", {}, ", ", h("span", { class: "num" }, g.sm_clock_mhz), " MHz") : "");
    }

    // RAM
    t.ram.big.textContent = gib(sys.ram_used, 1);
    t.ram.unit.textContent = `of ${gib(sys.ram_total, 0)} GiB`;
    const swapping = (sample.warnings || []).some((w) => w.code === "llm_swapping");
    t.ram.el.dataset.state = swapping ? "danger" : "";
    t.ram.sub.replaceChildren("Swap ", h("span", { class: "num" }, gib(sys.swap_used, 2)), " GiB",
      ", CPU ", h("span", { class: "num" }, pct(sys.cpu_percent)));

    drawSpark(t.tps.canvas, spark.tps, cssVar("--tps"));
    drawSpark(t.vram.canvas, spark.vram, cssVar("--vram"), { max: g ? g.mem_total / 2 ** 30 : null });
    drawSpark(t.ram.canvas, spark.ram, cssVar("--ram"), { max: sys.ram_total ? sys.ram_total / 2 ** 30 : null });
    drawSpark(t.gpu.canvas, spark.gpu, cssVar("--gpu"), { max: 100 });
  };

  store.on("history", (hist) => {
    for (const k of Object.keys(spark)) spark[k].length = 0;
    for (const s of (hist || []).slice(-SPARK_N)) push(s);
  });
  store.on("sample", (sample) => { push(sample); render(sample); });
  store.on("theme", () => { const s = store.get("sample"); if (s) render(s); }, { immediate: false });
}
