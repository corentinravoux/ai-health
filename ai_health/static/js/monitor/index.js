// Monitor mode layout.
import { store } from "../store.js";
import { h } from "../ui.js";
import { gib } from "../format.js";
import * as charts from "./charts.js";
import * as processes from "./processes.js";
import * as readouts from "./readouts.js";
import * as servers from "./servers.js";
import * as vrammap from "./vrammap.js";
import * as warnings from "./warnings.js";

export function mount(root) {
  const machine = h("span", { class: "machine" });
  const r = h("div", { class: "readouts" });
  const v = h("section", { class: "panel vram", "aria-label": "VRAM map" });
  const c = h("section", { class: "panel charts", "aria-label": "History" });
  const w = h("div", { class: "warnings" });
  const sv = h("div", { class: "servers", style: { display: "flex", flexDirection: "column", gap: "var(--s4)" } });
  const pr = h("section", { class: "panel procs" });
  const empty = h("div", { class: "panel mon-empty", hidden: true },
    h("h2", {}, "No llama.cpp process running"),
    h("p", { class: "dim" }, "Start a server and it appears here within a second. With ", h("code", {}, "--metrics"),
      " the tokens/s readout is exact."),
    h("pre", {}, "~/Software/llama.cpp/build/bin/llama-server \\\n  -m ~/models/model.gguf -ngl 99 -c 8192 --metrics"));
  root.append(h("div", { class: "mon" },
    h("header", { class: "mon-head" }, h("h1", {}, "Monitor"), machine),
    r, v, c, h("aside", { class: "side" }, w, empty, sv, pr)));
  readouts.mount(r); vrammap.mount(v); charts.mount(c);
  warnings.mount(w); servers.mount(sv); processes.mount(pr);

  store.on("sample", (sample) => {
    const g = sample.gpu?.devices?.[0];
    machine.textContent = [g ? `${g.name}, ${gib(g.mem_total, 0)} GiB VRAM` : "No NVIDIA GPU",
      `${gib(sample.system?.ram_total, 0)} GiB RAM`].join(", ");
    empty.hidden = (sample.servers || []).length > 0 || (sample.processes || []).length > 0 || (sample.stopped || []).length > 0;
  });
}
