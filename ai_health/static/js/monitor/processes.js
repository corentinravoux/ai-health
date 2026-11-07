// All llama.cpp processes as a small tree table (router -> instances).
import { store } from "../store.js";
import { h } from "../ui.js";
import { DASH, basename, bytes, pct } from "../format.js";

export function mount(root) {
  const tbody = h("tbody");
  const table = h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Process"), h("th", {}, "VRAM"), h("th", {}, "RAM"), h("th", {}, "CPU"), h("th", {}, "Threads"))),
    tbody);
  root.append(h("div", { class: "panel-head" }, h("h2", {}, "llama.cpp processes")), table);
  store.on("sample", (sample) => {
    const procs = sample.processes || [];
    root.hidden = procs.length === 0;
    const byPid = new Map(procs.map((p) => [p.pid, p]));
    const roots = procs.filter((p) => !byPid.has(p.ppid));
    const rows = [];
    const add = (p, child) => {
      rows.push(h("tr", { class: child ? "child" : "" },
        h("td", { title: p.model || "" }, p.name, h("span", { class: "role" }, `${p.pid}`), p.model ? h("div", { class: "faint" }, basename(p.model)) : ""),
        h("td", { class: "num" }, bytes(p.vram_bytes)), h("td", { class: "num" }, bytes(p.rss)),
        h("td", { class: "num" }, pct(p.cpu_percent)), h("td", { class: "num" }, p.threads ?? DASH)));
      for (const c of p.children || []) if (byPid.has(c)) add(byPid.get(c), true);
    };
    roots.forEach((p) => add(p, false));
    tbody.replaceChildren(...rows);
  });
}
