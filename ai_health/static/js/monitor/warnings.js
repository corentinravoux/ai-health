// Warnings, most severe first, each with what to do about it.
import { store } from "../store.js";
import { h } from "../ui.js";

export function mount(root) {
  store.on("sample", (sample) => {
    const ws = sample.warnings || [];
    root.hidden = ws.length === 0;
    root.replaceChildren(...ws.map((w) => h("div", { class: "warning", "data-level": w.level, role: w.level === "danger" ? "alert" : null },
      h("i"), h("div", {}, w.text), w.hint ? h("div", { class: "hint" }, w.hint) : "")));
  });
}
