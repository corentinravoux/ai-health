import { h } from "../ui.js";

export function mount(root) {
  root.append(h("div", { class: "empty" }, h("h2", {}, "Monitor"), "Waiting for the first sample…"));
}
