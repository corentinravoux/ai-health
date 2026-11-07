// Explorer start screen: pick a GGUF (native dialog, drop, path, recent, found).
import { api } from "../api.js";
import { h } from "../ui.js";
import { basename, bytes } from "../format.js";

export function mount(root, { onOpen }) {
  const input = h("input", { type: "text", placeholder: "/home/you/models/model.gguf", "aria-label": "Path to a .gguf file", spellcheck: "false" });
  const status = h("div", { class: "loading", hidden: true });
  const recent = h("ul");
  const found = h("ul");
  const pickBtn = h("button", { class: "btn", hidden: true }, "Choose file…");
  const openBtn = h("button", { class: "btn primary" }, "Open");
  const drop = h("div", { class: "drop" },
    h("div", { class: "drop-row" }, input, openBtn, pickBtn),
    h("div", { class: "hint" }, "Drop a .gguf file here in the desktop window, or paste its path. Only the header is read; weights load one tensor at a time when you inspect them."));
  root.append(h("div", { class: "ex-open" },
    h("h1", {}, "Explore a model"),
    h("p", {}, "See the architecture of a GGUF model as a diagram, inspect every weight tensor, and check which layers your llama-server flags put on the GPU."),
    drop, status,
    h("div", { class: "files" },
      h("div", {}, h("h2", {}, "Recent"), recent),
      h("div", {}, h("h2", {}, "Found on this PC"), found))));

  const go = async (path) => {
    if (!path) return;
    status.hidden = false;
    status.replaceChildren(h("span", { class: "spinner" }), `Reading ${basename(path)}…`);
    try {
      await onOpen(path);
      status.hidden = true;
    } catch (e) {
      status.replaceChildren(h("span", { style: { color: "var(--danger)" } }, e.message));
    }
  };
  openBtn.onclick = () => go(input.value.trim());
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(input.value.trim()); });

  const enablePicker = () => {
    if (window.pywebview?.api?.pick_gguf) {
      pickBtn.hidden = false;
      pickBtn.onclick = async () => { const p = await window.pywebview.api.pick_gguf(); if (p) { input.value = p; go(p); } };
    }
  };
  enablePicker();
  addEventListener("pywebviewready", enablePicker);
  window.aiHealthDropped = (p) => { input.value = p; go(p); };
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    // In the desktop window the native handler reports the real path; a plain
    // browser does not expose it, so ask for the path instead.
    setTimeout(() => { if (!input.value) status.hidden = false, status.replaceChildren("This browser does not expose dropped file paths. Paste the path instead."); }, 300);
  });

  const list = (ul, items, empty) => {
    ul.replaceChildren(...(items.length ? items.map((f) => h("li", {},
      h("button", { onclick: () => go(f.path), title: f.path, disabled: f.exists === false },
        h("span", { class: "name" }, basename(f.path)), h("span", { class: "num" }, f.exists === false ? "missing" : bytes(f.bytes)))))
      : [h("li", { class: "none" }, empty)]));
  };
  const refresh = async () => {
    try {
      const f = await api.get("/api/model/files");
      list(recent, f.recent, "Models you open appear here.");
      list(found, f.found, `No .gguf files in ${f.search_dirs.slice(0, 2).join(" or ")}.`);
    } catch (e) {
      list(recent, [], e.message);
    }
  };
  refresh();
  return { refresh };
}
