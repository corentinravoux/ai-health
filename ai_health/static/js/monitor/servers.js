// One card per llama-server: model, flags, throughput, slots, memory.
import { store } from "../store.js";
import { h } from "../ui.js";
import { DASH, basename, bytes, pct, rate } from "../format.js";

const ROLE = { server: "server", "router-child": "router instance", router: "router", external: "external" };

function card(s) {
  const tp = s.throughput || {};
  const busy = !!tp.busy;
  const hint = s.metrics_status === "disabled"
    ? h("div", { class: "note" }, "Started without ", h("code", {}, "--metrics"), ": tokens/s is estimated from /slots. Add the flag for exact numbers.")
    : s.health === "auth" || s.metrics_status === "auth"
      ? h("div", { class: "note" }, "This server needs an API key, so its endpoints cannot be read.")
      : s.health === "unreachable" ? h("div", { class: "note" }, `Not answering on port ${s.port}.`) : null;
  const slots = (s.slots || []).map((sl) => h("div", { class: "slot" },
    `Slot ${sl.id} `, h("b", {}, sl.busy ? "generating" : "idle"),
    sl.busy && sl.n_decoded != null ? `, ${sl.n_decoded} tokens so far` : "",
    sl.n_ctx ? h("span", { class: "faint" }, `, context ${sl.n_ctx}`) : ""));
  const explore = s.model ? h("button", { class: "btn", onclick: () => {
    store.set("exploreRequest", { path: s.model, pid: s.pid });
  } }, "Explore this model") : null;
  return h("div", { class: "panel server" },
    h("div", { class: "server-top" }, h("span", { class: "health", "data-s": s.health, title: `health: ${s.health}` }),
      h("h3", { title: s.model || "" }, s.name || basename(s.model)), h("span", { class: "grow" }),
      h("span", { class: "chip" }, ROLE[s.role] || s.role, h("b", {}, `:${s.port}`))),
    h("div", { class: "server-tps", "data-busy": String(busy) },
      h("span", { class: "num" }, rate(tp.gen_tps)), h("span", { class: "dim" }, busy ? "tok/s generating" : "tok/s (last)"),
      tp.prompt_tps != null ? h("span", { class: "faint" }, `prompt ${rate(tp.prompt_tps)} tok/s`) : ""),
    s.flags?.length ? h("div", { class: "flags" }, ...s.flags.map(([k, v]) => h("span", { class: "chip" }, k, h("b", {}, v)))) : "",
    s.pid ? h("div", { class: "kv" },
      h("span", {}, "VRAM"), h("span", {}, "RAM"), h("span", {}, "Swap"), h("span", {}, "CPU"), h("span", {}, "Threads"),
      h("span", { class: "num" }, bytes(s.vram_bytes)), h("span", { class: "num" }, bytes(s.rss)),
      h("span", { class: "num" }, s.swap == null ? DASH : bytes(s.swap)), h("span", { class: "num" }, pct(s.cpu_percent)),
      h("span", { class: "num" }, s.threads ?? DASH)) : "",
    ...slots, hint, explore);
}

export function mount(root) {
  store.on("sample", (sample) => {
    const servers = sample.servers || [];
    root.replaceChildren(...servers.map(card));
  });
}
