// One card per llama-server: model, flags, throughput, slots, memory.
import { store } from "../store.js";
import { h } from "../ui.js";
import { DASH, basename, bytes, pct, rate } from "../format.js";

const ROLE = { server: "server", "router-child": "router instance", router: "router", external: "external" };

// Card content is rebuilt every second, but each card element and its
// "Explore this model" button persist (keyed by pid), so focus and clicks survive.
function body(s) {
  const tp = s.throughput || {};
  const busy = !!tp.busy;
  const hint = s.auth && s.metrics_status !== "ok" && s.slots_status !== "ok"
    ? h("div", { class: "note" }, "Started with ", h("code", {}, "--api-key"), ": its endpoints need the key, so tokens/s cannot be read. Memory and VRAM figures are still live.")
    : s.health === "unreachable" ? h("div", { class: "note" }, `Not answering on port ${s.port}.`)
      : s.metrics_status === "disabled"
        ? h("div", { class: "note" }, "Started without ", h("code", {}, "--metrics"), ": tokens/s is live from /slots. Add the flag for the exact average per request.")
        : null;
  const all = s.slots || [];
  const shown = all.filter((sl) => sl.busy || sl.n_prompt);
  const idle = all.length - shown.length;
  const slots = (shown.length ? shown : all.slice(0, 1)).map((sl) => {
    const used = sl.n_prompt != null ? sl.n_prompt + (sl.n_decoded || 0) : null;
    return h("div", { class: "slot" },
      h("div", {}, `Slot ${sl.id} `, h("b", {}, sl.busy ? "generating" : "idle"),
        sl.busy && sl.n_decoded != null ? `, ${sl.n_decoded} tokens so far` : ""),
      used != null && sl.n_ctx ? h("div", { class: "ctxbar", title: `${used} of ${sl.n_ctx} context tokens in use` },
        h("span", { style: { width: `${Math.min(100, (100 * used) / sl.n_ctx)}%` } })) : "",
      used != null && sl.n_ctx ? h("div", { class: "faint" }, `context ${used} of ${sl.n_ctx} tokens`)
        : sl.n_ctx ? h("div", { class: "faint" }, `context ${sl.n_ctx}`) : "");
  });
  const queue = tp.requests_processing != null && tp.requests_deferred != null
    ? h("span", { class: "faint" }, `${tp.requests_processing} processing, ${tp.requests_deferred} queued`) : "";
  return h("div", { class: "server-body" },
    h("div", { class: "server-top" }, h("span", { class: "health", "data-s": s.health, title: `health: ${s.health}` }),
      h("h3", { title: s.model || "" }, s.name || basename(s.model)), h("span", { class: "grow" }),
      h("span", { class: "chip" }, ROLE[s.role] || s.role, h("b", {}, `:${s.port}`))),
    h("div", { class: "server-tps", "data-busy": String(busy) },
      h("span", { class: "num" }, rate(tp.gen_tps)), h("span", { class: "dim" }, busy ? "tok/s generating" : "tok/s (last)"),
      tp.prompt_tps != null ? h("span", { class: "faint" }, `prompt ${rate(tp.prompt_tps)} tok/s`) : ""),
    queue ? h("div", { class: "queue" }, queue) : "",
    s.flags?.length ? h("div", { class: "flags" }, ...s.flags.map(([k, v]) => h("span", { class: "chip" }, k, h("b", {}, v)))) : "",
    s.pid ? h("div", { class: "kv" },
      h("span", {}, "VRAM"), h("span", {}, "RAM"), h("span", {}, "Swap"), h("span", {}, "CPU"), h("span", {}, "Threads"),
      h("span", { class: "num" }, bytes(s.vram_bytes)), h("span", { class: "num" }, bytes(s.rss)),
      h("span", { class: "num" }, s.swap == null ? DASH : bytes(s.swap)), h("span", { class: "num" }, pct(s.cpu_percent)),
      h("span", { class: "num" }, s.threads ?? DASH)) : "",
    ...slots, idle > 0 && shown.length ? h("div", { class: "slot faint" }, `${idle} more idle slot${idle > 1 ? "s" : ""}`) : "",
    hint);
}

const cards = new Map();  // key -> { el, inner, btn, s }

function keyOf(s) { return s.pid != null ? `pid:${s.pid}` : `ext:${s.host}:${s.port}`; }

function upsert(s) {
  const key = keyOf(s);
  let c = cards.get(key);
  if (!c) {
    c = { el: h("div", { class: "panel server" }), inner: h("div"), btn: null, s };
    c.btn = h("button", { class: "btn", onclick: () => store.set("exploreRequest", { path: c.s.model, pid: c.s.pid }) }, "Explore this model");
    c.el.append(c.inner, c.btn);
    cards.set(key, c);
  }
  c.s = s;
  c.inner.replaceChildren(body(s));
  c.btn.hidden = !s.model;
  return c.el;
}

export function mount(root) {
  store.on("sample", (sample) => {
    const servers = sample.servers || [];
    const stopped = (sample.stopped || []).map((g) => h("div", { class: "panel server stopped" },
      h("div", { class: "server-top" }, h("span", { class: "health", "data-s": "error" }),
        h("h3", { title: g.model || "" }, g.name), h("span", { class: "grow" }),
        h("span", { class: "chip" }, "stopped", h("b", {}, `:${g.port}`))),
      h("div", { class: "note" }, `Stopped at ${new Date(g.stopped_at * 1000).toLocaleTimeString([], { hour12: false })}${g.pid ? ` (pid ${g.pid})` : ""}.`)));
    const els = servers.map(upsert);
    const live = new Set(servers.map(keyOf));
    for (const k of [...cards.keys()]) if (!live.has(k)) cards.delete(k);
    // move/insert without detaching cards that are already in place
    const want = [...els, ...stopped];
    want.forEach((el, i) => { if (root.children[i] !== el) root.insertBefore(el, root.children[i] || null); });
    while (root.children.length > want.length) root.lastChild.remove();
  });
}
