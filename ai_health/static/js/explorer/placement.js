// Placement panel: pick a running server (or type flags) -> where every tensor
// lives and whether it fits in VRAM.
import { api } from "../api.js";
import { store } from "../store.js";
import { h } from "../ui.js";
import { basename, bytes, gib } from "../format.js";

export function mount(root) {
  const select = h("select", { "aria-label": "Placement source" });
  const flags = h("input", { type: "text", placeholder: "-ngl 99 -c 8192 --n-cpu-moe 4", "aria-label": "llama.cpp flags", spellcheck: "false" });
  const clear = h("button", { class: "btn" }, "Hide overlay");
  const result = h("div", { class: "place-body" });
  root.append(h("div", { class: "place-row" }, h("h3", {}, "GPU placement"), select, flags, clear), result);
  let lastServers = null;
  let timer = null;

  const run = async () => {
    const v = select.value;
    flags.hidden = v !== "flags";
    if (v === "none") { store.set("placement", null); result.replaceChildren(h("div", { class: "dim", style: { fontSize: "12.5px" } },
      "Choose a running llama-server or type flags to see which layers land on the GPU.")); return; }
    const body = v === "flags" ? { flags: flags.value } : { pid: Number(v) };
    try {
      store.set("placement", await api.post("/api/model/placement", body));
    } catch (e) {
      result.replaceChildren(h("div", { class: "insp-note" }, e.message));
    }
  };
  select.addEventListener("change", run);
  flags.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 300); });
  clear.onclick = () => { select.value = "none"; run(); };

  const fillServers = (sample) => {
    const servers = (sample?.servers || []).filter((s) => s.pid && s.args);
    const key = servers.map((s) => s.pid).join(",");
    if (key === lastServers) return;
    lastServers = key;
    const cur = select.value;
    select.replaceChildren(h("option", { value: "none" }, "No overlay"),
      ...servers.map((s) => h("option", { value: String(s.pid) }, `llama-server :${s.port}, ${basename(s.model)}`)),
      h("option", { value: "flags" }, "Type flags…"));
    if ([...select.options].some((o) => o.value === cur)) select.value = cur;
  };
  store.on("sample", fillServers);
  store.on("placementRequest", (req) => {
    if (!req) return;
    fillServers(store.get("sample"));
    if (req.pid && [...select.options].some((o) => o.value === String(req.pid))) select.value = String(req.pid);
    else { select.value = "flags"; flags.value = req.flags || ""; }
    run();
  }, { immediate: false });
  store.on("model", () => { if (select.value !== "none") run(); }, { immediate: false });

  store.on("placement", (r) => {
    if (!r) { flags.hidden = select.value !== "flags"; return; }
    const v = r.vram, p = r.placement;
    const cap = v.capacity || v.total || 1;
    const scaleMax = Math.max(cap, v.total);
    const pctW = (b) => `${(100 * b) / scaleMax}%`;
    const short = v.capacity ? v.total - v.capacity : null;
    result.replaceChildren(
      h("div", { class: "vbar", title: "Weights on GPU, KV cache, compute buffer, against VRAM capacity" },
        h("span", { style: { width: pctW(v.weights), background: "var(--gpu-place)" } }),
        h("span", { style: { width: pctW(v.kv_cache), background: "color-mix(in srgb, var(--gpu-place) 55%, var(--bg))" } }),
        h("span", { style: { width: pctW(v.compute_buffer), background: "repeating-linear-gradient(135deg, var(--gpu-place) 0 2px, transparent 2px 5px)" } }),
        v.capacity ? h("span", { class: "cap", style: { left: `calc(${pctW(v.capacity)} - 1px)` } }) : ""),
      h("div", { class: "place-res" },
        h("span", {}, "Weights ", h("b", {}, bytes(v.weights)), ", KV cache ", h("b", {}, bytes(v.kv_cache)), ", compute ", h("b", {}, bytes(v.compute_buffer))),
        v.capacity ? (short > 0
          ? h("b", { class: "bad" }, `Does not fit: ${gib(short, 2)} GiB over ${gib(v.capacity, 0)} GiB`)
          : h("b", { class: "good" }, `Fits, ${gib(-short, 2)} GiB to spare`)) : "",
        h("span", {}, "CPU keeps ", h("b", {}, bytes(p.cpu_weight_bytes))),
        p.estimated ? h("span", { class: "badge", title: "Depends on --fit or free VRAM at load time; computed from current free VRAM." }, "estimated") : ""),
      h("details", {}, h("summary", {}, `How this was worked out (${r.source})`),
        h("ul", { class: "rules" }, ...[...p.rules, ...v.notes].map((x) => h("li", {}, x)))));
  });
  run();
}
