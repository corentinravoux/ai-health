// Header band: model facts, quant mix, actions.
import { api } from "../api.js";
import { store } from "../store.js";
import { h, toast } from "../ui.js";
import { bytes, num, params } from "../format.js";

const QCOLORS = ["var(--fg-dim)", "var(--ram)", "var(--gpu)", "var(--cpu)", "var(--tps)", "var(--fg-faint)", "var(--other)"];

const COMPONENT_LABEL = { embedding: "Embedding", attention: "Attention", ffn: "Feed-forward", moe_experts: "Experts",
  moe_shared: "Shared expert", moe_router: "Router", ssm: "State-space", norm: "Norms", output: "Output head",
  vision: "Vision/audio", other: "Other" };

function detailsPanel(arch) {
  const sm = arch.summary;
  const max = Math.max(...sm.by_component.map((c) => c.params), 1);
  const bars = h("div", { class: "bars" }, ...sm.by_component.map((c) => h("div", {},
    h("span", {}, COMPONENT_LABEL[c.component] || c.component),
    h("i", { style: { width: `${Math.max(1, (100 * c.params) / max)}%` } }),
    h("span", { class: "num" }, `${(100 * c.share).toFixed(1)}%`))));
  const search = h("input", { type: "search", placeholder: "Filter metadata", "aria-label": "Filter metadata" });
  const tbody = h("tbody");
  const entries = Object.entries(arch.metadata || {});
  const fill = () => {
    const q = search.value.trim().toLowerCase();
    tbody.replaceChildren(...entries.filter(([k, v]) => !q || k.toLowerCase().includes(q) || String(v).toLowerCase().includes(q))
      .map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, Array.isArray(v) ? `[${v.join(", ")}]` : String(v)))));
  };
  search.addEventListener("input", fill);
  fill();
  return h("div", { class: "details-panel" },
    h("div", {}, h("h3", {}, "Parameters by component"), bars),
    h("div", {}, h("div", { class: "strip-head" }, h("h3", {}, "Metadata"), h("span", { class: "grow" }), search),
      h("div", { class: "meta-scroll" }, h("table", { class: "meta-table" }, tbody))));
}

export function mount(root, { onClose }) {
  let panel = null;
  const render = async () => {
    const arch = store.get("model");
    if (!arch) return;
    const sm = arch.summary, hp = arch.hparams;
    const total = sm.bytes_total || 1;
    const qs = sm.by_qtype.slice(0, 6);
    let viewers = { netron: false, model_explorer: false };
    try { viewers = await api.get("/api/viewers"); } catch { /* optional */ }
    const launch = (tool) => api.post("/api/viewers/launch", { tool })
      .then((r) => toast(r.message)).catch((e) => toast(e.message, { error: true }));
    root.replaceChildren(
      h("div", { class: "title" }, h("h1", { title: arch.path }, arch.name),
        h("div", { class: "sub" }, `${arch.arch} architecture, ${arch.layers.length} layers${arch.block.hybrid ? ", hybrid" : ""}${hp.n_expert ? `, ${hp.n_expert} experts` : ""}`)),
      h("div", { class: "facts" },
        fact(params(sm.params_total), "parameters"),
        fact(bytes(arch.file_bytes), "on disk"),
        fact(sm.bits_per_weight ? sm.bits_per_weight.toFixed(2) : "—", "bits per weight"),
        fact(num(hp.ctx_train), "training context"),
        fact(bytes(sm.kv_bytes_per_token_f16, 0), "KV per token (f16)")),
      h("div", { class: "qmix", title: "Bytes by quantization type" },
        h("div", { class: "qmix-bar" }, ...qs.map((q, i) => h("span", { style: { width: `${(100 * q.bytes) / total}%`, background: QCOLORS[i] } }))),
        h("div", { class: "qmix-legend" }, ...qs.map((q, i) => h("span", {}, h("i", { style: { background: QCOLORS[i] } }), `${q.qtype} ${Math.round((100 * q.bytes) / total)}%`)))),
      h("div", { class: "actions" },
        h("button", { class: "btn", "aria-expanded": "false", onclick: (e) => {
          const open = e.currentTarget.getAttribute("aria-expanded") !== "true";
          e.currentTarget.setAttribute("aria-expanded", String(open));
          if (panel) { panel.remove(); panel = null; }
          if (open) { panel = detailsPanel(arch); root.after(panel); }
        } }, "Model details"),
        viewers.netron ? h("button", { class: "btn", onclick: () => launch("netron") }, "Open in Netron") : "",
        viewers.model_explorer ? h("button", { class: "btn", onclick: () => launch("model_explorer") }, "Open in Model Explorer") : "",
        h("button", { class: "btn", onclick: onClose }, "Open another model")));
  };
  store.on("model", () => { if (panel) { panel.remove(); panel = null; } render(); });
}

function fact(value, label) {
  return h("div", { class: "fact" }, h("span", { class: "num" }, value), h("span", {}, label));
}
