// Header band: model facts, quant mix, actions.
import { api } from "../api.js";
import { store } from "../store.js";
import { h, toast } from "../ui.js";
import { bytes, num, params } from "../format.js";

const QCOLORS = ["var(--fg-dim)", "var(--ram)", "var(--gpu)", "var(--cpu)", "var(--tps)", "var(--fg-faint)", "var(--other)"];

export function mount(root, { onClose }) {
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
        viewers.netron ? h("button", { class: "btn", onclick: () => launch("netron") }, "Open in Netron") : "",
        viewers.model_explorer ? h("button", { class: "btn", onclick: () => launch("model_explorer") }, "Open in Model Explorer") : "",
        h("button", { class: "btn", onclick: onClose }, "Open another model")));
  };
  store.on("model", render);
}

function fact(value, label) {
  return h("div", { class: "fact" }, h("span", { class: "num" }, value), h("span", {}, label));
}
