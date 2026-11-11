// Explorer mode: start screen or workspace (summary, tree, diagram, inspector).
import { api } from "../api.js";
import { store } from "../store.js";
import { h, toast } from "../ui.js";
import * as diagram from "./diagram.js";
import * as inspector from "./inspector.js";
import * as layerstrip from "./layerstrip.js";
import * as openScreen from "./open.js";
import * as placement from "./placement.js";
import * as summary from "./summary.js";
import * as tensortree from "./tensortree.js";

export function mount(root) {
  const start = h("div", { class: "ex-start" });
  const sum = h("header", { class: "ex-sum" });
  const tree = h("aside", { class: "ex-tree", "aria-label": "Tensors" });
  const place = h("div", { class: "place" });
  const dia = h("div", { class: "diagram-wrap" });
  const strip = h("div", { class: "strip" });
  const insp = h("aside", { class: "ex-insp", "aria-label": "Weight inspector" });
  const work = h("div", { class: "ex", hidden: true }, sum, tree, h("section", { class: "ex-center" }, place, dia, strip), insp);
  root.append(start, work);

  const open = async (path) => {
    const arch = await api.post("/api/model/open", { path });
    const tensors = await api.get("/api/model/tensors");
    store.set("tensors", tensors);
    store.set("tensorIndex", new Map(tensors.map((t) => [t.name, t])));
    store.set("selectedTensor", null);
    store.set("layer", 0);
    store.set("placement", null);
    store.set("model", arch);
    start.hidden = true;
    work.hidden = false;
  };
  const opener = openScreen.mount(start, { onOpen: open });
  summary.mount(sum, { onClose: async () => {
    await api.post("/api/model/close").catch(() => {});
    work.hidden = true; start.hidden = false;
    store.set("model", null);
    opener.refresh();
  } });
  tensortree.mount(tree);
  placement.mount(place);
  diagram.mount(dia);
  layerstrip.mount(strip);
  inspector.mount(insp);

  // "Explore this model" from a server card in the monitor
  store.on("exploreRequest", async (req) => {
    if (!req) return;
    document.querySelector('[data-mode-btn="explorer"]').click();
    try {
      if (store.get("model")?.path !== req.path) await open(req.path);
      store.set("placementRequest", { pid: req.pid });
    } catch (e) {
      toast(`Could not open ${req.path}: ${e.message}`, { error: true });
    }
  }, { immediate: false });

  // a model opened with --open is already loaded server-side
  api.get("/api/model/architecture").then(async (arch) => {
    const tensors = await api.get("/api/model/tensors");
    store.set("tensors", tensors);
    store.set("tensorIndex", new Map(tensors.map((t) => [t.name, t])));
    store.set("layer", 0);
    store.set("model", arch);
    start.hidden = true; work.hidden = false;
  }).catch(() => { /* nothing open yet */ });
}
