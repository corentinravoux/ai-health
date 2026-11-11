// App shell: mode switch, theme, connection state, data feed.
import { api } from "./api.js";
import { store } from "./store.js";
import { toast } from "./ui.js";

const HISTORY_INIT_S = 900;

function setMode(mode) {
  store.set("mode", mode);
  for (const sec of document.querySelectorAll(".mode")) sec.hidden = sec.id !== mode;
  for (const b of document.querySelectorAll("[data-mode-btn]")) {
    if (b.dataset.modeBtn === mode) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  }
  document.body.dataset.mode = mode;
}

function setTheme(theme, { persist = false } = {}) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("ai-health-theme", theme); } catch { /* private mode */ }
  store.set("theme", theme);
  // the desktop window runs in private mode: the settings file is the durable copy
  if (persist) api.post("/api/settings", { theme }).catch(() => {});
}

function initShell() {
  for (const b of document.querySelectorAll("[data-mode-btn]")) {
    b.addEventListener("click", () => setMode(b.dataset.modeBtn));
  }
  document.getElementById("theme-btn").addEventListener("click", () => {
    setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light", { persist: true });
  });
  addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "1") setMode("monitor");
    if (e.key === "2") setMode("explorer");
  });
  let theme = "dark";
  try { theme = localStorage.getItem("ai-health-theme") || "dark"; } catch { /* ignore */ }
  setTheme(theme);
  api.get("/api/settings").then((st) => { if (st?.theme && st.theme !== theme) setTheme(st.theme); }).catch(() => {});
  store.on("conn", (state) => {
    const el = document.getElementById("conn");
    el.dataset.state = state;
    el.querySelector(".conn-label").textContent = state === "live" ? "Live" : "Reconnecting";
  });
  setMode(location.hash === "#explorer" ? "explorer" : "monitor");
}

async function initData() {
  try {
    store.set("meta", await api.get("/api/meta"));
    store.set("history", await api.get(`/api/history?seconds=${HISTORY_INIT_S}`));
  } catch (e) {
    toast(`Could not reach the ai-health backend: ${e.message}`, { error: true });
  }
  api.stream((sample) => store.set("sample", sample));
}

async function initViews() {
  const views = [
    ["./monitor/index.js", "monitor"],
    ["./explorer/index.js", "explorer"],
  ];
  for (const [path, id] of views) {
    try {
      const mod = await import(path);
      mod.mount(document.getElementById(id));
    } catch (e) {
      console.error(`view ${id}: ${e?.stack || e}`);
    }
  }
}

initShell();
await initViews();
initData();
