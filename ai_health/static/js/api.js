// Talks to the local ai-health server. The per-launch token comes from the
// page URL and is sent on every API call.
import { store } from "./store.js";

const TOKEN = new URLSearchParams(location.search).get("token") || "";

async function request(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: { "X-AI-Health-Token": TOKEN, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const err = new Error((data && data.error) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

export const api = {
  get: (path) => request("GET", path),
  post: (path, body = {}) => request("POST", path, body),
  stream(onSample) {
    let es = null;
    let retry = 1000;
    const open = () => {
      es = new EventSource(`/api/stream?token=${encodeURIComponent(TOKEN)}`);
      es.onopen = () => { retry = 1000; store.set("conn", "live"); };
      es.onmessage = (ev) => {
        try { onSample(JSON.parse(ev.data)); } catch (e) { console.error(e); }
      };
      es.onerror = () => {
        store.set("conn", "reconnecting");
        es.close();
        setTimeout(open, retry);
        retry = Math.min(retry * 2, 10000);
      };
    };
    open();
    return () => es && es.close();
  },
};
