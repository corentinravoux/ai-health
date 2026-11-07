// Small DOM helpers shared by views.
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const svgNS = "http://www.w3.org/2000/svg";
export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

let toastTimer = null;
export function toast(message, { error = false, ms = 3500 } = {}) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.classList.toggle("error", error);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), ms);
}

let tip = null;
export function showTip(x, y, content) {
  if (!tip) { tip = h("div", { class: "tooltip", role: "tooltip" }); document.body.append(tip); }
  tip.replaceChildren(...(Array.isArray(content) ? content : [content]));
  tip.hidden = false;
  const pad = 14;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(x + pad, innerWidth - r.width - 8)}px`;
  tip.style.top = `${Math.min(y + pad, innerHeight - r.height - 8)}px`;
}
export function hideTip() { if (tip) tip.hidden = true; }

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
