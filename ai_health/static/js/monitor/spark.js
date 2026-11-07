// Minimal canvas sparkline (last N values), drawn in a signal colour.
export function drawSpark(canvas, values, color, { max = null, fill = true } = {}) {
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const vals = values.map((v) => (v == null || !Number.isFinite(v) ? null : v));
  const known = vals.filter((v) => v != null);
  if (known.length < 2) return;
  const top = max ?? (Math.max(...known) * 1.15 || 1);
  const n = vals.length;
  const x = (i) => (i / (n - 1)) * w;
  const y = (v) => h - 2 - (v / top) * (h - 6);
  ctx.beginPath();
  let started = false;
  vals.forEach((v, i) => {
    if (v == null) { started = false; return; }
    if (!started) { ctx.moveTo(x(i), y(v)); started = true; } else ctx.lineTo(x(i), y(v));
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  if (fill) {
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, color + "40");
    grad.addColorStop(1, color + "00");
    ctx.lineTo(x(n - 1), h);
    ctx.lineTo(x(vals.findIndex((v) => v != null)), h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }
}
