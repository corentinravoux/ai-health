// Number formatting shared by every view. Unknown is always "—".
const DASH = "—";
const known = (x) => x !== null && x !== undefined && Number.isFinite(x);

export function bytes(n, digits = 2) {
  if (!known(n)) return DASH;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  let v = Math.abs(n);
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  const d = i === 0 ? 0 : v >= 100 ? 0 : v >= 10 ? Math.min(1, digits) : digits;
  return `${(n < 0 ? -v : v).toFixed(d)} ${units[i]}`;
}
export function gib(n, digits = 1) { return known(n) ? (n / 2 ** 30).toFixed(digits) : DASH; }
export function rate(x) { return known(x) ? (x >= 100 ? x.toFixed(0) : x.toFixed(1)) : DASH; }
export function pct(x, digits = 0) { return known(x) ? `${x.toFixed(digits)}%` : DASH; }
export function num(n) {
  if (!known(n)) return DASH;
  return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}
export function params(n) {
  if (!known(n)) return DASH;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 1 : 2)} B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} k`;
  return `${n}`;
}
export function dur(s) {
  if (!known(s)) return DASH;
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}
export function sci(x) {
  if (!known(x)) return DASH;
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return x.toExponential(2);
  return x.toFixed(a < 0.1 ? 4 : 3);
}
export function shape(s) { return Array.isArray(s) ? s.join(" × ") : DASH; }
export function basename(p) { return p ? String(p).split("/").pop() : DASH; }
export { DASH, known };
