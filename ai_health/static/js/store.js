// Tiny pub/sub store. Keys: sample, history, meta, mode, model, selectedTensor,
// placement, conn.
const state = new Map();
const subs = new Map();

export const store = {
  get(key) { return state.get(key); },
  set(key, value) {
    state.set(key, value);
    for (const fn of subs.get(key) || []) {
      try { fn(value); } catch (e) { console.error(`store listener for ${key}`, e); }
    }
  },
  on(key, fn, { immediate = true } = {}) {
    if (!subs.has(key)) subs.set(key, new Set());
    subs.get(key).add(fn);
    if (immediate && state.has(key)) fn(state.get(key));
    return () => subs.get(key).delete(fn);
  },
};
