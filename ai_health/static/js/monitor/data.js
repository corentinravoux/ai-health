// Derive the numbers the monitor shows from a raw sample.
export function gpu0(sample) {
  return sample?.gpu?.devices?.[0] || null;
}

// Aggregate throughput over all servers: summed while generating.
export function throughput(sample) {
  const servers = sample?.servers || [];
  let gen = null, prompt = null, busy = false, source = "none", tokens = null;
  for (const s of servers) {
    const t = s.throughput || {};
    if (t.source && t.source !== "none") {
      if (source === "none" || t.source === "metrics") source = t.source;
    }
    if (t.busy) {
      busy = true;
      if (t.gen_tps != null) gen = (gen || 0) + t.gen_tps;
    }
    if (t.prompt_tps != null) prompt = Math.max(prompt || 0, t.prompt_tps);
    if (t.tokens_last_min != null) tokens = (tokens || 0) + t.tokens_last_min;
  }
  // last known rate (shown dimmed) when idle
  let last = null;
  for (const s of servers) if (s.throughput?.gen_tps != null) last = (last || 0) + s.throughput.gen_tps;
  return { gen, prompt, busy, source, tokens, last, n: servers.length };
}

export function series(sample) {
  const g = gpu0(sample);
  const sys = sample?.system || {};
  const tp = throughput(sample);
  const gib = 2 ** 30;
  return {
    t: sample.t,
    gen: tp.n === 0 || tp.source === "none" ? null : tp.busy ? (tp.gen ?? 0) : 0,
    prompt: tp.n === 0 || tp.source !== "metrics" ? null : tp.prompt,
    vram: g ? g.mem_used / gib : null,
    vramTotal: g ? g.mem_total / gib : null,
    ram: sys.ram_used != null ? sys.ram_used / gib : null,
    swap: sys.swap_used != null ? sys.swap_used / gib : null,
    gpuUtil: g ? g.util_gpu : null,
    cpu: sys.cpu_percent ?? null,
  };
}
