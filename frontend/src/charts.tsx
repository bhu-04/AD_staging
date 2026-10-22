import { fmt } from "./api";
const PAL = ["#2f6f68", "#8a6d3b", "#4a5f8a", "#9a4d4d", "#6b7f3a"];
export function BarChart({ labels, series, height = 220, max, valueFmt = (v: number) => fmt(v, 2) }: { labels: string[]; series: { name: string; values: (number | null)[] }[]; height?: number; max?: number; valueFmt?: (v: number) => string }) {
  const W = Math.max(360, labels.length * (series.length * 22 + 20) + 60), H = height, pad = { l: 40, r: 8, t: 8, b: 56 };
  const m = max ?? Math.max(0.0001, ...series.flatMap(s => s.values.filter((v): v is number => v !== null)));
  const gw = (W - pad.l - pad.r) / labels.length, bw = Math.min(26, (gw - 10) / series.length);
  const y = (v: number) => pad.t + (H - pad.t - pad.b) * (1 - v / m);
  return (<div className="chart-wrap"><svg role="img" aria-label={`Bar chart of ${series.map(s => s.name).join(", ")}`} viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: Math.min(W, 520) }}>
    {[0, 0.5, 1].map(t => (<g key={t}><line x1={pad.l} x2={W - pad.r} y1={y(m * t)} y2={y(m * t)} stroke="#d9dcd6" /><text x={pad.l - 6} y={y(m * t) + 4} textAnchor="end" className="ax">{valueFmt(m * t)}</text></g>))}
    {labels.map((l, i) => (<g key={l}>{series.map((s, k) => { const v = s.values[i]; const x = pad.l + i * gw + (gw - bw * series.length) / 2 + k * bw; return v === null ? (<text key={k} x={x + bw / 2} y={H - pad.b - 4} textAnchor="middle" className="ax">n/a</text>) : (<g key={k}><title>{`${s.name} · ${l}: ${valueFmt(v)}`}</title><rect x={x} y={y(v)} width={bw - 2} height={Math.max(0, H - pad.b - y(v))} fill={PAL[k % PAL.length]} /></g>); })}
      <text x={pad.l + i * gw + gw / 2} y={H - pad.b + 16} textAnchor="middle" className="ax">{l.length > 14 ? l.slice(0, 13) + "…" : l}</text></g>))}
  </svg>{series.length > 1 && <Legend names={series.map(s => s.name)} />}</div>);
}
export function Legend({ names }: { names: string[] }) { return <ul className="legend">{names.map((n, i) => <li key={n}><span style={{ background: PAL[i % PAL.length] }} />{n}</li>)}</ul>; }
/** One row per class; each dot is one pass/member probability, tick is the mean. */
export function StripPlot({ classes, samples, unit }: { classes: string[]; samples: number[][]; unit: string }) {
  const W = 480, rowH = 44, H = classes.length * rowH + 28, x0 = 56, x1 = W - 12, X = (p: number) => x0 + (x1 - x0) * p;
  return (<div className="chart-wrap"><svg role="img" aria-label={`Probability per ${unit} by class`} viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 360 }}>
    {[0, 0.25, 0.5, 0.75, 1].map(t => (<g key={t}><line x1={X(t)} x2={X(t)} y1={0} y2={H - 20} stroke="#e4e6e1" /><text x={X(t)} y={H - 6} textAnchor="middle" className="ax">{t}</text></g>))}
    {classes.map((c, ci) => { const ps = samples.map(s => s[ci]); const mean = ps.reduce((a, b) => a + b, 0) / ps.length; const cy = ci * rowH + rowH / 2; return (<g key={c}><text x={x0 - 8} y={cy + 4} textAnchor="end" className="ax">{c}</text><line x1={x0} x2={x1} y1={cy} y2={cy} stroke="#d9dcd6" />
      {ps.map((p, i) => <circle key={i} cx={X(p)} cy={cy + ((i % 5) - 2) * 3} r={3.5} fill={PAL[0]} fillOpacity={0.45}><title>{`${unit} ${i + 1}: ${p.toFixed(4)}`}</title></circle>)}
      <line x1={X(mean)} x2={X(mean)} y1={cy - 14} y2={cy + 14} stroke="#1f2a2a" strokeWidth={2}><title>{`mean ${mean.toFixed(4)}`}</title></line></g>); })}
  </svg><p className="muted small">Each dot is one {unit}; the dark tick is the mean.</p></div>);
}
export function ProbBars({ items }: { items: { cls: string; prob: number }[] }) {
  return (<div className="probs" role="list">{items.map(p => (<div className="prob-row" role="listitem" key={p.cls}><span>{p.cls}</span><div className="track"><div className="fill" style={{ width: `${p.prob * 100}%` }} /></div><span className="mono">{p.prob.toFixed(4)}</span></div>))}</div>);
}
