import { useMemo, useState } from "react";
import { METRIC_LABELS, MetricKey, ModelRec, fmt, pct } from "../api";
import { BarChart } from "../charts";
import { Gate } from "../ui";

const COLS: MetricKey[] = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_macro_ovr", "ece", "brier", "latency_s"];
const show = (k: string, v: number | null) => (METRIC_LABELS[k]?.pct ? pct(v) : fmt(v));

export default function Performance() {
  const [arch, setArch] = useState("all"), [uq, setUq] = useState("all"), [sort, setSort] = useState<{ k: string; asc: boolean }>({ k: "rank", asc: true });
  const [sel, setSel] = useState<string>(""), [cmpA, setA] = useState(""), [cmpB, setB] = useState(""), [chartM, setChartM] = useState<MetricKey>("accuracy");
  return (<Gate>{M => {
    const archs = [...new Set(M.models.map(m => m.architecture))], uqs = [...new Set(M.models.map(m => m.uq_method))];
    const rows = useMemo(() => {
      const v = (m: ModelRec): number | string | null => sort.k === "rank" ? m.rank : sort.k === "code" ? m.code : (m.metrics as any)[sort.k];
      return M.models.filter(m => (arch === "all" || m.architecture === arch) && (uq === "all" || m.uq_method === uq))
        .sort((a, b) => { const x = v(a), y = v(b); if (x === null) return 1; if (y === null) return -1; return (x < y ? -1 : x > y ? 1 : 0) * (sort.asc ? 1 : -1); });
    }, [M, arch, uq, sort]);
    const cur = M.models.find(m => m.id === (sel || M.best_by_rank || M.models[0].id))!;
    const A = M.models.find(m => m.id === cmpA), B = M.models.find(m => m.id === cmpB);
    const head = (k: string, label: string, num = true) => (<th className={num ? "num" : ""} aria-sort={sort.k === k ? (sort.asc ? "ascending" : "descending") : "none"}><button onClick={() => setSort(s => ({ k, asc: s.k === k ? !s.asc : true }))}>{label}{sort.k === k ? (sort.asc ? " ▲" : " ▼") : ""}</button></th>);
    return (<>
      <h2>Model performance dashboard</h2>
      <p className="lead">Test-split results from saved evaluation files. "n/a" means the metric is not present in the saved results; it is not zero. Per-model ROC curves are not saved, only the best model's ROC plot.</p>
      <section className="panel"><h3>Experimental matrix</h3>
        <div className="matrix"><span />{["MC Dropout", "Deep Ensemble", "Evidential DL"].map(u => <span className="h" key={u}>{u}</span>)}
          {archs.map(a => (<><span className="h" key={a}>{a}</span>{uqs.map(u => { const m = M.models.find(x => x.architecture === a && x.uq_method === u)!; return (<button key={m.id} className={`cell ${m.ready ? "" : "off"} ${m.id === cur.id ? "sel" : ""}`} onClick={() => setSel(m.id)} aria-pressed={m.id === cur.id}><b>{m.code}</b> {m.rank === 1 && <span className="chip ok">best</span>}<br /><span className="small">{m.ready ? `Acc ${pct(m.metrics.accuracy)}` : "checkpoint missing"}</span></button>); })}</>))}</div></section>
      <section className="panel"><div className="row noprint"><div><label htmlFor="fa">Architecture</label><select id="fa" value={arch} onChange={e => setArch(e.target.value)}><option value="all">All</option>{archs.map(a => <option key={a}>{a}</option>)}</select></div>
        <div><label htmlFor="fu">Uncertainty method</label><select id="fu" value={uq} onChange={e => setUq(e.target.value)}><option value="all">All</option>{uqs.map(a => <option key={a}>{a}</option>)}</select></div></div>
        <div className="scroll" style={{ marginTop: 16 }}><table><thead><tr>{head("rank", "Rank", false)}{head("code", "Config", false)}<th>Model</th>{COLS.map(k => head(k, METRIC_LABELS[k].label))}</tr></thead>
          <tbody>{rows.map(m => (<tr key={m.id} className={`${m.rank === 1 ? "best" : ""} ${m.id === cur.id ? "sel" : ""}`} onClick={() => setSel(m.id)} style={{ cursor: "pointer" }}><td>{m.rank ?? "n/a"}</td><td>{m.code}</td><td>{m.architecture} + {m.uq_method}</td>{COLS.map(k => <td className="num" key={k}>{show(k, m.metrics[k])}</td>)}</tr>))}</tbody></table></div>
        <p className="muted small">Highlighted row: repository's top-ranked configuration. Click a row to inspect it below.</p></section>
      <section className="panel"><div className="row noprint"><div><label htmlFor="cm">Chart metric</label><select id="cm" value={chartM} onChange={e => setChartM(e.target.value as MetricKey)}>{COLS.map(k => <option key={k} value={k}>{METRIC_LABELS[k].label}</option>)}</select></div></div>
        <h3 style={{ marginTop: 16 }}>{METRIC_LABELS[chartM].label} ({METRIC_LABELS[chartM].better === "high" ? "higher" : "lower"} is better)</h3>
        <BarChart labels={rows.map(m => m.code)} series={[{ name: METRIC_LABELS[chartM].label, values: rows.map(m => m.metrics[chartM]) }]} max={METRIC_LABELS[chartM].better === "high" && chartM !== "latency_s" ? 1 : undefined} valueFmt={v => fmt(v, 3)} /></section>
      <section className="panel"><h3>Detail: {cur.code} {cur.architecture} + {cur.uq_method} {!cur.ready && <span className="chip bad">checkpoint missing</span>}</h3>
        <div className="grid g2"><div><dl>{Object.entries(cur.metrics).map(([k, v]) => <><dt key={k}>{METRIC_LABELS[k]?.label ?? k}</dt><dd>{show(k, v)}</dd></>)}</dl></div>
          <div><h3>Class-wise metrics</h3>{cur.classwise.length ? (<table><thead><tr><th>Class</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">Specificity</th><th className="num">F1</th><th className="num">AUC</th></tr></thead><tbody>{cur.classwise.map(c => <tr key={c.cls}><td>{c.cls}</td><td className="num">{fmt(c.precision)}</td><td className="num">{fmt(c.recall)}</td><td className="num">{fmt(c.specificity)}</td><td className="num">{fmt(c.f1)}</td><td className="num">{fmt(c.auc)}</td></tr>)}</tbody></table>) : <p className="muted">Class-wise metrics were not saved for this configuration.</p>}</div></div>
        <div className="grid g2" style={{ marginTop: 16 }}>
          <div><h3>Confusion matrix</h3>{cur.plots.confusion_matrix ? <img className="plot" alt={`Confusion matrix for ${cur.code}`} src={`/api/plots/${cur.plots.confusion_matrix}`} /> : <p className="muted">Not available.</p>}</div>
          <div><h3>Reliability diagram</h3>{cur.plots.reliability ? <img className="plot" alt={`Reliability diagram for ${cur.code}`} src={`/api/plots/${cur.plots.reliability}`} /> : <p className="muted">Not available.</p>}</div></div>
        {M.shared_plots.includes("TEAM4/best_roc_curves.png") && <div style={{ marginTop: 16 }}><h3>ROC curves (saved for the repository's best configuration only)</h3><img className="plot" alt="ROC curves of the best configuration" src="/api/plots/TEAM4/best_roc_curves.png" /></div>}</section>
      <section className="panel"><h3>Model versus model</h3>
        <div className="row noprint">{[["A", cmpA, setA], ["B", cmpB, setB]].map(([n, v, f]: any) => <div key={n}><label htmlFor={`c${n}`}>Model {n}</label><select id={`c${n}`} value={v} onChange={e => f(e.target.value)}><option value="">Select…</option>{M.models.map(m => <option key={m.id} value={m.id}>{m.code} {m.architecture} + {m.uq_method}</option>)}</select></div>)}</div>
        {A && B ? (<table style={{ marginTop: 16 }}><thead><tr><th>Metric</th><th className="num">{A.code}</th><th className="num">{B.code}</th><th>Better</th></tr></thead><tbody>{COLS.map(k => { const a = A.metrics[k], b = B.metrics[k]; const w = a === null || b === null || a === b ? "n/a or tie" : ((a > b) === (METRIC_LABELS[k].better === "high") ? A.code : B.code); return <tr key={k}><td>{METRIC_LABELS[k].label}</td><td className="num">{show(k, a)}</td><td className="num">{show(k, b)}</td><td>{w}</td></tr>; })}</tbody></table>) : <p className="muted">Select two configurations to compare.</p>}</section>
    </>);
  }}</Gate>);
}
