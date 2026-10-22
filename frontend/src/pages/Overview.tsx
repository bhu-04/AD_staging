import { Link } from "react-router-dom";
import { fmt, pct, ModelRec } from "../api";
import { BarChart } from "../charts";
import { Gate, Stat } from "../ui";

const best = (ms: ModelRec[], f: (m: ModelRec) => number | null, dir: 1 | -1) => ms.filter(m => f(m) !== null).sort((a, b) => dir * ((f(b) as number) - (f(a) as number)))[0];
const nm = (m?: ModelRec) => (m ? `${m.code} ${m.architecture} + ${m.uq_method}` : "n/a");

export default function Overview() {
  return (<Gate>{M => {
    const ms = M.models, ready = ms.filter(m => m.ready).length;
    const acc = best(ms, m => m.metrics.accuracy, 1), f1 = best(ms, m => m.metrics.f1_macro, 1), ece = best(ms, m => m.metrics.ece, -1);
    const top = ms.find(m => m.id === M.best_by_rank);
    const archs = [...new Set(ms.map(m => m.architecture))], uqs = [...new Set(ms.map(m => m.uq_method))];
    const ranked = [...ms].filter(m => m.rank !== null).sort((a, b) => a.rank! - b.rank!);
    const dist = M.dataset_distribution;
    return (<>
      <h2>Alzheimer's Disease Staging | Model Evaluation &amp; Inference</h2>
      <p className="lead">Uncertainty-aware classification of structural MRI slices into {M.classes.join(", ")} using three compact backbones and three uncertainty estimators. All figures are read from the saved evaluation outputs of this repository.</p>
      <div className="grid g4">
        <Stat k="Model configurations" v={`${ready} / ${ms.length}`} s="with all checkpoints present" />
        <Stat k="Best overall (repo ranking)" v={top ? `${top.code}` : "n/a"} s={nm(top)} />
        <Stat k="Highest accuracy" v={pct(acc?.metrics.accuracy)} s={nm(acc)} />
        <Stat k="Highest macro-F1" v={fmt(f1?.metrics.f1_macro)} s={nm(f1)} />
        <Stat k="Lowest ECE (best calibration)" v={fmt(ece?.metrics.ece)} s={nm(ece)} />
      </div>
      <div className="grid g2">
        <section className="panel"><h3>Accuracy by architecture and uncertainty method</h3>
          <BarChart labels={archs} max={1} series={uqs.map(u => ({ name: u, values: archs.map(a => ms.find(m => m.architecture === a && m.uq_method === u)?.metrics.accuracy ?? null) }))} /></section>
        <section className="panel"><h3>Macro-F1 by architecture and uncertainty method</h3>
          <BarChart labels={archs} max={1} series={uqs.map(u => ({ name: u, values: archs.map(a => ms.find(m => m.architecture === a && m.uq_method === u)?.metrics.f1_macro ?? null) }))} /></section>
        <section className="panel"><h3>Calibration (ECE, lower is better)</h3>
          <BarChart labels={archs} series={uqs.map(u => ({ name: u, values: archs.map(a => ms.find(m => m.architecture === a && m.uq_method === u)?.metrics.ece ?? null) }))} /></section>
        <section className="panel"><h3>Dataset split and class distribution</h3>
          {dist ? (<div className="scroll"><table><thead><tr>{Object.keys(dist[0]).map(k => <th key={k}>{k}</th>)}</tr></thead><tbody>{dist.map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{v}</td>)}</tr>)}</tbody></table></div>) : <p className="muted">dataset_distribution.csv not found.</p>}
          <p className="muted small">Counts are images (slices) per split as saved by the pipeline. The test split is small, so metrics carry wide uncertainty.</p></section>
      </div>
      <section className="panel"><h3>Model ranking</h3>
        <div className="scroll"><table><thead><tr><th>Rank</th><th>Configuration</th><th className="num">Accuracy</th><th className="num">Macro-F1</th><th className="num">ROC-AUC</th><th className="num">ECE</th><th className="num">Brier</th><th className="num">Score</th></tr></thead>
          <tbody>{ranked.map(m => (<tr key={m.id} className={m.rank === 1 ? "best" : ""}><td>{m.rank}</td><td>{nm(m)}</td><td className="num">{pct(m.metrics.accuracy)}</td><td className="num">{fmt(m.metrics.f1_macro)}</td><td className="num">{fmt(m.metrics.roc_auc_macro_ovr)}</td><td className="num">{fmt(m.metrics.ece)}</td><td className="num">{fmt(m.metrics.brier)}</td><td className="num">{fmt(m.final_score)}</td></tr>))}</tbody></table></div>
        {M.selection_rule && <p className="muted small">Ranking rule from the saved results: {M.selection_rule}.</p>}</section>
      <div className="row"><Link className="btn" to="/performance">Open evaluation dashboard</Link><Link className="btn sec" to="/inference">Open inference workstation</Link></div>
    </>);
  }}</Gate>);
}
