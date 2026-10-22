import { Link } from "react-router-dom";
import { Prediction } from "../api";
import { ProbBars, StripPlot } from "../charts";
import { useApp } from "../ui";

export function UncertaintyView({ r }: { r: Prediction }) {
  const st = r.stages.find(s => s.key === "uncertainty")!, d = st.detail!, classes = r.probabilities.map(p => p.cls);
  return (<>
    <p><b>{st.method}</b> on {r.model.code} {r.model.architecture}.</p>
    <div className="scroll"><table><thead><tr><th>Metric</th><th className="num">Value</th><th>Range</th><th>Definition</th></tr></thead><tbody>{st.metrics!.map(m => <tr key={m.key}><td>{m.label}</td><td className="num mono">{m.value.toFixed(4)}</td><td>{m.range}</td><td style={{ whiteSpace: "normal" }}>{m.definition}</td></tr>)}</tbody></table></div>
    <h3 style={{ marginTop: 16 }}>Class probabilities (aggregate)</h3><ProbBars items={r.probabilities} />
    {st.detail_kind === "per_pass" && <><h3 style={{ marginTop: 16 }}>Per-pass probabilities ({d.n_passes} stochastic passes)</h3><StripPlot classes={classes} samples={d.per_pass_probs} unit="pass" />
      <p className="small muted">Per-class std: {classes.map((c, i) => `${c} ${d.prob_std[i].toFixed(4)}`).join(" · ")}. Pass argmax counts: {classes.map((c, i) => `${c} ${d.per_pass_pred.filter((x: number) => x === i).length}`).join(" · ")}.</p></>}
    {st.detail_kind === "per_member" && <><h3 style={{ marginTop: 16 }}>Per-member predictions ({d.n_members} members)</h3><StripPlot classes={classes} samples={d.per_member_probs} unit="member" />
      <table><thead><tr><th>Member</th><th>Predicted</th>{classes.map(c => <th key={c} className="num">p({c})</th>)}</tr></thead><tbody>{d.per_member_probs.map((p: number[], i: number) => <tr key={i}><td>{i}</td><td>{classes[d.per_member_pred[i]]}</td>{p.map((v, j) => <td key={j} className="num">{v.toFixed(4)}</td>)}</tr>)}</tbody></table></>}
    {st.detail_kind === "dirichlet" && <><h3 style={{ marginTop: 16 }}>Dirichlet parameters</h3><table><thead><tr><th>Class</th><th className="num">Evidence</th><th className="num">alpha</th></tr></thead><tbody>{classes.map((c, i) => <tr key={c}><td>{c}</td><td className="num">{d.evidence[i].toFixed(4)}</td><td className="num">{d.alpha[i].toFixed(4)}</td></tr>)}</tbody></table></>}
  </>);
}

export default function Uncertainty() {
  const { result } = useApp();
  return (<>
    <h2>Uncertainty analysis</h2>
    <p className="lead">Shows only the quantities defined for the uncertainty method of the most recent inference result.</p>
    {result ? <section className="panel"><UncertaintyView r={result} /></section> : <div className="panel"><p>No inference result yet. Run a prediction first.</p><Link className="btn" to="/inference">Go to inference workstation</Link></div>}
    <section className="panel"><h3>Confidence versus uncertainty</h3>
      <p><b>Confidence</b> is the largest class probability the model outputs. <b>Uncertainty</b> describes how spread out or unstable the prediction is: entropy measures spread of the predictive distribution, mutual information and ensemble disagreement measure instability across stochastic passes or members (epistemic), expected entropy measures per-pass spread (aleatoric), and vacuity measures how little evidence the Dirichlet holds.</p>
      <p>A confident prediction can be wrong, and low uncertainty does not guarantee a clinically correct result. Output probabilities are not guaranteed to be calibrated; see ECE and reliability diagrams on the performance page.</p></section>
  </>);
}
