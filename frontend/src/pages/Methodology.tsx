import { useEffect, useState } from "react";
import { Methodology as Meth, getMethodology } from "../api";
import { ErrorBox, Spinner } from "../ui";

export default function Methodology() {
  const [m, setM] = useState<Meth | null>(null), [e, setE] = useState<string | null>(null);
  useEffect(() => { getMethodology().then(setM).catch(x => setE(x.message)); }, []);
  if (e) return <ErrorBox msg={e} />;
  if (!m) return <Spinner text="Loading methodology" />;
  return (<>
    <h2>Model architecture &amp; methodology</h2>
    <p className="lead">Values are read from src/config.py, the checkpoints directory and saved metrics.</p>
    <div className="grid g2">
      <section className="panel"><h3>Input and preprocessing</h3><p>Input: one grayscale slice, tensor 1x1x{m.image_size[0]}x{m.image_size[1]}, classes {m.classes.join(" / ")}.</p><ol>{m.preprocessing.map(p => <li key={p}>{p}</li>)}</ol>
        <p className="muted small">Raw folder mapping: {Object.entries(m.raw_folder_to_class).map(([k, v]) => `${k} → ${v}`).join("; ")}.</p></section>
      <section className="panel"><h3>Configuration</h3><dl><dt>Dropout p</dt><dd>{m.dropout_p}</dd><dt>MC Dropout passes</dt><dd>{m.mc_samples}</dd><dt>Ensemble members</dt><dd>{m.ensemble_size}</dd><dt>ViT</dt><dd className="mono">{Object.entries(m.vit).map(([k, v]) => `${k}=${v}`).join(", ")}</dd><dt>Training</dt><dd className="mono">epochs={m.training.epochs}, batch={m.training.batch_size}, lr={m.training.lr}, seed={m.training.seed}</dd><dt>Split ratios</dt><dd className="mono">{Object.entries(m.training.split).map(([k, v]) => `${k}=${v}`).join(", ")}</dd>
        {Object.entries(m.parameters_m).map(([k, v]) => <><dt key={k}>Params ({k})</dt><dd>{v === null ? "n/a" : `${v} M`}</dd></>)}</dl></section></div>
    <section className="panel"><h3>Checkpoints</h3><div className="scroll"><table><thead><tr><th>Config</th><th>Model</th><th>Status</th><th>Files</th></tr></thead><tbody>{m.models.map(x => <tr key={x.id}><td>{x.code}</td><td>{x.architecture} + {x.uq_method}</td><td><span className={`chip ${x.ready ? "ok" : "bad"}`}>{x.ready ? "ready" : "missing"}</span></td><td className="mono">{x.checkpoints.join(", ")}</td></tr>)}</tbody></table></div></section>
    <section className="panel"><h3>Limitations</h3><ul>{m.notes.map(n => <li key={n}>{n}</li>)}<li>Inference is single-slice; no volume-level or subject-level aggregation.</li><li>Grad-CAM is available for CNN2D and DenseNet2D only.</li><li>Evaluation used a small test split, so reported metrics are imprecise.</li></ul></section>
  </>);
}
