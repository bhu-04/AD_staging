import { DragEvent, ReactNode, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Prediction, Stage, predict } from "../api";
import { ProbBars } from "../charts";
import { ErrorBox, Gate, Spinner, useApp } from "../ui";
import { UncertaintyView } from "./Uncertainty";

const OK = ["image/png", "image/jpeg"], MAX = 10 * 1024 * 1024;
const Dl = ({ o }: { o: Record<string, unknown> }) => <dl>{Object.entries(o).map(([k, v]) => <><dt key={k}>{k.replace(/_/g, " ")}</dt><dd className="mono">{Array.isArray(v) ? v.map(x => typeof x === "number" ? +x.toFixed(4) : String(x)).join(", ") : String(v)}</dd></>)}</dl>;
const Imgs = ({ s }: { s: Stage }) => <div className="imgs">{s.images?.map(i => <figure key={i.url + i.label}><img src={i.url} alt={i.label} /><figcaption>{i.label}</figcaption></figure>)}</div>;

function Body({ s, r }: { s: Stage; r: Prediction }): ReactNode {
  switch (s.key) {
    case "original": return <><Imgs s={s} /><Dl o={s.details!} /><p className="muted small">The unmodified upload (downscaled for display only if larger than 512 px).</p></>;
    case "validation": return <><ul>{s.checks!.map(c => <li key={c.name}><span className={`chip ${c.ok ? "ok" : "bad"}`}>{c.ok ? "pass" : "fail"}</span> {c.name}</li>)}</ul><Dl o={s.details!} /></>;
    case "preprocessing": return <><Imgs s={s} /><ol>{s.steps!.map(t => <li key={t.name}><b>{t.name}</b>: <span className="mono">{t.detail}</span></li>)}</ol><p className="muted small">Executed with the repository's src/preprocessing.py functions. Display images are contrast-scaled only for viewing.</p></>;
    case "model_input": return <><Imgs s={s} /><Dl o={s.details!} /></>;
    case "inference": return <><Dl o={s.details!} /><ProbBars items={s.probabilities!} /></>;
    case "uncertainty": return <UncertaintyView r={r} />;
    case "final": return <><h3>{r.prediction.class_name}</h3><ProbBars items={r.probabilities} /><Dl o={{ confidence: r.prediction.confidence.toFixed(4), uncertainty_method: r.uncertainty_method, inference_ms: r.duration_ms.inference.toFixed(1), total_ms: r.duration_ms.total.toFixed(1), model: `${r.model.code} ${r.model.architecture}` }} />
      {r.warnings.map(w => <div key={w} className="alert warn">{w}</div>)}<p className="muted small">Confidence is the largest model output probability. It is not accuracy and is not guaranteed to be calibrated.</p></>;
    case "explainability": { const e = s.explanation!; return e.available ? <><div className="imgs"><figure><img src={e.image!} alt="Grad-CAM overlay" style={{ imageRendering: "auto" }} /><figcaption>Grad-CAM for class {e.target_class}</figcaption></figure></div><p>{e.method}. {e.note}</p><p className="muted small">Highlights regions that increased this class's score in the network; it is not evidence of pathology.</p></> : <p>Attribution unavailable. {e.reason}</p>; }
  }
  return null;
}

export default function Inference() {
  const { result, setResult, modelId, setModelId } = useApp();
  const [file, setFile] = useState<File | null>(null), [prev, setPrev] = useState<string>(""), [err, setErr] = useState<string | null>(null), [busy, setBusy] = useState(false), [over, setOver] = useState(false);
  const inp = useRef<HTMLInputElement>(null);
  useEffect(() => { if (!file) { setPrev(""); return; } const u = URL.createObjectURL(file); setPrev(u); return () => URL.revokeObjectURL(u); }, [file]);
  const pick = (f?: File | null) => {
    setErr(null); if (!f) return;
    if (!OK.includes(f.type) && !/\.(png|jpe?g)$/i.test(f.name)) return setErr("Unsupported file. The trained models take a single 2D axial slice as PNG or JPEG. NIfTI/DICOM volumes are not supported.");
    if (f.size > MAX) return setErr("File exceeds 10 MB.");
    setFile(f); setResult(null);
  };
  const run = async () => { if (!file || !modelId) return; setBusy(true); setErr(null); try { setResult(await predict(file, modelId)); } catch (e: any) { setErr(e.message); } finally { setBusy(false); } };
  const reset = () => { setFile(null); setResult(null); setErr(null); if (inp.current) inp.current.value = ""; };
  const exportJson = () => { const b = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }), a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = `prediction_${result!.id}.json`; a.click(); URL.revokeObjectURL(a.href); };
  return (<Gate>{M => {
    const stages: Stage[] = result ? (() => { const s = [...result.stages]; const i = s.findIndex(x => x.key === "explainability"); s.splice(i < 0 ? s.length : i, 0, { key: "final", title: "Final prediction", status: "complete" }); return s; })() : [];
    return (<>
      <h2>MRI image inference workstation</h2>
      <p className="lead">Upload a single 2D axial MRI slice (PNG or JPEG). No label is needed. Inference runs on the backend with the repository's trained checkpoints.</p>
      {err && <ErrorBox msg={err} />}
      <section className="panel noprint"><div className="row"><div><label htmlFor="mdl">Model configuration</label><select id="mdl" value={modelId} onChange={e => { setModelId(e.target.value); setResult(null); }}>{M.models.map(m => <option key={m.id} value={m.id} disabled={!m.ready}>{m.code} {m.architecture} + {m.uq_method}{m.ready ? "" : " (checkpoint missing)"}</option>)}</select></div>
        <button className="btn" onClick={run} disabled={!file || busy || !modelId}>Run inference</button><button className="btn sec" onClick={reset} disabled={busy}>Reset</button></div>
        <div className={`drop ${over ? "over" : ""}`} style={{ marginTop: 16 }} tabIndex={0} role="button" aria-label="Choose or drop an MRI image" onClick={() => inp.current?.click()} onKeyDown={e => (e.key === "Enter" || e.key === " ") && inp.current?.click()}
          onDragOver={(e: DragEvent) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={(e: DragEvent) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}>
          Drag and drop a PNG or JPEG here, or press to browse<input ref={inp} type="file" accept=".png,.jpg,.jpeg,image/png,image/jpeg" hidden onChange={e => pick(e.target.files?.[0])} /></div>
        {file && <div className="imgs" style={{ marginTop: 16 }}><figure><img src={prev} alt="Selected upload preview" /></figure><dl><dt>name</dt><dd>{file.name}</dd><dt>type</dt><dd>{file.type || "unknown"}</dd><dt>size</dt><dd>{(file.size / 1024).toFixed(1)} KB</dd></dl></div>}
        {busy && <Spinner text="Running preprocessing, inference and uncertainty estimation" />}</section>
      {result && <><div className="row noprint" style={{ marginBottom: 8 }}><button className="btn sec" onClick={exportJson}>Export JSON</button><button className="btn sec" onClick={() => window.print()}>Print report</button><Link className="btn sec" to="/uncertainty">Open uncertainty analysis</Link></div>
        {stages.map((s, i) => <details className="stage" key={s.key} open={s.key === "final"}><summary><span className="num-badge">{i + 1}</span>{s.title}<span className={`chip ${s.status === "complete" ? "ok" : "na"}`}>{s.status}</span></summary><div className="stage-body">{Body({ s, r: result })}</div></details>)}</>}
    </>);
  }}</Gate>);
}
