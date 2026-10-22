"""FastAPI inference + results layer for AD_staging. Reuses src/ model, preprocessing and
uncertainty code; no ML logic is re-implemented except per-pass bookkeeping (see _mc/_de)."""
from __future__ import annotations

import io, json, math, re, shutil, tempfile, threading, time, uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src import config, preprocessing
from src.models.cnn2d import CNN2D
from src.models.densenet2d import DenseNet2D
from src.models.vit2d import ViT2D
from src.uncertainty import bedl as bedl_mod
from src.uncertainty.mc_dropout import enable_mc_dropout
from src.visualize_pipeline import grad_cam

ROOT = config.PROJECT_ROOT
CK = config.CHECKPOINT_DIR
MET = config.METRICS_DIR
PLOTS = config.PLOTS_DIR
CLASSES = config.CLASS_NAMES
K = len(CLASSES)
STORE = Path(tempfile.gettempdir()) / "ad_staging_predictions"
STORE.mkdir(exist_ok=True)
MAX_BYTES = 10 * 1024 * 1024
ALLOWED = {".png", ".jpg", ".jpeg"}
LOCK = threading.Lock()  # MC Dropout mutates module train/eval state of cached models

ARCHS = {  # id -> (class name, constructor, label used in legacy summary CSVs, row index)
    "cnn2d": ("CNN2D", lambda: CNN2D(num_classes=K, dropout_p=config.DROPOUT_P), "3D CNN", 1),
    "densenet2d": ("DenseNet2D", lambda: DenseNet2D(num_classes=K, dropout_p=config.DROPOUT_P), "DenseNet-121", 2),
    "vit2d": ("ViT2D", lambda: ViT2D(num_classes=K, dropout_p=config.DROPOUT_P), "ViT/Swin", 3),
}
UQS = {  # id -> (display, combination suffix, legacy CSV label, column index)
    "mc_dropout": ("MC Dropout", "MC_Dropout", "MC Dropout", 1),
    "deep_ensemble": ("Deep Ensemble", "Deep_Ensembles", "Deep Ensemble", 2),
    "bedl": ("Evidential DL", "BEDL", "Evidential DL", 3),
}


def model_id(a, u): return f"{a}_{u}"


def ckpt_paths(a, u):
    if u == "mc_dropout": return [CK / f"{a}_best.pt"]
    if u == "bedl": return [CK / f"{a}_bedl_best.pt"]
    return [CK / "ensemble" / a / f"member_{i}.pt" for i in range(config.ENSEMBLE_SIZE)]


def _csv(name):
    p = MET / name
    return pd.read_csv(p) if p.exists() else None


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def metrics_table():
    """One record per configuration from saved evaluation CSVs. Missing metrics stay null."""
    cc, rk = _csv("combination_comparison.csv"), _csv("ranking.csv")
    cw = _csv("classwise.csv")
    out = {}
    for a, (cname, _, alabel, ri) in ARCHS.items():
        for u, (udisp, usuf, ulabel, ci) in UQS.items():
            mid = model_id(a, u)
            paths = ckpt_paths(a, u)
            rec = dict(id=mid, code=f"C{ri}{ci}", architecture_id=a, architecture=cname, uq_id=u, uq_method=udisp,
                       ready=all(p.exists() for p in paths), checkpoints=[str(p.relative_to(ROOT)) for p in paths],
                       metrics={k: None for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                                                  "roc_auc_macro_ovr", "ece", "brier", "latency_s", "mean_entropy",
                                                  "mean_variance", "train_time_s"]},
                       rank=None, final_score=None, classwise=[],
                       plots={"confusion_matrix": f"confusion_matrix_{cname}{usuf}.png",
                              "reliability": f"reliability_diagram_{cname}{usuf}.png"})
            for k in ("confusion_matrix", "reliability"):
                if not (PLOTS / rec["plots"][k]).exists(): rec["plots"][k] = None
            if cc is not None:
                r = cc[cc.combination == f"{cname}+{usuf}"]
                if len(r):
                    r = r.iloc[0]; m = rec["metrics"]
                    for dst, src in [("accuracy", "accuracy"), ("precision_macro", "precision_macro"), ("recall_macro", "recall_macro"),
                                     ("f1_macro", "f1_macro"), ("roc_auc_macro_ovr", "roc_auc_macro_ovr"),
                                     ("ece", "expected_calibration_error"), ("brier", "brier_score"),
                                     ("latency_s", "inference_time_sec_per_sample"), ("mean_entropy", "mean_predictive_entropy"),
                                     ("mean_variance", "mean_predictive_variance"), ("train_time_s", "train_time_sec")]:
                        m[dst] = _num(r.get(src))
            if rk is not None:
                r = rk[(rk.Architecture == alabel) & (rk["UQ Method"] == ulabel)]
                if len(r): rec["rank"] = int(r.iloc[0]["Rank"]); rec["final_score"] = _num(r.iloc[0]["Final Score"])
            if cw is not None:
                r = cw[cw["Architecture + UQ"] == f"{alabel} + {ulabel}"]
                rec["classwise"] = [dict(cls=x["class"], precision=_num(x["precision"]), recall=_num(x["sensitivity_recall"]),
                                         specificity=_num(x["specificity"]), f1=_num(x["f1"]), auc=_num(x["auc"])) for _, x in r.iterrows()]
            out[mid] = rec
    return out


app = FastAPI(title="AD Staging API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    t = metrics_table()
    return {"status": "ok", "torch": torch.__version__, "device": "cpu", "models_ready": sum(m["ready"] for m in t.values()), "models_total": len(t)}


@app.get("/api/models")
def models():
    return [{k: v for k, v in m.items() if k in ("id", "code", "architecture_id", "architecture", "uq_id", "uq_method", "ready", "checkpoints")} for m in metrics_table().values()]


@app.get("/api/models/{mid}")
def model_detail(mid: str):
    t = metrics_table()
    if mid not in t: raise HTTPException(404, f"Unknown model '{mid}'")
    m = dict(t[mid])
    m["input"] = {"shape": [1, 1, *config.IMAGE_SIZE], "formats": sorted(ALLOWED), "preprocessing": "grayscale -> bilinear resize -> zero-mean/unit-variance"}
    m["capabilities"] = {"mc_samples": config.MC_DROPOUT_SAMPLES if m["uq_id"] == "mc_dropout" else None,
                         "ensemble_members": config.ENSEMBLE_SIZE if m["uq_id"] == "deep_ensemble" else None,
                         "gradcam": m["architecture_id"] != "vit2d"}
    return m


@app.get("/api/metrics")
def metrics():
    t = metrics_table()
    dd = _csv("dataset_distribution.csv")
    best = None
    rk = [m for m in t.values() if m["rank"] is not None]
    if rk: best = min(rk, key=lambda m: m["rank"])["id"]
    sel = None
    jp = MET / "TEAM4_complete_results.json"
    if jp.exists():
        j = json.loads(jp.read_text()); sel = j.get("selection_rule")
    return {"models": list(t.values()), "best_by_rank": best, "selection_rule": sel,
            "dataset_distribution": None if dd is None else json.loads(dd.to_json(orient="records")),
            "classes": CLASSES, "shared_plots": sorted(str(p.relative_to(PLOTS)).replace("\\", "/") for p in (PLOTS / "TEAM4").glob("*.png"))}


@app.get("/api/methodology")
def methodology():
    t = metrics_table()
    return {"classes": CLASSES, "raw_folder_to_class": config.RAW_FOLDER_TO_CLASS, "image_size": list(config.IMAGE_SIZE),
            "dropout_p": config.DROPOUT_P, "mc_samples": config.MC_DROPOUT_SAMPLES, "ensemble_size": config.ENSEMBLE_SIZE,
            "vit": dict(patch=config.VIT_PATCH_SIZE, embed_dim=config.VIT_EMBED_DIM, depth=config.VIT_DEPTH, heads=config.VIT_NUM_HEADS, mlp=config.VIT_MLP_DIM),
            "training": dict(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, seed=config.RANDOM_SEED, split=config.SPLIT_RATIOS),
            "parameters_m": {r["Architecture"]: _num(r["Parameters (M)"]) for _, r in (_csv("computational_efficiency.csv") if _csv("computational_efficiency.csv") is not None else pd.DataFrame()).drop_duplicates("Architecture").iterrows()} if _csv("computational_efficiency.csv") is not None else {},
            "models": [{k: m[k] for k in ("id", "code", "architecture", "uq_method", "ready", "checkpoints")} for m in t.values()],
            "preprocessing": ["Grayscale conversion (PIL 'L')", f"Bilinear resize to {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}", "Zero-mean / unit-variance normalization", "Quality check (shape and normalization statistics)"],
            "notes": ["Backbones are compact 2D slice-level stand-ins for the 3D architectures named in the project document (see src/models docstrings).",
                      "Skull stripping and registration are assumed pre-applied by the dataset publisher; they are not performed here.",
                      "Some legacy summary files label architectures '3D CNN', 'DenseNet-121', 'ViT/Swin'; the implemented models are CNN2D, DenseNet2D and ViT2D."]}


# ---- model cache ------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load(a: str, u: str):
    _, fn, _, _ = ARCHS[a]
    ms = []
    for p in ckpt_paths(a, u):
        m = fn(); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); ms.append(m)
    return ms


# ---- image helpers ---------------------------------------------------------------------------
def save_png(arr: np.ndarray, path: Path, scale: int = 1):
    im = Image.fromarray(arr.astype(np.uint8), mode="L" if arr.ndim == 2 else "RGB")
    if scale > 1: im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)


def to_u8(a):
    a = a.astype(np.float32); lo, hi = a.min(), a.max()
    return ((a - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def jet(v):  # v in [0,1] -> RGB uint8 (piecewise-linear jet)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def ent(p): return float(-(p * torch.log(p + 1e-12)).sum())


def vec(t): return [float(x) for x in t.tolist()]


def metric(key, label, value, rng, definition): return dict(key=key, label=label, value=value, range=rng, definition=definition)


# ---- uncertainty methods ---------------------------------------------------------------------
def _mc(model, x, T):
    enable_mc_dropout(model)  # repo helper: eval mode with Dropout layers active
    with torch.no_grad():
        L = torch.stack([model(x)[0] for _ in range(T)])  # (T,C)
    model.eval()
    P = F.softmax(L, 1); pm = P.mean(0)
    ee = float(np.mean([ent(p) for p in P])); pe = ent(pm)
    detail = dict(per_pass_probs=[vec(p) for p in P], per_pass_pred=[int(i) for i in P.argmax(1)], logits_mean=vec(L.mean(0)), logits_std=vec(L.std(0)),
                  prob_std=vec(P.std(0)), n_passes=T)
    ms = [metric("predictive_entropy", "Predictive entropy", pe, f"0 to ln({K}) = {math.log(K):.4f} nats", "H[mean_t p_t]: entropy of the mean predictive distribution (total uncertainty)."),
          metric("expected_entropy", "Expected entropy", ee, f"0 to ln({K})", "mean_t H[p_t]: average entropy of individual stochastic passes (aleatoric component)."),
          metric("mutual_information", "Mutual information", pe - ee, f"0 to ln({K})", "H[mean p] minus mean H[p_t]: disagreement between dropout masks (epistemic component)."),
          metric("probability_variance", "Probability variance", float(P.var(0).mean()), "0 to 0.25", "Mean over classes of the per-class variance of p_t across passes (same definition as repo mc_dropout_predict).")]
    return pm, L.mean(0), ms, detail, "per_pass"


def _de(members, x):
    with torch.no_grad():
        L = torch.stack([m(x)[0] for m in members])
    P = F.softmax(L, 1); pm = P.mean(0)
    ee = float(np.mean([ent(p) for p in P])); pe = ent(pm)
    agg = int(pm.argmax())
    detail = dict(per_member_probs=[vec(p) for p in P], per_member_pred=[int(i) for i in P.argmax(1)], per_member_logits=[vec(l) for l in L], n_members=len(members))
    ms = [metric("predictive_entropy", "Predictive entropy", pe, f"0 to ln({K}) = {math.log(K):.4f} nats", "H[mean_m p_m] of the ensemble-averaged distribution."),
          metric("expected_entropy", "Expected entropy", ee, f"0 to ln({K})", "Average entropy of the individual member predictions."),
          metric("mutual_information", "Mutual information (ensemble disagreement)", pe - ee, f"0 to ln({K})", "H[mean p] minus mean H[p_m]: how much members disagree."),
          metric("probability_variance", "Probability variance", float(P.var(0).mean()), "0 to 0.25", "Mean per-class variance across members (same as repo deep_ensemble_predict)."),
          metric("member_agreement", "Members agreeing with aggregate", float((P.argmax(1) == agg).float().mean()), "0 to 1", "Fraction of members whose argmax equals the aggregate prediction.")]
    return pm, L.mean(0), ms, detail, "per_member"


def _bedl(model, x):
    with torch.no_grad():
        logits = model(x)
    alpha = bedl_mod.logits_to_alpha(logits)
    r = bedl_mod.bedl_predict(model, x)
    S = float(alpha.sum())
    detail = dict(alpha=vec(alpha[0]), evidence=vec(alpha[0] - 1), strength=S)
    ms = [metric("vacuity", "Vacuity (epistemic)", float(r["variance"][0]), "K/S, in (0, 1]", "K / S where S = sum(alpha): mass assigned to 'unknown' (repo dirichlet_vacuity)."),
          metric("aleatoric", "Expected aleatoric (Dirichlet)", float(r["aleatoric"][0]), "0 to 0.25", "Repo dirichlet_aleatoric: sum_k p_k(1-p_k)/(S+1)."),
          metric("predictive_entropy", "Predictive entropy", float(r["entropy"][0]), f"0 to ln({K})", "Entropy of the Dirichlet expected probabilities alpha/S."),
          metric("dirichlet_strength", "Dirichlet strength S", S, f">= {K}", "Total concentration sum(alpha); larger means more evidence.")]
    return r["p_mean"][0], logits[0], ms, detail, "dirichlet"


# ---- predict -----------------------------------------------------------------------------------
def _cleanup():
    dirs = sorted([d for d in STORE.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    for d in dirs[:-50]: shutil.rmtree(d, ignore_errors=True)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    t_total = time.perf_counter()
    m = metrics_table().get(model)
    if m is None: raise HTTPException(404, f"Unknown model '{model}'")
    if not m["ready"]: raise HTTPException(409, f"Checkpoint(s) missing for '{model}': {m['checkpoints']}")
    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext in (".nii", ".gz", ".dcm"):
        raise HTTPException(415, "3D volumes (NIfTI/DICOM) are not supported: the trained models take a single 2D axial slice (PNG/JPEG).")
    if ext not in ALLOWED: raise HTTPException(415, f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(ALLOWED))}.")
    raw = await file.read()
    if len(raw) > MAX_BYTES: raise HTTPException(413, "File exceeds 10 MB.")
    try:
        img = Image.open(io.BytesIO(raw)); fmt, mode, (w, h) = img.format, img.mode, img.size; img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "File could not be decoded as an image.")
    if min(w, h) < 16: raise HTTPException(422, f"Image too small ({w}x{h}); minimum side is 16 px.")

    pid = uuid.uuid4().hex
    d = STORE / pid; d.mkdir()
    src_path = d / ("input" + ext); src_path.write_bytes(raw)
    url = lambda n: f"/api/predictions/{pid}/{n}"
    warnings, stages = [], []

    # stage 1-2: original + validation
    o = img.convert("RGB"); o.thumbnail((512, 512)); o.save(d / "s1_original.png")
    stages.append(dict(key="original", title="Original MRI", status="complete", images=[dict(label="Uploaded image", url=url("s1_original.png"))],
                       details={"file": name, "format": fmt, "mode": mode, "width": w, "height": h, "size_bytes": len(raw)}))
    raw_arr = preprocessing.load_slice_as_array(str(src_path))
    checks = [dict(name="Decodes as image", ok=True), dict(name="Accepted format (PNG/JPEG)", ok=True),
              dict(name="Minimum side >= 16 px", ok=True), dict(name="Non-constant intensity", ok=bool(raw_arr.std() > 0))]
    if raw_arr.std() == 0: raise HTTPException(422, "Image has constant intensity; normalization is undefined.")
    checks.append(dict(name=f"Compatible with {m['architecture']} (expects {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]} grayscale slice)", ok=True))
    stages.append(dict(key="validation", title="Input validation", status="complete", checks=checks, details={"expected_input": f"1x1x{config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}"}))
    if w != h: warnings.append(f"Image is not square ({w}x{h}); the repo's resize does not preserve aspect ratio, so anatomy will be stretched.")
    if mode not in ("L", "RGB", "RGBA"): warnings.append(f"Image mode '{mode}' was converted to grayscale.")

    # stage 3: preprocessing (repo functions, step by step)
    save_png(to_u8(raw_arr), d / "s3_gray.png")
    res = preprocessing.resize_array(raw_arr)
    save_png(to_u8(res), d / "s3_resized.png", 4)
    std = preprocessing.normalize_intensity(res)
    save_png(to_u8(std), d / "s3_normalized.png", 4)
    qc = preprocessing.quality_check(std)
    if not qc: warnings.append("Repo quality_check failed after normalization.")
    stages.append(dict(key="preprocessing", title="Preprocessing", status="complete", images=[
        dict(label=f"Grayscale ({w}x{h})", url=url("s3_gray.png")), dict(label=f"Resized {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]} (shown 4x)", url=url("s3_resized.png")),
        dict(label="Normalized (display min-max scaled)", url=url("s3_normalized.png"))],
        steps=[dict(name="Grayscale conversion", detail=f"raw mean {raw_arr.mean():.2f}, std {raw_arr.std():.2f}"),
               dict(name="Bilinear resize", detail=f"{w}x{h} -> {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}; mean {res.mean():.2f}, std {res.std():.2f}"),
               dict(name="Zero-mean / unit-variance", detail=f"mean {std.mean():.2e}, std {std.std():.4f}"), dict(name="Quality check", detail="passed" if qc else "failed")]))

    # stage 4: model-ready
    x = torch.from_numpy(std.astype(np.float32))[None, None]
    stages.append(dict(key="model_input", title="Model-ready input", status="complete", images=[dict(label="Tensor channel 0", url=url("s3_normalized.png"))],
                       details={"tensor_shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""), "model": f"{m['code']} {m['architecture']} + {m['uq_method']}", "checkpoints": m["checkpoints"]}))

    # stage 5-6: inference + UQ
    a, u = m["architecture_id"], m["uq_id"]
    t0 = time.perf_counter()
    with LOCK:
        mods = load(a, u)
        if u == "mc_dropout": pm, logits, ums, detail, kind = _mc(mods[0], x, config.MC_DROPOUT_SAMPLES)
        elif u == "deep_ensemble": pm, logits, ums, detail, kind = _de(mods, x)
        else: pm, logits, ums, detail, kind = _bedl(mods[0], x)
    infer_ms = (time.perf_counter() - t0) * 1000
    probs = vec(pm); pred = int(np.argmax(probs)); conf = probs[pred]
    transform = {"mc_dropout": "softmax per stochastic pass, then mean over passes", "deep_ensemble": "softmax per member, then mean over members",
                 "bedl": "softplus evidence + 1 = Dirichlet alpha; p = alpha / sum(alpha)"}[u]
    stages.append(dict(key="inference", title="Model inference", status="complete",
                       details={"output_logits": vec(logits), "logit_note": "mean of per-pass logits" if u == "mc_dropout" else "mean of member logits" if u == "deep_ensemble" else "raw logits", "probability_transform": transform,
                                "class_order": CLASSES}, probabilities=[dict(cls=c, prob=p) for c, p in zip(CLASSES, probs)]))
    stages.append(dict(key="uncertainty", title="Uncertainty quantification", status="complete", method=m["uq_method"], metrics=ums, detail_kind=kind, detail=detail))

    # explainability (repo grad_cam; needs `features` layer -> CNN2D / DenseNet2D only)
    expl = dict(available=False, reason=None, method=None, image=None)
    gm = mods[0]
    if not hasattr(gm, "features"):
        expl["reason"] = f"{m['architecture']} has no convolutional `features` layer; the repository's Grad-CAM does not apply to the Vision Transformer, and no other attribution method is implemented."
    else:
        try:
            cam, tc, _ = grad_cam(gm, x.clone(), target_class=pred)
            base = to_u8(std); rgb = np.stack([base] * 3, -1).astype(np.float32)
            ov = (0.55 * rgb + 0.45 * jet(cam)).astype(np.uint8)
            save_png(ov, d / "s8_gradcam.png", 4)
            expl.update(available=True, method="Grad-CAM (src/visualize_pipeline.grad_cam, layer `features`)", image=url("s8_gradcam.png"), target_class=CLASSES[tc],
                        note="Computed on member_0 only." if u == "deep_ensemble" else "Computed on the evaluated checkpoint.")
        except Exception as e:  # noqa
            expl["reason"] = f"Grad-CAM failed: {e}"
    stages.append(dict(key="explainability", title="Explainability", status="complete" if expl["available"] else "unavailable", explanation=expl))

    # warnings from real evaluation data only
    mt = m["metrics"]
    if mt["recall_macro"] is not None and mt["recall_macro"] <= 1 / K + 1e-3:
        warnings.append(f"Saved test evaluation for this configuration shows macro recall {mt['recall_macro']:.3f} (= chance for {K} classes): it predicts a single class on the evaluation set, so its outputs are not informative.")
    if mt["accuracy"] is not None: warnings.append(f"Held-out test accuracy of this configuration was {mt['accuracy']*100:.1f}% on a small test split; softmax confidence is not calibrated accuracy (test ECE {mt['ece']:.3f}).") if mt["ece"] is not None else None
    warnings.append("Research prototype: not validated for clinical use. Low uncertainty does not guarantee a correct prediction.")

    total_ms = (time.perf_counter() - t_total) * 1000
    result = dict(id=pid, model=dict(id=m["id"], code=m["code"], architecture=m["architecture"], uq_method=m["uq_method"]),
                  input=dict(filename=name, format=fmt, mode=mode, width=w, height=h, size_bytes=len(raw)), stages=stages,
                  prediction=dict(class_index=pred, class_name=CLASSES[pred], confidence=conf),
                  probabilities=[dict(cls=c, prob=p) for c, p in zip(CLASSES, probs)], uncertainty_method=m["uq_method"],
                  uncertainty_metrics=ums, tensor_shape=list(x.shape), duration_ms=dict(inference=infer_ms, total=total_ms),
                  warnings=warnings, explanation=expl)
    (d / "result.json").write_text(json.dumps(result))
    _cleanup()
    return result


@app.get("/api/predictions/{pid}")
def get_prediction(pid: str):
    if not re.fullmatch(r"[0-9a-f]{32}", pid): raise HTTPException(400, "Bad id")
    p = STORE / pid / "result.json"
    if not p.exists(): raise HTTPException(404, "Prediction expired or not found")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/predictions/{pid}/{name}")
def get_artifact(pid: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{32}", pid) or not re.fullmatch(r"s\d_[a-z]+\.png", name): raise HTTPException(400, "Bad path")
    p = STORE / pid / name
    if not p.exists(): raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/png")


if PLOTS.exists(): app.mount("/api/plots", StaticFiles(directory=PLOTS), name="plots")
DIST = ROOT / "frontend" / "dist"
if DIST.exists(): app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
