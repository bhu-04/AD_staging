export interface Metrics { accuracy: number | null; precision_macro: number | null; recall_macro: number | null; f1_macro: number | null; f1_weighted: number | null; roc_auc_macro_ovr: number | null; ece: number | null; brier: number | null; latency_s: number | null; mean_entropy: number | null; mean_variance: number | null; train_time_s: number | null }
export type MetricKey = keyof Metrics;
export interface ClassRow { cls: string; precision: number | null; recall: number | null; specificity: number | null; f1: number | null; auc: number | null }
export interface ModelRec { id: string; code: string; architecture_id: string; architecture: string; uq_id: string; uq_method: string; ready: boolean; checkpoints: string[]; metrics: Metrics; rank: number | null; final_score: number | null; classwise: ClassRow[]; plots: { confusion_matrix: string | null; reliability: string | null } }
export interface MetricsPayload { models: ModelRec[]; best_by_rank: string | null; selection_rule: string | null; dataset_distribution: Record<string, number | string>[] | null; classes: string[]; shared_plots: string[] }
export interface Methodology { classes: string[]; raw_folder_to_class: Record<string, string>; image_size: number[]; dropout_p: number; mc_samples: number; ensemble_size: number; vit: Record<string, number>; training: { epochs: number; batch_size: number; lr: number; seed: number; split: Record<string, number> }; parameters_m: Record<string, number | null>; models: Pick<ModelRec, "id" | "code" | "architecture" | "uq_method" | "ready" | "checkpoints">[]; preprocessing: string[]; notes: string[] }
export interface UMetric { key: string; label: string; value: number; range: string; definition: string }
export interface Stage { key: string; title: string; status: string; images?: { label: string; url: string }[]; details?: Record<string, unknown>; checks?: { name: string; ok: boolean }[]; steps?: { name: string; detail: string }[]; probabilities?: { cls: string; prob: number }[]; metrics?: UMetric[]; method?: string; detail_kind?: "per_pass" | "per_member" | "dirichlet"; detail?: Record<string, any>; explanation?: Explanation }
export interface Explanation { available: boolean; reason: string | null; method: string | null; image: string | null; target_class?: string; note?: string }
export interface Prediction { id: string; model: { id: string; code: string; architecture: string; uq_method: string }; input: { filename: string; format: string; mode: string; width: number; height: number; size_bytes: number }; stages: Stage[]; prediction: { class_index: number; class_name: string; confidence: number }; probabilities: { cls: string; prob: number }[]; uncertainty_method: string; uncertainty_metrics: UMetric[]; tensor_shape: number[]; duration_ms: { inference: number; total: number }; warnings: string[]; explanation: Explanation }

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try { r = await fetch(url, init); } catch { throw new Error("Cannot reach the backend at /api. Start it with: uvicorn api.main:app --port 8000"); }
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail ?? m; } catch { /* ignore */ } throw new Error(typeof m === "string" ? m : JSON.stringify(m)); }
  return r.json();
}
export const getMetrics = () => j<MetricsPayload>("/api/metrics");
export const getMethodology = () => j<Methodology>("/api/methodology");
export const getHealth = () => j<{ status: string; torch: string; models_ready: number; models_total: number }>("/api/health");
export function predict(file: File, model: string) { const f = new FormData(); f.append("file", file); f.append("model", model); return j<Prediction>("/api/predict", { method: "POST", body: f }); }
export const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "n/a" : v.toFixed(d));
export const pct = (v: number | null | undefined) => (v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(1)}%`);
export const METRIC_LABELS: Record<string, { label: string; better: "high" | "low"; pct?: boolean }> = {
  accuracy: { label: "Accuracy", better: "high", pct: true }, precision_macro: { label: "Precision (macro)", better: "high" }, recall_macro: { label: "Recall (macro)", better: "high" },
  f1_macro: { label: "Macro-F1", better: "high" }, f1_weighted: { label: "Weighted F1", better: "high" }, roc_auc_macro_ovr: { label: "ROC-AUC (macro OvR)", better: "high" },
  ece: { label: "ECE", better: "low" }, brier: { label: "Brier score", better: "low" }, latency_s: { label: "Latency (s/sample)", better: "low" } };
