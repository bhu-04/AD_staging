import { Component, ReactNode, createContext, useContext, useEffect, useState } from "react";
import { MetricsPayload, Prediction, getMetrics } from "./api";

export const Stat = ({ k, v, s }: { k: string; v: string; s?: string }) => <div className="panel stat"><div className="k">{k}</div><div className="v">{v}</div>{s && <div className="s">{s}</div>}</div>;
export const Spinner = ({ text }: { text: string }) => <p role="status"><span className="spinner" /> {text}</p>;
export const ErrorBox = ({ msg }: { msg: string }) => <div className="alert err" role="alert">{msg}</div>;
export class Boundary extends Component<{ children: ReactNode }, { err: string | null }> {
  state = { err: null as string | null };
  static getDerivedStateFromError(e: Error) { return { err: e.message }; }
  render() { return this.state.err ? <ErrorBox msg={`This view crashed: ${this.state.err}`} /> : this.props.children; }
}
interface Ctx { metrics: MetricsPayload | null; error: string | null; result: Prediction | null; setResult: (p: Prediction | null) => void; modelId: string; setModelId: (s: string) => void }
const C = createContext<Ctx>(null as unknown as Ctx);
export const useApp = () => useContext(C);
export function Provider({ children }: { children: ReactNode }) {
  const [metrics, setM] = useState<MetricsPayload | null>(null), [error, setE] = useState<string | null>(null), [result, setResult] = useState<Prediction | null>(null), [modelId, setModelId] = useState("");
  useEffect(() => { getMetrics().then(m => { setM(m); setModelId(m.models.find(x => x.ready && x.id === m.best_by_rank)?.id ?? m.models.find(x => x.ready)?.id ?? ""); }).catch(e => setE(e.message)); }, []);
  return <C.Provider value={{ metrics, error, result, setResult, modelId, setModelId }}>{children}</C.Provider>;
}
export function Gate({ children }: { children: (m: MetricsPayload) => ReactNode }) {
  const { metrics, error } = useApp();
  if (error) return <ErrorBox msg={error} />;
  if (!metrics) return <Spinner text="Loading evaluation results" />;
  return <>{children(metrics)}</>;
}
