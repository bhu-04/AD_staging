import React from "react";
import { createRoot } from "react-dom/client";
import { HashRouter, NavLink, Route, Routes } from "react-router-dom";
import "./styles.css";
import { Boundary, Provider } from "./ui";
import Overview from "./pages/Overview";
import Performance from "./pages/Performance";
import Inference from "./pages/Inference";
import Uncertainty from "./pages/Uncertainty";
import Methodology from "./pages/Methodology";

const nav: [string, string][] = [["/", "Research overview"], ["/performance", "Model performance"], ["/inference", "MRI inference"], ["/uncertainty", "Uncertainty analysis"], ["/methodology", "Architecture & methodology"]];
function App() {
  return (<HashRouter><Provider><div className="shell">
    <nav className="side" aria-label="Primary"><h1>Alzheimer's Disease Staging<span>Model Evaluation &amp; Inference</span></h1>
      {nav.map(([to, l]) => <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>{l}</NavLink>)}
      <div className="foot">Research prototype. Not for clinical use.</div></nav>
    <main><Boundary><Routes><Route path="/" element={<Overview />} /><Route path="/performance" element={<Performance />} /><Route path="/inference" element={<Inference />} /><Route path="/uncertainty" element={<Uncertainty />} /><Route path="/methodology" element={<Methodology />} /></Routes></Boundary></main>
  </div></Provider></HashRouter>);
}
createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
