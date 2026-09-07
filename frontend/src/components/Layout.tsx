import { Activity, BarChart3, BrainCircuit, CircleHelp, Database, History, House, Settings2 } from "lucide-react";
import { useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { guideSeenKey, StageGuide } from "./Help";
import { useWorkflow } from "../workflow";

const steps = [
  { path: "/data", label: "Data setup", icon: Database },
  { path: "/model", label: "Model & parameters", icon: BrainCircuit },
  { path: "/search", label: "Search engine", icon: Settings2 },
  { path: "/live", label: "Live experiments", icon: Activity },
  { path: "/results", label: "Results", icon: BarChart3 },
  { path: "/history", label: "History", icon: History }
];

export function AppLayout() {
  const location = useLocation();
  const { draft, run, message, metadata } = useWorkflow();
  const activeIndex = Math.max(0, steps.findIndex((step) => location.pathname.startsWith(step.path)));
  return (
    <div className="shell">
      <header className="appHeader">
        <Link className="brand" to="/" aria-label="PSPSO overview">
          <img className="brandLogo" src="/brand/pspso-logo-inverse.svg" alt="PSPSO" />
        </Link>
        <div className="runContext">
          <span>{metadata?.tasks[draft.task]?.label ?? draft.task}</span><strong>{run ? `${run.status} · ${run.run_id.slice(0, 8)}` : "Draft not started"}</strong>
        </div>
      </header>
      <nav className="workflowNav" aria-label="Experiment workflow">
        <NavLink to="/" end className={({ isActive }) => `workflowStep ${isActive ? "active" : ""}`}>
          <House size={17} /><span>Overview</span>
        </NavLink>
        {steps.map((step, index) => {
          const Icon = step.icon;
          return <NavLink key={step.path} to={run && ["/live", "/results"].includes(step.path) ? `${step.path}/${run.run_id}` : step.path} className={({ isActive }) => `workflowStep ${isActive ? "active" : ""} ${index < activeIndex ? "visited" : ""}`}>
            <span className="stepNumber">{index + 1}</span><Icon size={17} /><span>{step.label}</span>
          </NavLink>;
        })}
      </nav>
      {message && <div role="alert" className="globalMessage">{message}</div>}
      <main className="page"><Outlet /></main>
    </div>
  );
}

export function PageHeading({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description: string; actions?: React.ReactNode }) {
  const stage = Number(eyebrow.match(/\d+/)?.[0] ?? 0);
  const [guideOpen, setGuideOpen] = useState(() => stage > 0 && localStorage.getItem(guideSeenKey(stage)) !== "true");
  function closeGuide() { localStorage.setItem(guideSeenKey(stage), "true"); setGuideOpen(false); }
  return <><header className="pageHeading"><div><span>{eyebrow}</span><h1>{title}</h1><p>{description}</p></div><div className="pageActions">{actions}<button type="button" className="secondary" onClick={() => setGuideOpen(!guideOpen)} aria-expanded={guideOpen}><CircleHelp size={16}/> {guideOpen ? "Hide guide" : "Stage guide"}</button></div></header>{guideOpen && <StageGuide stage={stage} onClose={closeGuide}/>}</>;
}

export function ValidationSummary({ stage }: { stage: "data" | "model" | "search" | "full" }) {
  const { validations } = useWorkflow();
  const validation = validations[stage];
  if (!validation) return null;
  if (validation.valid) return <div className="validation success">Checks passed. This stage is ready.</div>;
  return <div className="validation error"><strong>Please resolve these checks</strong>{Object.entries(validation.errors).flatMap(([section, errors]) => errors.map((error) => <p key={`${section}-${error}`}><b>{section}:</b> {error}</p>))}</div>;
}

export function StatusBadge({ status }: { status: string }) { return <span className={`statusBadge ${status}`}>{status}</span>; }

export function Metric({ label, value }: { label: string; value: React.ReactNode }) { return <div className="metric"><span>{label}</span><strong>{value}</strong></div>; }
