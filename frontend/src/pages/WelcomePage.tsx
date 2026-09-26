import { Activity, ArrowRight, BarChart3, Check, Database, History, SlidersHorizontal, Workflow } from "lucide-react";
import { Link } from "react-router-dom";
import { useWorkflow } from "../workflow";

const capabilities = [
  { icon: Database, title: "Understand and prepare your data", text: "Inspect a CSV or example dataset, handle missing values and outliers, and define numeric, nominal or ordinal features." },
  { icon: Workflow, title: "Choose a fair evaluation", text: "Use cross-validation or a train / validation / test split. Keep chronological order for time-series experiments." },
  { icon: SlidersHorizontal, title: "Search model settings", text: "Compare particle swarm, random and grid search. Set the model, metric and compute budget, or compare several models together." },
  { icon: BarChart3, title: "Keep results you can inspect", text: "Watch progress, compare scores, explore predictions and download saved models, settings and reproducible data partitions." },
];

export function WelcomePage() {
  const { run } = useWorkflow();
  return <div className="welcomePage">
    <section className="welcomeHero">
      <div><span className="welcomeEyebrow">PSPSO · DEVELOPMENT BUILD · LOCAL WORKSPACE</span><h1>Better model settings.<br/><em>Experiments you can trust.</em></h1><p>PSPSO helps you prepare tabular data, tune machine-learning models and understand their results. This dashboard and CLI are still under active development while version 1.0 is prepared for release.</p><div className="welcomeActions"><Link className="primary" to="/data">Set up an experiment <ArrowRight size={17}/></Link><Link className="secondary" to="/history"><History size={17}/> View history</Link></div><span className="welcomeLocal"><Check size={15}/> Single-user workspace · Runs on your computer · Dashboard and CLI share the same data</span></div>
      <div className="welcomeJourney"><span>YOUR EXPERIMENT, END TO END</span>{["Data setup", "Model & parameters", "Search engine", "Live experiments", "Results", "History"].map((label, index) => <div key={label}><b>{String(index + 1).padStart(2, "0")}</b><strong>{label}</strong>{index === 3 ? <Activity size={17}/> : <Check size={16}/>}</div>)}</div>
    </section>
    {run && <Link className="welcomeResume" to={run.status === "completed" ? `/results/${run.run_id}` : `/live/${run.run_id}`}><Activity size={19}/><span><strong>Return to your experiment</strong><small>{run.status} · {run.run_id.slice(0, 8)}</small></span><ArrowRight size={18}/></Link>}
    <div className="welcomeSectionHeading"><span>WHAT YOU CAN DO</span><h2>Prepare, compare, understand.</h2><p>For regression, binary classification and multiclass classification.</p></div>
    <div className="welcomeCapabilities">{capabilities.map(({ icon: Icon, title, text }) => <section key={title}><div className="welcomeIcon"><Icon size={23}/></div><h3>{title}</h3><p>{text}</p></section>)}</div>
    <section className="welcomeStart"><div><h2>Start with a small experiment</h2><p>Choose an included dataset to explore the workflow, then bring your own CSV. Each stage explains the choices and checks your setup before you run.</p></div><Link to="/data" className="secondary">Explore Data setup <ArrowRight size={17}/></Link></section>
    <footer className="welcomeFooter"><span>PSPSO uses particle swarm optimization, with random and grid search for comparison.</span><span className="welcomeLinks"><a href="https://ayhaidar.github.io/pspso/" target="_blank" rel="noreferrer">Documentation ↗</a><a href="/api/v1/docs" target="_blank" rel="noreferrer">API reference ↗</a></span></footer>
  </div>;
}
