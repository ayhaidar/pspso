import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/Layout";
import { PageErrorBoundary } from "./components/PageErrorBoundary";
import { WorkflowProvider } from "./workflow";
import "./styles.css";

const DataPage = lazy(() => import("./pages/DataPage").then((module) => ({ default: module.DataPage })));
const WelcomePage = lazy(() => import("./pages/WelcomePage").then((module) => ({ default: module.WelcomePage })));
const ModelPage = lazy(() => import("./pages/ModelPage").then((module) => ({ default: module.ModelPage })));
const SearchPage = lazy(() => import("./pages/SearchPage").then((module) => ({ default: module.SearchPage })));
const LivePage = lazy(() => import("./pages/LivePage").then((module) => ({ default: module.LivePage })));
const ResultsPage = lazy(() => import("./pages/ResultsPage").then((module) => ({ default: module.ResultsPage })));
const HistoryPage = lazy(() => import("./pages/HistoryPage").then((module) => ({ default: module.HistoryPage })));

function App() {
  return <BrowserRouter><WorkflowProvider><Suspense fallback={<div className="routeLoading">Loading workspace…</div>}><Routes><Route element={<AppLayout/>}>
    <Route index element={<WelcomePage/>}/>
    <Route path="data" element={<DataPage/>}/>
    <Route path="model" element={<ModelPage/>}/>
    <Route path="search" element={<SearchPage/>}/>
    <Route path="live" element={<LivePage/>}/>
    <Route path="live/:runId" element={<LivePage/>}/>
    <Route path="results" element={<ResultsPage/>}/>
    <Route path="results/:runId" element={<ResultsPage/>}/>
    <Route path="history" element={<HistoryPage/>}/>
    <Route path="*" element={<Navigate to="/" replace/>}/>
  </Route></Routes></Suspense></WorkflowProvider></BrowserRouter>;
}

createRoot(document.getElementById("root")!).render(<StrictMode><PageErrorBoundary><App/></PageErrorBoundary></StrictMode>);
