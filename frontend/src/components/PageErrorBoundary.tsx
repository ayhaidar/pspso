import { Component, type ReactNode } from "react";

export class PageErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <main className="page"><section className="surface" role="alert"><h1>This page could not be loaded</h1><p>The dashboard may have been updated while this tab was open. Reload to get the current version.</p><button className="primary" onClick={() => window.location.reload()}>Reload dashboard</button></section></main>;
    return this.props.children;
  }
}
