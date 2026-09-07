# Architecture

FastAPI's lifespan owns the sole run manager for its workspace. Creating an app
or reading a repository does not claim that ownership. Workers run outside the
HTTP process; the CLI routes managed start, cancel and retry operations through
the same `/api/v1` service used by React.

```mermaid
flowchart LR
    UI[React dashboard] --> API[FastAPI service]
    CLI[CLI start / retry / cancel] --> API
    API --> M[One leased run manager]
    READ[CLI list / show] --> DB[(SQLite repository)]
    M <--> DB
    M --> W[Supervised worker process]
    W --> OPT[Optimizer and per-fold pipelines]
    OPT --> DB
    W --> ART[Saved dataset, model and artifacts]
    API --> ART
    API -->|Snapshots and replayable SSE| UI
```

## Ownership, recovery and cancellation

An atomic SQLite transaction claims a service lease. The repository persists
service identity, PID, heartbeat, queued attempts, worker identity, termination
intent and parent run links. A second live service cannot claim the workspace.
The monitor renews its lease and stops if it loses ownership. One active run is
the default; `PSPSO_MAX_WORKERS` controls run-level concurrency.

```mermaid
stateDiagram-v2
    [*] --> queued: persist run and attempt
    queued --> running: service starts a supervised worker
    queued --> cancelled: cancel before dispatch
    running --> completed: artifacts saved and success recorded
    running --> failed: execution error or timeout
    running --> cancelled: cancellation and cleanup
    running --> interrupted: service shutdown or stale ownership
    interrupted --> [*]
    failed --> [*]
    cancelled --> [*]
    completed --> [*]
```

Shutdown interrupts active attempts and leaves queued attempts for the next
service. Recovery reconciles completed records and marks stale active attempts
interrupted. Retry creates a new run and retains the original record.
Cancellation first persists intent; cooperative checks stop work between fits
and PyTorch epochs. After a bounded grace period, the supervisor terminates the
full process tree and waits for cleanup. Windows Job Objects also terminate
workers if their service process dies. Terminal updates are idempotent and late
progress events cannot resurrect a finished run.

## Candidate concurrency

```mermaid
flowchart TD
    I[PSO iteration: freeze particle positions] --> Q[Bounded candidate queue]
    Q --> A[Worker slot 1: sequential folds]
    Q --> B[Worker slot 2: sequential folds]
    A --> G[Iteration barrier: gather candidate scores]
    B --> G
    G --> U[Update bests, velocities and positions]
    U --> I
    G --> R[Final winner refit]
```

`runtime.trial_workers` bounds candidate concurrency independently of run-level
concurrency. Grid and random search use bounded batches. Model-level `n_jobs` is
one when candidates run concurrently. Fit events identify candidate, fold,
iteration, actual worker slot and unique fit ID. Planned fits include final refit;
early stopping and failures can leave unused budget.

## Event delivery

```mermaid
sequenceDiagram
    participant W as Worker
    participant D as SQLite
    participant A as FastAPI
    participant B as Browser
    W->>D: Persist ordered event
    B->>A: SSE after last known sequence
    A->>D: Read events after cursor
    A->>B: id, event type, JSON data
    A-->>B: Heartbeat while idle
    Note over A,B: Connection lost
    B->>A: Reconnect with Last-Event-ID
    A->>B: Replay missing events
    Note over B: Deduplicate by sequence number
    B->>A: Periodic snapshot + incremental history
```

The browser keeps its selected run ID in the route, so reloads reopen the same
run. Polling reconciles snapshots even while SSE is connected. Events are
persisted before delivery; worker output goes directly to a file and is also
tailed into structured log events.

## Artifact creation

```mermaid
flowchart LR
    S[Frozen specification and dataset] --> E[Fold evaluation]
    E --> R[Selected pipeline and final refit]
    R --> M[Model and preprocessor]
    R --> D[Saved diagnostics]
    E --> I[Partition and fold indices]
    S --> P[Environment, dependency versions, source hash and seeds]
    M --> A[Artifact manifest]
    D --> A
    I --> A
    P --> A
    A --> F[Persist final result then terminal event]
```

Serialization failures emit visible warnings. Prediction and explanation
endpoints require saved inputs and models; they never fit replacements while
reading a result. Numbered, transactional migrations extend existing v1
workspaces without touching pre-1.0 data.
