# Adaptive Multi-Agent LLM System for Task Decomposition and Collaborative Problem Solving

**Final Year Computer Science & Engineering Major Project**  
*Current Milestone: Stages 1–5 — Orchestrated Multi-Agent Pipeline with RAG*

---

## 1. Project Overview

The objective of this engineering major project is to build an adaptive, orchestrator-driven multi-agent LLM system capable of taking high-level, complex analytical tasks, decomposing them into discrete subtasks, routing them to specialized agents, evaluating intermediate outputs, and dynamically adapting execution plans when results fall short of quality benchmarks.

### Current Implementation: Stages 1–5

The system implements a dynamic, dependency-aware orchestrated pipeline:

$$\text{User Task} \longrightarrow \text{Planner (TaskGraph)} \longrightarrow \text{Orchestrator (Parallel + Capabilities + RAG)} \longrightarrow \text{Evaluator} \longrightarrow \text{Synthesizer} \longrightarrow \text{Final Answer}$$

---

## 2. Architecture

```
                                  +---------------------------------------+
                                  |                 USER                  |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   REACT FRONTEND (TanStack + Vite)    |
                                  |   Execution Graph · Control Plane     |
                                  +---------------------------------------+
                                                      |  HTTP /api/run, /api/status
                                                      v
                                  +---------------------------------------+
                                  |       FASTAPI BACKEND & ROUTER        |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     ORCHESTRATED PIPELINE (Stage 2-5) |
                                  +---------------------------------------+
                                                      |
                          +---------------------------+---------------------------+
                          |                           |                           |
                          v                           v                           v
                  +----------------+        +------------------+        +------------------+
                  | PLANNER AGENT  |        | CAPABILITY       |        | RAG / KNOWLEDGE  |
                  | TaskGraph DAG  |        | REGISTRY         |        | (Stage 5)        |
                  | (Stage 2)      |        | (Stage 3-4)      |        | Embeddings +     |
                  +----------------+        +------------------+        | Retrieval        |
                          |                    |            |           +------------------+
                          v                    v            v
                  +----------------+   +-----------+  +-----------+
                  | ORCHESTRATOR   |   | RESEARCH  |  | ANALYSIS  |
                  | Parallel DAG   |   | AGENT     |  | AGENT     |
                  | Execution      |   +-----------+  +-----------+
                  | (Stage 4)      |
                  +----------------+
                          |
                          v
                  +---------------------------------------+
                  |           EVALUATOR AGENT             |
                  |  Scores output (0-100), PASS/FAIL     |
                  +---------------------------------------+
                          |
                          v
                  +---------------------------------------+
                  |          SYNTHESIZER AGENT            |
                  |  Produces final user-facing answer    |
                  +---------------------------------------+
                          |
                          v
                  +---------------------------------------+
                  |   FINAL ANSWER & OBSERVABLE TRACE     |
                  +---------------------------------------+
```

---

## 3. Implemented Stages

| Stage | Component | Description |
| :--- | :--- | :--- |
| **Stage 1** | Planner Agent | Dynamic task decomposition into typed subtasks with dependency graph |
| **Stage 2** | TaskGraph Model | DAG-based task modeling with validation (cycles, self-deps, unique IDs) |
| **Stage 3** | Orchestrator + Capability Registry | Dependency-aware execution with capability-based agent routing |
| **Stage 4** | Bounded Parallel Execution | Concurrent task execution up to configurable limit (`max_concurrency=4`) |
| **Stage 5** | RAG Knowledge Retrieval | Embedding-based retrieval with in-memory or pgvector repository |

---

## 4. Agent Responsibilities

| Agent | Responsibility | Input | Output |
| :--- | :--- | :--- | :--- |
| **Planner Agent** | Task Decomposition into a dependency-aware TaskGraph | User Task Prompt | `PlannerOutput` → `TaskGraph` |
| **Research Agent** | Targeted factual investigation per subtask | User Context + Single `TaskNode` | `ResearchResult` |
| **Analyst Agent** | Multi-dimensional synthesis and trade-off analysis | User Task + All Research Findings | Analytical Synthesis Report |
| **Evaluator Agent** | Quality audit on Correctness, Completeness, Relevance, and Clarity | User Task + Analyst Output | `EvaluationResult` (score 0-100, PASS/FAIL) |
| **Synthesizer Agent** | Generates the final, authoritative user-facing response | Task + Research + Analysis + Evaluation | Final Polished Answer |

---

## 5. Technology Stack

- **Backend**: Python 3.10+, FastAPI, Pydantic v2, Pydantic-Settings, Uvicorn, HTTPX
- **Testing**: Pytest, Pytest-Asyncio
- **Frontend**: React 19, TypeScript, TanStack Router, TanStack Start, React Flow (@xyflow/react), Tailwind CSS v4, shadcn/ui, Vite 8
- **LLM Abstraction**: Generic `LLMProvider` interface (`GeminiProvider`, `MockLLMProvider`)
- **Knowledge/RAG**: Embedding providers (Gemini, Mock), In-memory or PostgreSQL+pgvector repository

---

## 6. Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Backend Setup

1. Navigate to `backend/`:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `backend/.env` to add your API key:
   ```env
   LLM_PROVIDER=gemini
   LLM_API_KEY=your_gemini_api_key_here
   LLM_MODEL=gemini-3.1-flash-lite
   EVALUATOR_PASS_THRESHOLD=80
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://localhost:8082
   ```
   > **Tip:** For offline development or testing without an API key, set `LLM_PROVIDER=mock`.

5. Start the FastAPI backend server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   The API docs are available at `http://localhost:8000/docs`.

---

### Frontend Setup

1. Navigate to `frontend/`:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   Open **`http://localhost:8082`** in your browser.

> [!IMPORTANT]
> The frontend dev server runs on **port 8082** (configured by `@lovable.dev/vite-tanstack-config`).
> Make sure `CORS_ORIGINS` in your backend `.env` includes `http://localhost:8082`.

---

## 7. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | System configuration status and LLM provider readiness |
| `POST` | `/api/run` | Execute the orchestrated multi-agent pipeline |
| `GET` | `/api/runs` | Retrieve recent execution traces (in-memory) |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 8. Automated Testing

All tests use `MockLLMProvider` and do **not** require live API credentials:

```bash
cd backend
./venv/bin/pytest -v
```

**Backend test coverage:**
- Pipeline end-to-end execution with orchestrated graph
- TaskGraph validation (cycles, dependencies, unique IDs)
- Orchestrator concurrency and capability routing
- Knowledge retrieval and embedding
- API endpoints and error handling

```bash
cd frontend
npm test
```

**Frontend test coverage:**
- API adapter transforms
- App routing
- Console presentation

---

## 9. Directory Structure

```
Multi-agent LLM System/
├── backend/
│   ├── app/
│   │   ├── main.py                          # FastAPI entrypoint & CORS
│   │   ├── config.py                        # Pydantic Settings & env loader
│   │   ├── api/
│   │   │   └── routes.py                    # REST endpoints (/api/run, /api/status, /api/runs)
│   │   ├── models/
│   │   │   ├── schemas.py                   # Pydantic models for tasks, agents & telemetry
│   │   │   └── task_graph.py                # TaskNode, TaskGraph DAG model (Stage 2)
│   │   ├── llm/
│   │   │   ├── base.py                      # Abstract LLMProvider interface & errors
│   │   │   └── provider.py                  # GeminiProvider & MockLLMProvider
│   │   ├── agents/
│   │   │   ├── planner.py                   # Task decomposition agent
│   │   │   ├── researcher.py                # Parametric research agent
│   │   │   ├── analyst.py                   # Synthesis & reasoning agent
│   │   │   ├── evaluator.py                 # Quality audit & scoring agent
│   │   │   └── synthesizer.py               # Final response synthesizer
│   │   ├── orchestration/
│   │   │   ├── orchestrator.py              # DAG executor with parallel concurrency (Stage 3-4)
│   │   │   ├── agents.py                    # CapabilityRegistry & task agents
│   │   │   └── tools.py                     # Controlled tool registry (Stage 4)
│   │   ├── knowledge/
│   │   │   ├── embeddings.py                # Embedding providers (Gemini, Mock)
│   │   │   ├── retrieval.py                 # RetrievalService (Stage 5)
│   │   │   ├── repository.py                # InMemory & Postgres pgvector repos
│   │   │   ├── chunking.py                  # Document chunking
│   │   │   ├── ingestion.py                 # Document ingestion
│   │   │   ├── models.py                    # Knowledge data models
│   │   │   └── sql/pgvector_schema.sql      # Optional PostgreSQL schema
│   │   └── pipeline/
│   │       ├── orchestrated_pipeline.py      # Stage 2-5 orchestrated coordinator
│   │       └── baseline_pipeline.py          # Run history recorder
│   ├── tests/                                # Pytest test suite
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── execution/ExecutionConsole.tsx # Pipeline execution console
│   │   │   ├── graph/ExecutionGraph.tsx       # React Flow DAG visualization
│   │   │   ├── graph/TaskNode.tsx             # Graph task node component
│   │   │   ├── layout/ControlPlane.tsx        # Main control plane UI
│   │   │   └── ui/                            # shadcn/ui component library
│   │   ├── services/
│   │   │   ├── api.ts                         # Backend API client
│   │   │   ├── adapters.ts                    # Response adapters
│   │   │   └── executionEvents.ts             # Execution event system
│   │   ├── types/
│   │   │   ├── api.ts                         # TypeScript API interfaces
│   │   │   └── execution.ts                   # Execution type definitions
│   │   ├── routes/                            # TanStack Router pages
│   │   ├── hooks/                             # React hooks
│   │   ├── router.tsx                         # Router configuration
│   │   └── styles.css                         # Tailwind CSS styles
│   ├── package.json
│   ├── vite.config.ts
│   └── vitest.config.ts
│
├── .gitignore
└── README.md
```

---

## 10. Roadmap: Planned Evolution

- **Stage 1**: Fixed Baseline Pipeline ✅
- **Stage 2**: Dynamic Task Graph (DAG-based subtask modeling) ✅
- **Stage 3**: Dependency-Aware Orchestrator (Capability registry & routing) ✅
- **Stage 4**: Bounded Parallel Execution (Async concurrency for independent tasks) ✅
- **Stage 5**: RAG Knowledge Retrieval (Embedding-based grounded context) ✅
- **Stage 6**: Evaluation-Driven Replanning (Automated feedback loops)
- **Stage 7**: External Tools & Code Execution (Live web search, sandboxed interpreters)
- **Stage 8**: Persistent Execution State (Resumable workflows & audit trails)
- **Stage 9**: Benchmarking (Single LLM vs Fixed Multi-Agent vs Adaptive Multi-Agent)

---

## 11. License

This project is developed as part of a Final Year Engineering Major Project.
