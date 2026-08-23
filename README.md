# Adaptive Multi-Agent LLM System for Task Decomposition and Collaborative Problem Solving (early)

**Final Year Computer Science & Engineering Major Project**  
*Current Milestone: Stage 1 — Fixed Multi-Agent Baseline Prototype*

---

## 1. Project Overview

The objective of this engineering major project is to build an adaptive, orchestrator-driven multi-agent LLM system capable of taking high-level, complex analytical tasks, decomposing them into discrete subtasks, routing them to specialized agents, evaluating intermediate outputs, and dynamically adapting execution plans when results fall short of quality benchmarks.

### Stage 1 Scope: Fixed Baseline Pipeline

We are starting from **Stage 1 (Fixed Multi-Agent Baseline)**. Stage 1 intentionally establishes the foundational multi-agent communication pipeline, provider-agnostic LLM abstraction, role separation, structured task decomposition, and observability telemetry before any dynamic or adaptive complexities are introduced.

> [!IMPORTANT]
> **Stage 1 is a FIXED MULTI-AGENT BASELINE.**
> It intentionally executes a deterministic, sequential pipeline:
> $$\text{User Task} \longrightarrow \text{Planner} \longrightarrow \text{Subtasks} \longrightarrow \text{Research Agents} \longrightarrow \text{Analyst} \longrightarrow \text{Evaluator} \longrightarrow \text{Synthesizer} \longrightarrow \text{Final Answer}$$

---

## 2. Stage 1 Conceptual Architecture

```
                                  +---------------------------------------+
                                  |                 USER                  |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         REACT FRONTEND (Vite)         |
                                  +---------------------------------------+
                                                      |  HTTP POST /api/run
                                                      v
                                  +---------------------------------------+
                                  |       FASTAPI BACKEND & ROUTER        |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       BASELINE PIPELINE COORDINATOR   |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |            PLANNER AGENT              |
                                  |  - Decomposes task into typed subtasks|
                                  +---------------------------------------+
                                                      |  Subtasks [T1, T2, ...]
                                                      v
                                  +---------------------------------------+
                                  |          RESEARCH AGENT(S)            |
                                  |  - Parametric LLM knowledge retrieval |
                                  +---------------------------------------+
                                                      |  Research Findings
                                                      v
                                  +---------------------------------------+
                                  |            ANALYST AGENT              |
                                  |  - Synthesis, trade-offs & reasoning  |
                                  +---------------------------------------+
                                                      |  Analytical Report
                                                      v
                                  +---------------------------------------+
                                  |           EVALUATOR AGENT             |
                                  |  - Scores output (0-100), PASS/FAIL   |
                                  +---------------------------------------+
                                                      |  Score & Feedback
                                                      v
                                  +---------------------------------------+
                                  |          SYNTHESIZER AGENT            |
                                  |  - Produces final user-facing answer  |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   FINAL ANSWER & OBSERVABLE TRACE     |
                                  +---------------------------------------+
```

---

## 3. Agent Responsibilities & Pipeline Workflow

| Agent | Responsibility | Input | Output |
| :--- | :--- | :--- | :--- |
| **Planner Agent** | Task Decomposition into discrete subtasks. | User Task Prompt | Validated `PlannerOutput` (`SubTask` list with `id`, `description`, `type`) |
| **Research Agent** | Targeted factual investigation per subtask. | User Context + Single `SubTask` | `ResearchResult` (Parametric knowledge findings) |
| **Analyst Agent** | Multi-dimensional synthesis and trade-off analysis. | User Task + All Research Findings | Structured Analytical Synthesis Report |
| **Evaluator Agent** | Quality audit on Correctness, Completeness, Relevance, and Clarity. | User Task + Analyst Output | `EvaluationResult` (`score`, `status`: PASS/FAIL, `feedback`) |
| **Synthesizer Agent** | Generates the final, authoritative user-facing response. | Task + Research + Analysis + Evaluation | Final Polished Answer presented in UI |

> [!NOTE]
> **Clarification regarding the Research Agent:**  
> Stage 1 operates purely on the LLM's **parametric knowledge**. There is NO live web search, web scraping, document retrieval, or RAG in Stage 1. Retrieval-Augmented Generation and external tools will be integrated in subsequent stages.

---

## 4. Technology Stack

- **Backend**: Python 3.10+, FastAPI, Pydantic v2, Pydantic-Settings, Uvicorn, HTTPX
- **Testing**: Pytest, Pytest-Asyncio
- **Frontend**: React 19, TypeScript, Vite, Modern CSS Design System (Lucide icons)
- **LLM Abstraction**: Generic `LLMProvider` interface decoupled from concrete implementations (`GeminiProvider`, `MockLLMProvider`)

---

## 5. Directory Structure

```
Multi-agent LLM System/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application entrypoint & CORS
│   │   ├── config.py                   # Pydantic Settings & environment loader
│   │   ├── api/
│   │   │   └── routes.py               # REST endpoints (/api/run, /api/status, /api/runs)
│   │   ├── models/
│   │   │   └── schemas.py              # Pydantic models for tasks, agents & telemetry
│   │   ├── llm/
│   │   │   ├── base.py                 # Abstract LLMProvider interface & custom errors
│   │   │   └── provider.py             # GeminiProvider & MockLLMProvider implementations
│   │   ├── agents/
│   │   │   ├── planner.py              # Task decomposition agent
│   │   │   ├── researcher.py           # Parametric knowledge research agent
│   │   │   ├── analyst.py              # Synthesis & reasoning agent
│   │   │   ├── evaluator.py            # Quality audit & scoring agent
│   │   │   └── synthesizer.py          # Final user response synthesizer
│   │   └── pipeline/
│   │       └── baseline_pipeline.py    # Fixed sequential coordinator & trace recorder
│   ├── tests/
│   │   ├── test_schemas.py             # Schema validation tests
│   │   ├── test_pipeline.py            # Pipeline mock integration tests
│   │   ├── test_api.py                 # FastAPI route tests
│   │   └── test_error_handling.py      # Missing key and validation error tests
│   ├── requirements.txt                # Python dependencies
│   └── .env.example                    # Template environment variables
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx              # Application header & provider status pill
│   │   │   ├── ConfigBanner.tsx        # Missing credentials / setup guidance alert
│   │   │   ├── TaskInput.tsx           # Task input textarea & demo prompt presets
│   │   │   ├── PipelineTimeline.tsx    # Stage-by-stage latency & status timeline
│   │   │   ├── SubtaskView.tsx         # Decomposed subtask card viewer
│   │   │   ├── ResearchView.tsx        # Parametric research findings accordion
│   │   │   ├── AnalystView.tsx         # Analytical synthesis card
│   │   │   ├── EvaluatorView.tsx       # Evaluator score badge (0-100) & critique
│   │   │   ├── FinalAnswerView.tsx     # Final answer viewer with copy function
│   │   │   └── ExecutionTraceLog.tsx   # Detailed latency and telemetry log table
│   │   ├── services/
│   │   │   └── api.ts                  # Backend API client
│   │   ├── types/
│   │   │   └── index.ts                # TypeScript data interfaces
│   │   ├── App.tsx                     # Main layout & tabbed execution trace manager
│   │   ├── index.css                   # Custom modern dark design system
│   │   └── main.tsx                    # React mounting entrypoint
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── .gitignore
└── README.md
```

---

## 6. Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Backend Setup

1. Open terminal and navigate to `backend/`:
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
   *Edit `backend/.env` to configure your settings:*
   ```env
   LLM_PROVIDER=gemini
   LLM_API_KEY=your_gemini_api_key_here
   LLM_MODEL=gemini-1.5-flash
   EVALUATOR_PASS_THRESHOLD=80
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173
   ```
   *(For offline development or testing without an API key, set `LLM_PROVIDER=mock`)*

5. Start the FastAPI backend server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   *The backend API documentation is available at `http://localhost:8000/docs`.*

---

### Frontend Setup

1. Open a new terminal and navigate to `frontend/`:
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
   *Open `http://localhost:5173` in your browser.*

---

## 7. Automated Testing

All automated tests use `MockLLMProvider` and do not require live API credentials:

```bash
cd backend
./venv/bin/pytest -v
```

**Test Coverage Includes:**
- Pydantic schema validation (`SubTask`, `PlannerOutput`, `EvaluationResult`, `RunRequest`)
- Full fixed pipeline execution end-to-end with measured stage latencies
- API endpoints (`GET /api/status`, `POST /api/run`, `GET /api/runs`, `GET /`)
- Error handling (missing API key 503 response, invalid input length 422, unconfigured providers)

---

## 8. Example Demonstration Tasks

The system includes pre-configured demo presets:

1. **Electric vs Petrol Vehicle Comparison:**
   > *"Compare electric vehicles and petrol vehicles for a college student considering cost, maintenance, environmental impact and practicality."*

2. **Renewable Energy Investment (India):**
   > *"Analyze whether India should increase investment in renewable energy considering economic, environmental and policy factors."*

---

## 9. What is Intentionally NOT Implemented in Stage 1

To maintain a clean baseline without premature complexity, the following capabilities are **explicitly deferred** to subsequent project stages:

- **Dynamic Routing & Agent Selection** (Stage 4)
- **Task Graph & Dependency Management** (Stage 3)
- **Dynamic Orchestration Engine** (Stage 2)
- **Parallel Subtask Execution** (Stage 5)
- **Evaluation-Driven Replanning Loops** (Stage 6)
- **Retrieval-Augmented Generation / Vector Databases** (Stage 7)
- **Live Search & External Tools** (Stage 8)
- **Persistent Database State (PostgreSQL / Redis / MongoDB)** (Stage 9)
- **Empirical Benchmarking Suite** (Stage 10)
- **Framework abstractions (LangChain, LangGraph, CrewAI)** (Intentionally avoided in favor of native, extensible modular design)

---

## 10. Roadmap: Planned Evolution

- **Stage 1**: Fixed Baseline Pipeline *(Current)*
- **Stage 2**: Orchestrator (Central state machine replacing fixed sequential scripts)
- **Stage 3**: Task Graph & Dependency Resolution (DAG-based subtask modeling)
- **Stage 4**: Dynamic Agent Selection (Contextual agent registry & assignment)
- **Stage 5**: Parallel Execution (Asynchronous concurrency for independent subtasks)
- **Stage 6**: Evaluation-Driven Replanning (Automated feedback loops and dynamic repair)
- **Stage 7**: RAG Integration (Vector embeddings and grounded knowledge retrieval)
- **Stage 8**: External Tools & Code Execution (Live web search, sandbox interpreters)
- **Stage 9**: Persistent Execution State (Resumable workflows & audit trails)
- **Stage 10**: Benchmarking (Empirical comparison: Single LLM vs Fixed Multi-Agent vs Adaptive Multi-Agent)
