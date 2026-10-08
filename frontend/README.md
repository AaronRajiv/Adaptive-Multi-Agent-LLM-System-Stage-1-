# Adaptive Canvas

Build a complete frontend redesign for the Adaptive Multi-Agent LLM System as detailed in the attached specifications (Build a complete frontend redesign for this (pasted).txt) with index.ts types and api.ts client.

Key requirements:

1. Strict 100vw x 100dvh single viewport layout (no page scrolling) with dark theme (#090d14) and glass translucent surfaces.
2. Tier 1 Top Bar (~48px): System title, subtitle, live status from GET /api/status (● SYSTEM ONLINE), demo task presets, Architecture/Live Execution toggle, Clear Canvas.
3. Tier 2 Task Command Bar (~52px): Glass input ("Enter complex analytical task..."), Run Pipeline button (⌘+Enter).
4. Tier 3 Upper Canvas: Interactive React Flow (@xyflow/react) task graph with custom nodes (TaskNode with status badges, specialized RAG Node with query/provenance, Evaluator Node with amber score ring, Synthesis Node), animated dependency edges, minimap, zoom and fit controls.
5. Tier 3 Lower Console (~200px): Split 45% Live Event Stream & Telemetry and 55% Inspector with tabs (Final Synthesis markdown view, Node Inspector, RAG Provenance).
6. Preserves real backend integration via VITE_API_BASE_URL (fallback http://localhost:8000) using /api/status, /api/run, and /api/runs with adapter layer, handling error states and graceful offline fallback.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/3c3830dc-73f2-435f-ba2c-057bd890eb6c).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
