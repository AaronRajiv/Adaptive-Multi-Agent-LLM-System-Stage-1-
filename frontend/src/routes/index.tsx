import { createFileRoute } from "@tanstack/react-router";
import { ControlPlane } from "@/components/layout/ControlPlane";
export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Adaptive Multi-Agent LLM System — Execution Control" },
      {
        name: "description",
        content:
          "Task orchestration control plane with an interactive dependency graph, real execution traces, and analytical synthesis.",
      },
      { property: "og:title", content: "Adaptive Multi-Agent LLM System — Execution Control" },
      {
        property: "og:description",
        content:
          "Inspect task graphs, agent activity, and returned execution telemetry in one focused workspace.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ControlPlane,
});
