import { BaseEdge, getSmoothStepPath, type EdgeProps, type Edge } from "@xyflow/react";

export type ExecutionFlowEdge = Edge<{ active: boolean; completed: boolean }, "execution">;

/** Motion reflects returned task state only; it never advances execution state. */
export function ExecutionEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
}: EdgeProps<ExecutionFlowEdge>) {
  const [path] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 18,
  });
  return (
    <g
      className={`execution-edge ${data?.active ? "is-active" : ""} ${data?.completed ? "is-completed" : ""} ${selected ? "is-selected" : ""}`}
    >
      <BaseEdge id={id} path={path} />
      {data?.active && (
        <circle r="1.8" className="execution-particle" aria-hidden="true">
          <animateMotion dur="3.6s" repeatCount="indefinite" path={path} />
        </circle>
      )}
    </g>
  );
}
