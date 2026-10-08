<!-- LOVABLE:BEGIN -->

> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.

<!-- LOVABLE:END -->

## Frontend architecture

- Keep external API calls in `src/services/api.ts` and normalize responses in `adapters.ts`; graph and inspector components consume only the normalized model to preserve API compatibility.
- Model declared dependencies only in execution view, and keep conceptual architecture separate; baseline responses do not prove dependency order or parallelism.
- Consume returned traces through `ExecutionEventSource` and label them post-run; keep any demo playback isolated to prevent simulated telemetry being mistaken for real execution.
- Mount the control plane at the index route and constrain page overflow globally; only graph interaction and console/inspector regions scroll internally.
- Render execution particles in a custom React Flow edge only for declared dependencies into a returned running task from a completed task; animation never mutates task state or implies a live stream.
