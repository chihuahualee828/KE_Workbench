"use client";

import { ApiStatus, useWorkbenchData } from "@/components/DataState";
import { CausalTraceHeatmap } from "@/components/CausalTraceHeatmap";

export default function TracePage() {
  const { models, runtime, status } = useWorkbenchData();
  return (
    <div>
      <section className="section flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Causal Tracing</h1>
          <p className="mt-2 text-sm text-ink/70">Corrupt subject activations, restore candidate states, and inspect indirect effects by layer and token.</p>
        </div>
        <ApiStatus status={status} />
      </section>
      <section className="section">
        <CausalTraceHeatmap models={models} runtime={runtime} />
      </section>
    </div>
  );
}
