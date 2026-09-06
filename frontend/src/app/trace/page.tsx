"use client";

import { PageHeader } from "@/components/PageHeader";
import { useWorkbenchData } from "@/components/DataState";
import { CausalTraceHeatmap } from "@/components/CausalTraceHeatmap";

export default function TracePage() {
  const { models, runtime, status } = useWorkbenchData();
  return (
    <div>
      <PageHeader eyebrow="Explore / Interpretability" title="Causal tracing" description="Find the states that influence an answer by restoring clean activations in a corrupted run." status={status} />
      <section className="section">
        <CausalTraceHeatmap models={models} runtime={runtime} />
      </section>
    </div>
  );
}
