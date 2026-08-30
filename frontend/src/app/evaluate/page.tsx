"use client";

import { ApiStatus, useWorkbenchData } from "@/components/DataState";
import { EvaluationDashboard } from "@/components/EvaluationDashboard";

export default function EvaluatePage() {
  const { models, patches, runtime, status } = useWorkbenchData();
  return (
    <div>
      <section className="section flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Evaluation</h1>
          <p className="mt-2 text-sm text-ink/70">Measure efficacy, paraphrase success, specificity, KL drift, entropy, and sample generations for patches.</p>
        </div>
        <ApiStatus status={status} />
      </section>
      <section className="section">
        <EvaluationDashboard models={models} patches={patches} runtime={runtime} />
      </section>
    </div>
  );
}
