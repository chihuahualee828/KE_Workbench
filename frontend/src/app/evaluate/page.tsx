"use client";

import { PageHeader } from "@/components/PageHeader";
import { useWorkbenchData } from "@/components/DataState";
import { EvaluationDashboard } from "@/components/EvaluationDashboard";

export default function EvaluatePage() {
  const { models, patches, runtime, status } = useWorkbenchData();
  return (
    <div>
      <PageHeader eyebrow="Experiment" title="Evaluation" description="Understand an edit’s impact on accuracy, generalization, and unrelated knowledge." status={status} />
      <section className="section">
        <EvaluationDashboard models={models} patches={patches} runtime={runtime} />
      </section>
    </div>
  );
}
