"use client";

import { PageHeader } from "@/components/PageHeader";
import { useWorkbenchData } from "@/components/DataState";
import { LogitLensPanel } from "@/components/LogitLensPanel";

export default function LogitLensPage() {
  const { models, runtime, status } = useWorkbenchData();
  return (
    <div>
      <PageHeader eyebrow="Explore / Interpretability" title="Logit lens" description="Follow predictions through the network to see where an answer takes shape." status={status} />
      <section className="section">
        <LogitLensPanel models={models} runtime={runtime} />
      </section>
    </div>
  );
}
