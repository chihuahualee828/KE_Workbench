"use client";

import { ApiStatus, useWorkbenchData } from "@/components/DataState";
import { LogitLensPanel } from "@/components/LogitLensPanel";

export default function LogitLensPage() {
  const { models, status } = useWorkbenchData();
  return (
    <div>
      <section className="section flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Logit Lens</h1>
          <p className="mt-2 text-sm text-ink/70">Track how target-token probability and top predictions evolve across transformer layers.</p>
        </div>
        <ApiStatus status={status} />
      </section>
      <section className="section">
        <LogitLensPanel models={models} />
      </section>
    </div>
  );
}

