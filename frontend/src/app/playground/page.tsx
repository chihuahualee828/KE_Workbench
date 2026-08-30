"use client";

import { ApiStatus, useWorkbenchData } from "@/components/DataState";
import { GenerationPanel } from "@/components/GenerationPanel";

export default function PlaygroundPage() {
  const { models, patches, runtime, status } = useWorkbenchData();
  return (
    <div>
      <section className="section flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Prompt Playground</h1>
          <p className="mt-2 text-sm text-ink/70">Run baseline and patched generations against supported local models.</p>
        </div>
        <ApiStatus status={status} />
      </section>
      <section className="section">
        <GenerationPanel models={models} patches={patches} runtime={runtime} />
      </section>
    </div>
  );
}
