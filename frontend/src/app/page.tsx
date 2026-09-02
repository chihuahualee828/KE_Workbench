"use client";

import Link from "next/link";

import { ApiStatus, useWorkbenchData } from "@/components/DataState";
import { GenerationPanel } from "@/components/GenerationPanel";
import { ModelCacheManager } from "@/components/ModelCacheManager";
import { PatchHistory } from "@/components/PatchHistory";

const capabilities = [
  "Generation",
  "Logit lens",
  "Causal tracing",
  "ROME single edits",
  "Knowledge Neurons single edits",
  "MEMIT batch edits",
  "Patch evaluation"
];

export default function HomePage() {
  const { models, patches, experiments, runtime, status, refresh } = useWorkbenchData();

  return (
    <div>
      <section className="section grid gap-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
        <div>
          <h1 className="text-3xl font-semibold text-ink md:text-4xl">Interactive LLM Knowledge Editing Workbench</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-ink/70">
            Explore factual prompts, inspect layer behavior, trace causal states, create reversible edit patches, and compare before/after metrics from one local interface.
          </p>
        </div>
        <ApiStatus status={status} />
      </section>

      <section className="section">
        <GenerationPanel models={models} patches={patches} runtime={runtime} onPatchesChanged={refresh} sessionScope="workbench" />
      </section>

      <section className="section grid gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <div className="panel">
          <h2 className="text-base font-semibold">Supported Features</h2>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {capabilities.map((capability) => (
              <div key={capability} className="rounded-md border border-line bg-panel px-3 py-2 text-sm">
                {capability}
              </div>
            ))}
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Link className="ghost-button" href="/logit-lens">
              Open Lens
            </Link>
            <Link className="ghost-button" href="/trace">
              Open Trace
            </Link>
            <Link className="ghost-button" href="/edit">
              Open Edits
            </Link>
          </div>
        </div>
        <div className="panel">
          <h2 className="text-base font-semibold">Recent Experiments</h2>
          <div className="mt-3 divide-y divide-line">
            {experiments.slice(0, 6).map((experiment) => (
              <div key={experiment.id} className="grid grid-cols-[1fr_auto] gap-3 py-2 text-sm">
                <span>{experiment.type.replace("_", " ")}</span>
                <span className="font-mono text-xs text-ink/60">{experiment.status}</span>
              </div>
            ))}
            {experiments.length === 0 ? <div className="py-3 text-sm text-ink/60">No experiments yet.</div> : null}
          </div>
        </div>
      </section>

      <section className="section">
        <PatchHistory patches={patches} runtime={runtime} onPatchesChanged={refresh} />
      </section>

      <section className="section">
        <ModelCacheManager models={models} apiStatus={status} onModelsChanged={refresh} />
      </section>
    </div>
  );
}
