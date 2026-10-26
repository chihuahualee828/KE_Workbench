"use client";

import { PageHeader } from "@/components/PageHeader";
import { useWorkbenchData } from "@/components/DataState";
import { EditForm } from "@/components/EditForm";
import { ModelCacheManager } from "@/components/ModelCacheManager";

export default function EditPage() {
  const { models, patches, runtime, status, refresh } = useWorkbenchData();
  return (
    <div>
      <PageHeader eyebrow="Experiment" title="Knowledge edits" description="Create and manage reversible knowledge edits with ROME, r-ROME, Knowledge Neurons, MEMIT, FT, LoRA, EMMET, MEMIT-FE, PMET, DPO, EAMET, AlphaEdit, or SPHERE." status={status} />
      <section className="section">
        <EditForm models={models} patches={patches} runtime={runtime} onPatchesChanged={refresh} />
      </section>
      <section className="section">
        <ModelCacheManager models={models} apiStatus={status} onModelsChanged={refresh} />
      </section>
    </div>
  );
}
