"use client";

import { ApiStatus, useWorkbenchData } from "@/components/DataState";
import { EditForm } from "@/components/EditForm";
import { ModelCacheManager } from "@/components/ModelCacheManager";

export default function EditPage() {
  const { models, patches, status, refresh } = useWorkbenchData();
  return (
    <div>
      <section className="section flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Edit Patches</h1>
          <p className="mt-2 text-sm text-ink/70">Create reversible ROME and MEMIT patch records, then review saved patch history.</p>
        </div>
        <ApiStatus status={status} />
      </section>
      <section className="section">
        <EditForm models={models} patches={patches} onPatchesChanged={refresh} />
      </section>
      <section className="section">
        <ModelCacheManager models={models} onModelsChanged={refresh} />
      </section>
    </div>
  );
}
