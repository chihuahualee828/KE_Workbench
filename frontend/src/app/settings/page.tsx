"use client";

import { PageHeader } from "@/components/PageHeader";
import { useWorkbenchData } from "@/components/DataState";
import { SettingsPanel } from "@/components/SettingsPanel";

export default function SettingsPage() {
  const { models, runtime, status } = useWorkbenchData();
  return (
    <div>
      <PageHeader eyebrow="Manage" title="Workspace settings" description="Configure your local runtime, models, and download credentials." status={status} />
      <section className="section">
        <SettingsPanel models={models} runtime={runtime} />
      </section>
    </div>
  );
}
