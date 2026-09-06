"use client";

import { PageHeader } from "@/components/PageHeader";
import { useWorkbenchData } from "@/components/DataState";
import { RunningStatusPanel } from "@/components/RunningStatusPanel";

export default function JobsPage() {
  const { status } = useWorkbenchData();
  return (
    <div>
      <PageHeader eyebrow="Manage" title="Activity" description="Follow model downloads, runtime loads, and your experiment history." status={status} />
      <section className="section">
        <RunningStatusPanel />
      </section>
    </div>
  );
}
