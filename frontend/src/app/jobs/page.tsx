"use client";

import { ApiStatus, useWorkbenchData } from "@/components/DataState";
import { RunningStatusPanel } from "@/components/RunningStatusPanel";

export default function JobsPage() {
  const { status } = useWorkbenchData();
  return (
    <div>
      <section className="section flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Jobs</h1>
          <p className="mt-2 text-sm text-ink/70">Monitor active downloads, model loads, and experiment history.</p>
        </div>
        <ApiStatus status={status} />
      </section>
      <section className="section">
        <RunningStatusPanel />
      </section>
    </div>
  );
}
