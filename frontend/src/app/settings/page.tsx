"use client";

import { ApiStatus, useWorkbenchData } from "@/components/DataState";
import { SettingsPanel } from "@/components/SettingsPanel";

export default function SettingsPage() {
  const { models, runtime, status } = useWorkbenchData();
  return (
    <div>
      <section className="section flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Settings</h1>
          <p className="mt-2 text-sm text-ink/70">Manage local runtime configuration and download credentials.</p>
        </div>
        <ApiStatus status={status} />
      </section>
      <section className="section">
        <SettingsPanel models={models} runtime={runtime} />
      </section>
    </div>
  );
}
