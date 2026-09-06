"use client";

import { PageHeader } from "@/components/PageHeader";
import { useWorkbenchData } from "@/components/DataState";
import { GenerationPanel } from "@/components/GenerationPanel";

export default function PlaygroundPage() {
  const { models, patches, runtime, status } = useWorkbenchData();
  return (
    <div>
      <PageHeader eyebrow="Explore" title="Prompt playground" description="Start with a prompt. See what your model knows, then compare it after an edit." status={status} />
      <section className="section">
        <GenerationPanel models={models} patches={patches} runtime={runtime} sessionScope="generate" />
      </section>
    </div>
  );
}
