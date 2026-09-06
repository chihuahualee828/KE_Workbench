"use client";

import { ArrowRight, ArrowUpRight, Box, Cpu, FlaskConical, History, Network, SlidersHorizontal } from "lucide-react";
import Link from "next/link";

import { useWorkbenchData } from "@/components/DataState";
import { GenerationPanel } from "@/components/GenerationPanel";
import { ExperimentHistory } from "@/components/ExperimentHistory";
import { ModelCacheManager } from "@/components/ModelCacheManager";
import { PageHeader } from "@/components/PageHeader";
import { PatchHistory } from "@/components/PatchHistory";

const workflows = [
  { href: "/trace", icon: Network, step: "01 / UNDERSTAND", title: "Look inside the model", description: "Trace the activations behind a factual answer.", color: "bg-cyan/10 text-cyan" },
  { href: "/edit", icon: SlidersHorizontal, step: "02 / EDIT", title: "Update a piece of knowledge", description: "Create a reversible, fact-specific weight edit.", color: "bg-amber-50 text-amber-700" },
  { href: "/evaluate", icon: FlaskConical, step: "03 / VALIDATE", title: "Measure what changed", description: "Compare edit quality and effects on other facts.", color: "bg-teal-50 text-teal-700" }
];

export default function HomePage() {
  const { models, patches, experiments, runtime, status, refresh } = useWorkbenchData();
  const loadedModels = runtime.filter((model) => model.loaded);
  const stats = [
    { label: "Models loaded", value: loadedModels.length, note: loadedModels.length ? "Ready for inference" : "Choose a model to begin", icon: Cpu, href: "#models" },
    { label: "Saved patches", value: patches.length, note: `${patches.filter((patch) => patch.active).length} currently active`, icon: Box, href: "#patches" },
    { label: "Experiments", value: experiments.length, note: `${experiments.filter((experiment) => experiment.status === "completed").length} completed`, icon: FlaskConical, href: "/jobs" }
  ];

  return (
    <div>
      <PageHeader eyebrow="Your workspace" title="Knowledge, under the microscope." description="Explore what your model knows. Understand how it works. Make a precise edit and measure the difference." status={status} />

      <section className="section grid gap-3 sm:grid-cols-3" aria-label="Workspace statistics">
        {stats.map((stat) => <Link key={stat.label} href={stat.href} className="panel group flex items-start justify-between gap-3 transition hover:border-cyan/35">
          <div><div className="text-xs font-medium text-ink/55">{stat.label}</div><div className="my-2 text-3xl font-semibold tracking-tight tabular-nums">{status === "ready" ? stat.value : "—"}</div><div className="text-xs text-ink/50">{status === "ready" ? stat.note : status === "loading" ? "Connecting to workspace…" : "Connect the backend to view"}</div></div>
          <stat.icon size={18} className="text-ink/35 transition group-hover:text-cyan" />
        </Link>)}
      </section>

      <section className="section grid gap-3 xl:grid-cols-3" aria-label="Explore the workflow">
        {workflows.map((item) => <Link key={item.href} href={item.href} className="panel group transition hover:-translate-y-0.5 hover:border-cyan/30 hover:shadow-md">
          <div className="mb-4 flex items-center justify-between"><span className={`inline-flex h-9 w-9 items-center justify-center rounded-lg ${item.color}`}><item.icon size={18} /></span><ArrowUpRight size={16} className="text-ink/30 transition group-hover:text-cyan" /></div>
          <p className="text-[9px] font-semibold tracking-[0.14em] text-ink/40">{item.step}</p><h2 className="mt-2 text-sm font-semibold">{item.title}</h2><p className="mt-1.5 text-xs leading-5 text-ink/55">{item.description}</p>
        </Link>)}
      </section>

      <section className="section">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-base font-semibold tracking-tight">Try a prompt</h2><p className="mt-1 text-xs text-ink/55">Get a baseline before you explore or edit.</p></div><Link className="inline-flex items-center gap-2 text-xs font-medium text-cyan hover:underline" href="/playground">Open playground<ArrowRight size={14} /></Link></div>
        {status === "ready" && !loadedModels.length ? <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-cyan/15 bg-cyan/5 px-4 py-3"><p className="text-xs leading-5 text-ink/70">Start by downloading a model and loading it into memory using the controls below.</p><Link href="#models" className="inline-flex items-center gap-2 text-xs font-medium text-cyan">Browse models<ArrowRight size={13} /></Link></div> : null}
        <GenerationPanel models={models} patches={patches} runtime={runtime} onPatchesChanged={refresh} sessionScope="workbench" />
      </section>

      <section className="section" id="patches"><PatchHistory patches={patches} runtime={runtime} onPatchesChanged={refresh} /></section>
      <section className="section"><div className="panel"><div className="panel-heading"><div className="flex items-center gap-2"><History size={17} className="text-ink/45" /><h2 className="text-sm font-semibold">Recent experiments</h2></div><Link href="/jobs" className="text-xs font-medium text-cyan hover:underline">View all activity</Link></div><ExperimentHistory experiments={experiments} limit={6} onChanged={refresh} showControls={false} /></div></section>
      <section className="section" id="models"><ModelCacheManager models={models} apiStatus={status} onModelsChanged={refresh} /></section>
    </div>
  );
}
