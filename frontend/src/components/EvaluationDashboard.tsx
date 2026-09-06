"use client";

import { BarChart3, FlaskConical, Loader2, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import { api } from "@/lib/api";
import type { ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { useWorkbenchSession } from "./WorkbenchSession";

export function EvaluationDashboard({ models, patches, runtime = [] }: { models: SupportedModel[]; patches: Patch[]; runtime?: ModelRuntimeStatus[] }) {
  const [modelId, setModelId] = usePersistentState("kew:evaluate:modelId", "gpt2-xl");
  const [patchId, setPatchId] = usePersistentState("kew:evaluate:patchId", patches[0]?.patch_id ?? "");
  const [prompts, setPrompts] = usePersistentState("kew:evaluate:prompts", "The Space Needle is located in the city of\nThe landmark known as the Space Needle is in\nThe Eiffel Tower is located in the city of");
  const { evaluationResult: result, setEvaluationResult: setResult } = useWorkbenchSession();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const modelLoaded = runtime.some((status) => status.model_id === modelId && status.loaded);

  useLoadedModelSelection(models, runtime, setModelId);

  async function run() {
    if (!modelLoaded || busy || !patchId) return;
    setError("");
    setBusy(true);
    try {
      const response = await api.evaluate({
        model_id: modelId,
        patch_id: patchId,
        prompts: prompts.split("\n").map((line) => line.trim()).filter(Boolean)
      });
      setResult(response.result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const metrics = (result?.metrics ?? {}) as Record<string, number>;
  const generations = Array.isArray(result?.sample_generations) ? result.sample_generations as Array<Record<string, unknown>> : [];

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[300px_minmax(0,1fr)]">
      <section className="panel">
        <div className="panel-heading"><h2 className="flex items-center gap-2 text-sm font-semibold"><SlidersHorizontal size={16} className="text-cyan" />Evaluation setup</h2></div>
        <fieldset disabled={busy} className="grid min-w-0 gap-4">
        <ModelSelector models={models} value={modelId} onChange={setModelId} />
        <label className="grid gap-2">
          <span className="label">Patch</span>
          <select className="control" value={patchId} onChange={(event) => setPatchId(event.target.value)}>
            <option value="">Select patch</option>
            {patches.map((patch) => (
              <option key={patch.patch_id} value={patch.patch_id}>
                {patch.method.toUpperCase()} {patch.patch_id}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-2">
          <span className="label">Prompts</span>
          <textarea className="control min-h-40" value={prompts} onChange={(event) => setPrompts(event.target.value)} />
          <span className="text-[11px] leading-5 text-ink/55">One prompt per line, up to 10 per evaluation. Include rephrased and unrelated facts.</span>
        </label>
        {!patches.length ? <p className="text-xs leading-5 text-ink/55">Create a knowledge edit first, then select its patch here.</p> : !modelLoaded ? <p className="text-xs leading-5 text-ink/55">Load the selected model to enable evaluation.</p> : null}
        <button className="tool-button" onClick={run} disabled={busy || !patchId || !modelLoaded} title={modelLoaded ? "Run evaluation" : "Load the selected model before evaluating"}>
          {busy ? <Loader2 size={16} className="animate-spin" /> : <BarChart3 size={16} />}
          {busy ? "Evaluating patch…" : "Run evaluation"}
        </button>
        </fieldset>
        {error ? <div role="alert" className="inline-error mt-4">{error}</div> : null}
      </section>
      <section className="panel grid gap-4" aria-busy={busy}>
        <h2 className="flex items-center gap-2 text-sm font-semibold"><FlaskConical size={17} className="text-cyan" />Evaluation results</h2>
        {busy ? <div role="status" className="flex items-center gap-2 text-xs text-cyan"><Loader2 size={14} className="animate-spin" />Comparing clean, selected-patch, and current-model generations…</div> : null}
        {result ? <>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {Object.entries(metrics).map(([key, value]) => (
            <div key={key} className="rounded-md border border-line bg-panel p-3">
              <div className="label">{key.replace(/_/g, " ")}</div>
              <div className="mt-2 text-2xl font-semibold tabular-nums">{typeof value === "number" ? value.toFixed(3) : String(value)}</div>
            </div>
          ))}
        </div>
        {generations.map((generation, index) => <div key={index} className="overflow-hidden rounded-lg border border-line"><p className="border-b border-line bg-panel px-4 py-3 text-xs font-medium leading-5">{String(generation.prompt ?? "")}</p><div className="grid divide-y divide-line">{[["clean_model", "Clean model"], ["selected_patch", "Selected patch"], ["current_model", "Current model"]].map(([key, label]) => <div key={key} className="p-4"><div className="label">{label}</div><p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6">{String(generation[key] ?? "—")}</p></div>)}</div></div>)}
        <details className="rounded-lg border border-line p-3"><summary className="text-xs font-medium text-ink/60">View raw evaluation data</summary><pre className="mt-3 overflow-auto rounded-lg bg-panel p-4 text-xs leading-5">{JSON.stringify(result, null, 2)}</pre></details>
        </> : <div className="empty-state min-h-[360px] rounded-lg border border-dashed border-line bg-panel/50"><span className="icon-tile bg-white shadow-sm"><BarChart3 size={22} strokeWidth={1.5} /></span><h3 className="text-sm font-medium">Put your edit to the test</h3><p className="max-w-sm text-xs leading-6 text-ink/55">Select a saved patch and a set of prompts to compare clean, patched, and current-model responses.</p></div>}
      </section>
    </div>
  );
}
