"use client";

import { BarChart3 } from "lucide-react";
import { useState } from "react";

import { api } from "@/lib/api";
import type { ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { useWorkbenchSession } from "./WorkbenchSession";

export function EvaluationDashboard({ models, patches, runtime = [] }: { models: SupportedModel[]; patches: Patch[]; runtime?: ModelRuntimeStatus[] }) {
  const [modelId, setModelId] = usePersistentState("kew:evaluate:modelId", "gpt2");
  const [patchId, setPatchId] = usePersistentState("kew:evaluate:patchId", patches[0]?.patch_id ?? "");
  const [prompts, setPrompts] = usePersistentState("kew:evaluate:prompts", "The Space Needle is located in the city of\nThe landmark known as the Space Needle is in\nThe Eiffel Tower is located in the city of");
  const { evaluationResult: result, setEvaluationResult: setResult } = useWorkbenchSession();
  const [busy, setBusy] = useState(false);
  const modelLoaded = runtime.some((status) => status.model_id === modelId && status.loaded);

  async function run() {
    if (!modelLoaded || busy) return;
    setBusy(true);
    try {
      const response = await api.evaluate({
        model_id: modelId,
        patch_id: patchId,
        prompts: prompts.split("\n").map((line) => line.trim()).filter(Boolean)
      });
      setResult(response.result);
    } finally {
      setBusy(false);
    }
  }

  const metrics = (result?.metrics ?? {}) as Record<string, number>;

  return (
    <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
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
        </label>
        <button className="tool-button" onClick={run} disabled={busy || !patchId || !modelLoaded} title={modelLoaded ? "Run evaluation" : "Load the selected model before evaluating"}>
          <BarChart3 size={16} />
          {busy ? "Evaluating" : "Evaluate"}
        </button>
      </section>
      <section className="panel grid gap-4">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {Object.entries(metrics).map(([key, value]) => (
            <div key={key} className="rounded-md border border-line bg-panel p-3">
              <div className="label">{key.replace(/_/g, " ")}</div>
              <div className="mt-1 text-2xl font-semibold">{value.toFixed(3)}</div>
            </div>
          ))}
        </div>
        <pre className="min-h-72 overflow-auto rounded-md border border-line bg-panel p-4 text-xs leading-5">
          {result ? JSON.stringify(result, null, 2) : "Evaluation metrics and sample generations appear here."}
        </pre>
      </section>
    </div>
  );
}
