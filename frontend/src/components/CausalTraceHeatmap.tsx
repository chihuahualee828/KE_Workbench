"use client";

import { Activity } from "lucide-react";
import { Fragment } from "react";
import { useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { ModelRuntimeStatus, SupportedModel, TraceCell } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { PromptEditor } from "./PromptEditor";

export function CausalTraceHeatmap({ models, runtime = [] }: { models: SupportedModel[]; runtime?: ModelRuntimeStatus[] }) {
  const [modelId, setModelId] = usePersistentState("kew:trace:modelId", models[0]?.id ?? "gpt2-xl");
  const [prompt, setPrompt] = usePersistentState("kew:trace:prompt", "The Space Needle is located in the city of");
  const [subject, setSubject] = usePersistentState("kew:trace:subject", "The Space Needle");
  const [target, setTarget] = usePersistentState("kew:trace:target", "Seattle");
  const [component, setComponent] = usePersistentState("kew:trace:component", "mlp");
  const [cells, setCells] = useState<TraceCell[]>([]);
  const [tokens, setTokens] = useState<string[]>([]);
  const [summary, setSummary] = useState<{ clean_probability: number; corrupted_probability: number } | null>(null);
  const [selected, setSelected] = useState<TraceCell | null>(null);
  const [busy, setBusy] = useState(false);
  const traceModels = useMemo(() => models.filter((model) => model.supports_causal_trace), [models]);
  const modelLoaded = runtime.some((status) => status.model_id === modelId && status.loaded);

  useLoadedModelSelection(traceModels, runtime, setModelId);

  const layers = useMemo(() => Array.from(new Set(cells.map((cell) => cell.layer))), [cells]);
  const maxEffect = Math.max(0.01, ...cells.map((cell) => Math.abs(cell.indirect_effect)));

  async function run() {
    if (!modelLoaded || busy) return;
    setBusy(true);
    try {
      const response = await api.causalTrace({
        model_id: modelId,
        prompt,
        subject,
        target,
        component,
        corruption: { type: "gaussian_noise", noise_level: 3.0 }
      });
      setCells(response.result.heatmap);
      setTokens(response.result.tokens);
      setSummary(response.result);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={traceModels} value={modelId} onChange={setModelId} />
        <PromptEditor value={prompt} onChange={setPrompt} rows={4} />
        <label className="grid gap-2">
          <span className="label">Subject</span>
          <input className="control" value={subject} onChange={(event) => setSubject(event.target.value)} />
        </label>
        <label className="grid gap-2">
          <span className="label">Expected Answer</span>
          <input className="control" value={target} onChange={(event) => setTarget(event.target.value)} />
        </label>
        <label className="grid gap-2">
          <span className="label">Component</span>
          <select className="control" value={component} onChange={(event) => setComponent(event.target.value)}>
            <option value="residual">Hidden State</option>
            <option value="mlp">MLP</option>
            <option value="attention">Attention</option>
          </select>
        </label>
        <button className="tool-button" onClick={run} disabled={busy || !modelLoaded} title={modelLoaded ? "Run causal trace" : "Load the selected model before causal tracing"}>
          <Activity size={16} />
          {busy ? "Running" : "Run Trace"}
        </button>
      </section>
      <section className="panel grid gap-4">
        {summary ? (
          <div className="grid gap-3 sm:grid-cols-3">
            <Metric label="Clean" value={summary.clean_probability} />
            <Metric label="Corrupted" value={summary.corrupted_probability} />
            <Metric label="Selected IE" value={selected?.indirect_effect ?? 0} />
          </div>
        ) : null}
        <div className="overflow-auto rounded-md border border-line bg-panel p-3">
          <div
            className="grid gap-1"
            style={{
              gridTemplateColumns: `72px repeat(${Math.max(tokens.length, 1)}, minmax(44px, 1fr))`
            }}
          >
            <div />
            {tokens.map((token, index) => (
              <div key={`${token}-${index}`} className="truncate text-center text-xs text-ink/70" title={token}>
                {token}
              </div>
            ))}
            {layers.map((layer) => (
              <Fragment key={`layer-${layer}`}>
                <div className="flex h-8 items-center text-xs font-medium text-ink/70">
                  L{layer}
                </div>
                {tokens.map((_, tokenIndex) => {
                  const cell = cells.find((item) => item.layer === layer && item.token_index === tokenIndex);
                  const intensity = cell ? Math.abs(cell.indirect_effect) / maxEffect : 0;
                  return (
                    <button
                      key={`${layer}-${tokenIndex}`}
                      className="h-8 rounded-sm border border-white"
                      style={{ backgroundColor: `rgba(28, 119, 132, ${0.08 + intensity * 0.82})` }}
                      onClick={() => cell && setSelected(cell)}
                      title={cell ? `Layer ${cell.layer}, token ${cell.token}, IE ${cell.indirect_effect}` : ""}
                    />
                  );
                })}
              </Fragment>
            ))}
          </div>
        </div>
        <pre className="min-h-24 whitespace-pre-wrap rounded-md border border-line bg-panel p-3 text-sm">
          {selected ? JSON.stringify(selected, null, 2) : "Click a heatmap cell to inspect restored probability."}
        </pre>
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-line bg-panel p-3">
      <div className="label">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value.toFixed(3)}</div>
    </div>
  );
}
