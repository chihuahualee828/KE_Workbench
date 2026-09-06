"use client";

import { Layers3, Loader2, Search, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "@/lib/api";
import type { LogitLensLayer, ModelRuntimeStatus, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { PromptEditor } from "./PromptEditor";

export function LogitLensPanel({ models, runtime = [] }: { models: SupportedModel[]; runtime?: ModelRuntimeStatus[] }) {
  const [modelId, setModelId] = usePersistentState("kew:lens:modelId", models[0]?.id ?? "gpt2-xl");
  const [prompt, setPrompt] = usePersistentState("kew:lens:prompt", "The Space Needle is located in the city of");
  const [targetToken, setTargetToken] = usePersistentState("kew:lens:targetToken", " Seattle");
  const [layers, setLayers] = useState<LogitLensLayer[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const lensModels = useMemo(() => models.filter((model) => model.supports_logit_lens), [models]);
  const modelLoaded = runtime.some((status) => status.model_id === modelId && status.loaded);

  useLoadedModelSelection(lensModels, runtime, setModelId);

  const chart = useMemo(
    () =>
      layers.map((layer) => ({
        layer: layer.layer,
        probability: layer.target_probability ?? 0
      })),
    [layers]
  );

  async function run() {
    if (!modelLoaded || busy || !prompt.trim()) return;
    setError("");
    setBusy(true);
    try {
      const result = await api.logitLens({
        model_id: modelId,
        prompt,
        target_token: targetToken,
        top_k: 6
      });
      setLayers(result.layers);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Logit lens failed. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[300px_minmax(0,1fr)]">
      <section className="panel">
        <div className="panel-heading"><h2 className="flex items-center gap-2 text-sm font-semibold"><SlidersHorizontal size={16} className="text-cyan" />Lens configuration</h2></div>
        <fieldset disabled={busy} className="grid min-w-0 gap-4">
        <ModelSelector models={lensModels} value={modelId} onChange={setModelId} />
        <PromptEditor value={prompt} onChange={setPrompt} rows={4} />
        <label className="grid gap-2">
          <span className="label">Target Token</span>
          <input className="control" value={targetToken} onChange={(event) => setTargetToken(event.target.value)} />
        </label>
        {!modelLoaded ? <p className="text-xs leading-5 text-ink/55">Load the selected model to inspect its predictions.</p> : null}
        <button className="tool-button" onClick={run} disabled={busy || !modelLoaded || !prompt.trim()} title={modelLoaded ? "Run logit lens" : "Load the selected model before running logit lens"}>
          {busy ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
          {busy ? "Inspecting layers…" : "Run logit lens"}
        </button>
        </fieldset>
        {error ? <div role="alert" className="inline-error mt-4">{error}</div> : null}
      </section>
      <section className="panel grid gap-4" aria-busy={busy}>
        <div><h2 className="flex items-center gap-2 text-sm font-semibold"><Layers3 size={17} className="text-cyan" />Predictions through the layers</h2><p className="mt-2 text-xs text-ink/50">Target probability at each Transformer layer</p></div>
        {busy ? <div role="status" className="flex items-center gap-2 text-xs text-cyan"><Loader2 size={14} className="animate-spin" />Inspecting model predictions…</div> : null}
        {layers.length ? <><div className="h-64 rounded-lg border border-line bg-panel/50 p-3">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chart}>
              <XAxis dataKey="layer" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 1]} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip />
              <Line type="monotone" dataKey="probability" stroke="#5b57d6" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line text-left">
                <th className="py-2 pr-4">Layer</th>
                <th className="py-2 pr-4">Rank</th>
                <th className="py-2 pr-4">Probability</th>
                <th className="py-2 pr-4">Top Tokens</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((layer) => (
                <tr key={layer.layer} className="border-b border-line/70">
                  <td className="py-2 pr-4">{layer.layer}</td>
                  <td className="py-2 pr-4">{layer.target_rank ?? "n/a"}</td>
                  <td className="py-2 pr-4 tabular-nums">{layer.target_probability == null ? "n/a" : `${(layer.target_probability * 100).toFixed(2)}%`}</td>
                  <td className="py-2 pr-4">{layer.top_tokens.map(([token]) => token).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div></> : <div className="empty-state min-h-[360px] rounded-lg border border-dashed border-line bg-panel/50"><span className="icon-tile bg-white shadow-sm"><Layers3 size={22} strokeWidth={1.5} /></span><h3 className="text-sm font-medium">Watch an answer emerge</h3><p className="max-w-sm text-xs leading-6 text-ink/55">Run the lens to compare target probability, rank, and top predictions as your prompt passes through the model.</p></div>}
      </section>
    </div>
  );
}
