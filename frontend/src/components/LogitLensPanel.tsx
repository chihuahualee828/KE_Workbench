"use client";

import { Search } from "lucide-react";
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
    if (!modelLoaded || busy) return;
    setBusy(true);
    try {
      const result = await api.logitLens({
        model_id: modelId,
        prompt,
        target_token: targetToken,
        top_k: 6
      });
      setLayers(result.layers);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={lensModels} value={modelId} onChange={setModelId} />
        <PromptEditor value={prompt} onChange={setPrompt} rows={4} />
        <label className="grid gap-2">
          <span className="label">Target Token</span>
          <input className="control" value={targetToken} onChange={(event) => setTargetToken(event.target.value)} />
        </label>
        <button className="tool-button" onClick={run} disabled={busy || !modelLoaded} title={modelLoaded ? "Run logit lens" : "Load the selected model before running logit lens"}>
          <Search size={16} />
          {busy ? "Running" : "Run Lens"}
        </button>
      </section>
      <section className="panel grid gap-4">
        <div className="h-64 rounded-md border border-line bg-panel p-3">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chart}>
              <XAxis dataKey="layer" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Line type="monotone" dataKey="probability" stroke="#1c7784" strokeWidth={2} dot={false} />
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
                  <td className="py-2 pr-4">{layer.target_probability ?? "n/a"}</td>
                  <td className="py-2 pr-4">{layer.top_tokens.map(([token]) => token).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
