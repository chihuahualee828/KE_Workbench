"use client";

import { Play } from "lucide-react";
import { useState } from "react";

import { api } from "@/lib/api";
import type { Patch, SupportedModel } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { PromptEditor } from "./PromptEditor";

export function GenerationPanel({ models, patches = [] }: { models: SupportedModel[]; patches?: Patch[] }) {
  const [modelId, setModelId] = usePersistentState("kew:generation:modelId", models[0]?.id ?? "gpt2");
  const [prompt, setPrompt] = usePersistentState("kew:generation:prompt", "The Space Needle is located in the city of");
  const [patchId, setPatchId] = usePersistentState("kew:generation:patchId", "");
  const [maxNewTokens, setMaxNewTokens] = usePersistentState("kew:generation:maxNewTokens", 32);
  const [decodingStrategy, setDecodingStrategy] = usePersistentState<"greedy" | "sampling" | "beam">("kew:generation:decodingStrategy", "greedy");
  const [temperature, setTemperature] = usePersistentState("kew:generation:temperature", 0.7);
  const [topP, setTopP] = usePersistentState("kew:generation:topP", 0.95);
  const [topK, setTopK] = usePersistentState("kew:generation:topK", 50);
  const [numBeams, setNumBeams] = usePersistentState("kew:generation:numBeams", 4);
  const [output, setOutput] = useState("");
  const [busy, setBusy] = useState(false);
  const [runtimeRefreshKey, setRuntimeRefreshKey] = useState(0);

  async function run() {
    setBusy(true);
    try {
      const result = await api.generate({
        model_id: modelId,
        prompt,
        max_new_tokens: clampNumber(maxNewTokens, 1, 128),
        decoding_strategy: decodingStrategy,
        temperature: clampNumber(temperature, 0.01, 5),
        top_p: clampNumber(topP, 0.01, 1),
        top_k: clampNumber(topK, 0, 500),
        num_beams: clampNumber(numBeams, 1, 16),
        patch_id: patchId || null
      });
      setOutput(result.text);
    } finally {
      setRuntimeRefreshKey((value) => value + 1);
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={models} value={modelId} onChange={setModelId} refreshKey={runtimeRefreshKey} />
        <PromptEditor value={prompt} onChange={setPrompt} />
        <div className="grid gap-3 md:grid-cols-2">
          <NumberInput label="Max New Tokens" value={maxNewTokens} onChange={setMaxNewTokens} min={1} max={128} />
          <label className="grid gap-2">
            <span className="label">Decoding</span>
            <select className="control" value={decodingStrategy} onChange={(event) => setDecodingStrategy(event.target.value as "greedy" | "sampling" | "beam")}>
              <option value="greedy">Greedy</option>
              <option value="sampling">Sampling</option>
              <option value="beam">Beam Search</option>
            </select>
          </label>
          {decodingStrategy === "sampling" ? (
            <>
              <NumberInput label="Temperature" value={temperature} onChange={setTemperature} min={0.01} max={5} step={0.05} />
              <NumberInput label="Top P" value={topP} onChange={setTopP} min={0.01} max={1} step={0.01} />
              <NumberInput label="Top K" value={topK} onChange={setTopK} min={0} max={500} />
            </>
          ) : null}
          {decodingStrategy === "beam" ? (
            <NumberInput label="Beams" value={numBeams} onChange={setNumBeams} min={1} max={16} />
          ) : null}
        </div>
        <label className="grid gap-2">
          <span className="label">Patch</span>
          <select className="control" value={patchId} onChange={(event) => setPatchId(event.target.value)}>
            <option value="">Current active model state</option>
            {patches.map((patch) => (
              <option key={patch.patch_id} value={patch.patch_id}>
                {patch.active ? "Active" : "Saved"} {patch.method.toUpperCase()} {patch.patch_id}
              </option>
            ))}
          </select>
        </label>
        <button className="tool-button" onClick={run} disabled={busy || !prompt.trim()} title="Generate completion">
          <Play size={16} />
          {busy ? "Running" : "Generate"}
        </button>
      </section>
      <section className="panel">
        <div className="label mb-2">Completion</div>
        <pre className="min-h-56 whitespace-pre-wrap rounded-md border border-line bg-panel p-4 text-sm leading-6 text-ink">
          {output || "Generation output appears here."}
        </pre>
      </section>
    </div>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
}) {
  return (
    <label className="grid gap-2">
      <span className="label">{label}</span>
      <input
        className="control"
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function clampNumber(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}
