"use client";

import { Check, Copy, Loader2, MessageSquare, Play, SlidersHorizontal, Sparkles } from "lucide-react";
import { useState } from "react";

import { api } from "@/lib/api";
import type { ModelRuntimeStatus, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { PromptEditor } from "./PromptEditor";
import { useWorkbenchSession } from "./WorkbenchSession";

export function GenerationPanel({ models, runtime = [], onPatchesChanged, sessionScope = "generation" }: { models: SupportedModel[]; runtime?: ModelRuntimeStatus[]; onPatchesChanged?: () => void | Promise<void>; sessionScope?: string }) {
  const [modelId, setModelId] = usePersistentState("kew:generation:modelId", models[0]?.id ?? "gpt2-xl");
  const [prompt, setPrompt] = usePersistentState("kew:generation:prompt", "The Space Needle is located in the city of");
  const [maxNewTokens, setMaxNewTokens] = usePersistentState("kew:generation:maxNewTokens", 32);
  const [decodingStrategy, setDecodingStrategy] = usePersistentState<"greedy" | "sampling" | "beam">("kew:generation:decodingStrategy", "greedy");
  const [temperature, setTemperature] = usePersistentState("kew:generation:temperature", 0.7);
  const [topP, setTopP] = usePersistentState("kew:generation:topP", 0.95);
  const [topK, setTopK] = usePersistentState("kew:generation:topK", 50);
  const [numBeams, setNumBeams] = usePersistentState("kew:generation:numBeams", 4);
  const { generationOutput, setGenerationOutput } = useWorkbenchSession();
  const output = generationOutput(sessionScope);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [runtimeRefreshKey, setRuntimeRefreshKey] = useState(0);
  const modelLoaded = runtime.some((status) => status.model_id === modelId && status.loaded);

  useLoadedModelSelection(models, runtime, setModelId);

  async function run() {
    if (!modelLoaded || !prompt.trim() || busy) return;
    setError("");
    setCopied(false);
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
        num_beams: clampNumber(numBeams, 1, 16)
      });
      setGenerationOutput(sessionScope, result.text);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed. Please try again.");
    } finally {
      setRuntimeRefreshKey((value) => value + 1);
      setBusy(false);
    }
  }

  async function copyOutput() {
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
    } catch {
      setError("Could not copy the completion. Select the text and copy it manually.");
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <section className="panel">
        <div className="panel-heading"><h2 className="flex items-center gap-2 text-sm font-semibold"><SlidersHorizontal size={16} className="text-cyan" />Prompt configuration</h2><span className="text-[10px] text-ink/40">INPUT</span></div>
        <fieldset disabled={busy} className="grid min-w-0 gap-4" onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); void run(); } }}>
        <ModelSelector models={models} value={modelId} onChange={setModelId} onRuntimeUnloaded={onPatchesChanged} refreshKey={runtimeRefreshKey} />
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
        {!modelLoaded ? <p className="text-xs leading-5 text-ink/55">Load the selected model to enable generation.</p> : null}
        <button className="tool-button" onClick={run} disabled={busy || !prompt.trim() || !modelLoaded} title={modelLoaded ? "Generate completion (Ctrl or ⌘ + Enter)" : "Load the selected model before generating"}>
          {busy ? <Loader2 size={16} className="animate-spin" /> : <Play size={15} fill="currentColor" />}
          {busy ? "Generating completion…" : "Generate completion"}
        </button>
        </fieldset>
        {error ? <div className="inline-error mt-4" role="alert">{error}</div> : null}
      </section>
      <section className="panel flex flex-col" aria-busy={busy}>
        <div className="panel-heading"><h2 className="flex items-center gap-2 text-sm font-semibold"><MessageSquare size={16} className="text-cyan" />Model completion</h2>{output ? <button type="button" className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={copyOutput}>{copied ? <Check size={13} /> : <Copy size={13} />}{copied ? "Copied" : "Copy"}</button> : <span className="text-[10px] text-ink/40">OUTPUT</span>}</div>
        <div className="flex flex-1 flex-col rounded-lg border border-line bg-panel/70" aria-live="polite">
          {busy ? <div className="empty-state flex-1" role="status"><Loader2 size={25} className="animate-spin text-cyan" /><p className="text-sm font-medium">Your model is thinking</p><p className="max-w-xs text-xs leading-5 text-ink/55">The completion will appear here when inference finishes.</p></div> : output ? <pre className="min-h-56 whitespace-pre-wrap p-5 font-sans text-sm leading-7 text-ink">{output}</pre> : <div className="empty-state flex-1"><span className="icon-tile h-12 w-12 bg-white shadow-sm"><Sparkles size={22} strokeWidth={1.5} /></span><p className="mt-1 text-sm font-medium">A prompt is a good place to start</p><p className="max-w-xs text-xs leading-5 text-ink/55">Choose a model, write a prompt, and generate a completion to explore its knowledge.</p><span className="mt-2 rounded border border-line bg-white px-2 py-1 text-[10px] text-ink/45">Ctrl / ⌘ + Enter to generate</span></div>}
        </div>
        <p className="mt-4 text-[11px] leading-5 text-ink/45">Generations use the selected model and its currently active patches.</p>
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
