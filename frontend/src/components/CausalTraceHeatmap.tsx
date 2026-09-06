"use client";

import { Activity, ArrowRight, ChevronDown, Crosshair, Info, Loader2, Network, SlidersHorizontal, WandSparkles } from "lucide-react";
import { useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { ModelRuntimeStatus, SupportedModel, TraceCell } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { PromptEditor } from "./PromptEditor";
import { TransformerGrid } from "./TransformerGrid";

const ROME_SUBJECT_PATTERN = /(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*/;
type TraceResult = {
  cells: TraceCell[];
  tokens: string[];
  clean: number;
  corrupted: number;
  configKey: string;
  modelName: string;
  prompt: string;
  target: string;
  samples: number;
};

export function CausalTraceHeatmap({ models, runtime = [] }: { models: SupportedModel[]; runtime?: ModelRuntimeStatus[] }) {
  const [modelId, setModelId] = usePersistentState("kew:trace:modelId", models[0]?.id ?? "gpt2-xl");
  const [prompt, setPrompt] = usePersistentState("kew:trace:prompt", "The Space Needle is located in the city of");
  const [subject, setSubject] = usePersistentState("kew:trace:subject", "The Space Needle");
  const [target, setTarget] = usePersistentState("kew:trace:target", "Seattle");
  const [samples, setSamples] = usePersistentState("kew:trace:samples", 10);
  const [seed, setSeed] = usePersistentState("kew:trace:seed", 1);
  const [result, setResult] = useState<TraceResult | null>(null);
  const [selected, setSelected] = useState<TraceCell | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [subjectMessage, setSubjectMessage] = useState("");
  const traceModels = useMemo(() => models.filter((model) => model.supports_causal_trace), [models]);
  const modelLoaded = runtime.some((status) => status.model_id === modelId && status.loaded);
  const selectedModel = traceModels.find((model) => model.id === modelId);
  const configKey = JSON.stringify({ modelId, prompt, subject, target, samples, seed });
  const stale = result !== null && result.configKey !== configKey;
  const layers = useMemo(() => Array.from(new Set(result?.cells.map((cell) => cell.layer) ?? [])).sort((a, b) => a - b), [result]);
  const validation = !prompt.trim() ? "Enter a prompt to trace." : !subject.trim() ? "Enter the subject to corrupt." : !prompt.includes(subject.trim()) ? "The subject must appear in your prompt." : !target.trim() ? "Enter the expected answer." : "";

  useLoadedModelSelection(traceModels, runtime, setModelId);

  function detectSubject() {
    const match = prompt.match(ROME_SUBJECT_PATTERN);
    if (match) {
      setSubject(match[0].trim());
      setSubjectMessage("Suggested from capitalization. Check that this is the subject you want to trace.");
    } else setSubjectMessage("Could not detect a subject. Enter the words from your prompt manually.");
  }

  async function run() {
    if (!modelLoaded || busy || validation) return;
    setError("");
    setBusy(true);
    try {
      const response = await api.causalTrace({
        model_id: modelId, prompt, subject: subject.trim(), target,
        component: "residual", samples, seed,
        corruption: { type: "gaussian_noise", noise_level: 3.0 }
      });
      setResult({ cells: response.result.heatmap, tokens: response.result.tokens, clean: response.result.clean_probability, corrupted: response.result.corrupted_probability, configKey, modelName: selectedModel?.name ?? modelId, prompt, target, samples });
      setSelected(response.result.heatmap[0] ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The trace failed. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const recovery = selected && result && Math.abs(result.clean - result.corrupted) > 0.000001 ? (selected.restored_probability - result.corrupted) / (result.clean - result.corrupted) : null;

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[300px_minmax(0,1fr)]">
      <section className="panel">
        <div className="panel-heading"><h2 className="flex items-center gap-2 text-sm font-semibold"><SlidersHorizontal size={16} className="text-cyan" />Trace configuration</h2></div>
        <fieldset disabled={busy} className="grid min-w-0 gap-4">
          <ModelSelector models={traceModels} value={modelId} onChange={setModelId} />
          <PromptEditor value={prompt} onChange={setPrompt} rows={4} />
          <label className="grid gap-2"><span className="label">Subject to corrupt</span><input className="control" value={subject} onChange={(event) => { setSubject(event.target.value); setSubjectMessage(""); }} /><span className="text-[11px] leading-5 text-ink/50">The entity in your prompt whose knowledge you want to locate.</span></label>
          <button className="ghost-button -mt-2 min-h-8 justify-self-start px-2 py-1 text-xs" type="button" onClick={detectSubject}><WandSparkles size={13} />Suggest subject</button>
          {subjectMessage ? <p role="status" className="text-xs leading-5 text-ink/60">{subjectMessage}</p> : null}
          <label className="grid gap-2"><span className="label">Expected answer</span><input className="control" value={target} onChange={(event) => setTarget(event.target.value)} placeholder="e.g. Seattle" /></label>
          <div className="rounded-lg border border-line bg-panel p-3"><div className="label">Restored component</div><div className="mt-1 text-sm font-medium">Residual stream</div><p className="mt-1 text-[11px] leading-5 text-ink/55">Restores the hidden state at the output of a Transformer block.</p></div>
          <details className="rounded-lg border border-line px-3 py-2.5"><summary className="flex items-center justify-between text-xs font-medium text-ink/65">Corruption settings<ChevronDown size={14} /></summary><div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
            <label className="grid gap-2"><span className="label">Corrupted samples</span><input className="control" type="number" min={1} max={32} value={samples} onChange={(event) => setSamples(Math.max(1, Math.min(32, Math.floor(Number(event.target.value) || 1))))} /></label>
            <label className="grid gap-2"><span className="label">Noise seed</span><input className="control" type="number" min={0} max={2147483647} value={seed} onChange={(event) => setSeed(Math.max(0, Math.min(2147483647, Math.floor(Number(event.target.value) || 0))))} /></label>
            <p className="text-[11px] leading-5 text-ink/55 sm:col-span-2 xl:col-span-1">Gaussian noise, scale 3. More samples reduce noise and use more GPU memory.</p>
          </div></details>
          {validation ? <p className="text-xs text-rust">{validation}</p> : !modelLoaded ? <p className="text-xs leading-5 text-ink/55">Load a supported model above to enable tracing.</p> : null}
          <button className="tool-button" onClick={run} disabled={busy || !modelLoaded || Boolean(validation)} title={modelLoaded ? "Evaluate all token and layer positions" : "Load the selected model before tracing"}>{busy ? <Loader2 size={16} className="animate-spin" /> : <Activity size={16} />}{busy ? "Tracing activations…" : "Run causal trace"}{!busy ? <ArrowRight size={14} className="ml-auto" /> : null}</button>
        </fieldset>
        {error ? <div className="inline-error mt-4" role="alert">{error}</div> : null}
        <p className="mt-3 text-[11px] leading-5 text-ink/45">A full trace evaluates every token and layer. Larger models and longer prompts take more time.</p>
      </section>

      <div className="grid min-w-0 gap-4">
        <section className="panel" aria-busy={busy}>
          <div className="panel-heading"><div><h2 className="flex items-center gap-2 text-sm font-semibold"><Network size={17} className="text-cyan" />Activation map</h2><p className="mt-1.5 text-xs text-ink/50">Token positions × shared Transformer layers</p></div><span className="rounded-full bg-cyan/5 px-2.5 py-1 text-[10px] font-medium text-cyan">Analysis only</span></div>
          {stale ? <div role="status" className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800">Configuration changed. This map shows the previous run; run the trace again to update it.</div> : null}
          {busy ? <div role="status" className="mb-4 flex items-center gap-3 rounded-lg bg-cyan/5 p-3 text-xs leading-5 text-cyan"><Loader2 size={16} className="shrink-0 animate-spin" />Running clean, corrupted, and restored forward passes. Results appear when the full trace finishes.</div> : null}
          {result ? <>
            <div className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink/50"><span className="font-medium text-ink/70">{result.modelName}</span><span>{result.tokens.length} tokens</span><span>{layers.length} layers</span><span>{result.samples} samples</span></div>
            <div className="mb-5 grid gap-3 sm:grid-cols-3"><Metric label="Clean probability" value={percent(result.clean)} /><Metric label="Corrupted probability" value={percent(result.corrupted)} /><Metric label="Restored probability" value={selected ? percent(selected.restored_probability) : "—"} accent /></div>
            <TransformerGrid cells={result.cells} tokens={result.tokens} selected={selected} onSelect={setSelected} />
            <p className="mt-2 text-[11px] leading-5 text-ink/45">Colors show signed probability change from the corrupted baseline, scaled to this trace.</p>
          </> : <div className="trace-canvas flex min-h-[420px] flex-col items-center justify-center rounded-lg border border-dashed border-line px-6 text-center"><span className="icon-tile mb-5 h-14 w-14 border border-cyan/10 bg-white shadow-sm"><Network size={28} strokeWidth={1.3} /></span><h3 className="text-base font-semibold">See where knowledge takes shape</h3><p className="mt-2 max-w-sm text-xs leading-6 text-ink/55">Set a prompt, choose its subject, and run a trace. Explore how restoring each hidden state changes the model’s answer.</p><div className="mt-6 flex flex-wrap justify-center gap-3 text-[10px] font-medium text-ink/45"><span>01 Configure</span><ArrowRight size={12} /><span>02 Run trace</span><ArrowRight size={12} /><span>03 Explore cells</span></div>{selectedModel ? <p className="mt-6 text-[11px] text-cyan">{selectedModel.name} · {selectedModel.n_layers} Transformer layers</p> : null}</div>}
        </section>

        <section className="panel">
          <div className="panel-heading"><h2 className="flex items-center gap-2 text-sm font-semibold"><Crosshair size={16} className="text-cyan" />Selected intervention</h2>{selected ? <span className="text-[10px] font-medium text-ink/45">EVALUATED CELL</span> : null}</div>
          {selected && result ? <>
            <div className="grid gap-4 sm:grid-cols-2"><label className="grid gap-2"><span className="label">Token position</span><select aria-label="Token position" className="control font-mono" value={selected.token_index} onChange={(event) => { const cell = result.cells.find((cell) => cell.layer === selected.layer && cell.token_index === Number(event.target.value)); if (cell) setSelected(cell); }}>{result.tokens.map((token, index) => <option key={index} value={index}>{index} · {JSON.stringify(token)}</option>)}</select></label><label className="grid gap-2"><span className="label">Transformer layer</span><select aria-label="Transformer layer" className="control" value={selected.layer} onChange={(event) => { const cell = result.cells.find((cell) => cell.layer === Number(event.target.value) && cell.token_index === selected.token_index); if (cell) setSelected(cell); }}>{layers.map((layer) => <option key={layer} value={layer}>Layer {layer}</option>)}</select></label></div>
            <div className="mt-4 grid gap-3 sm:grid-cols-2"><Metric label="Indirect effect" value={`${selected.indirect_effect >= 0 ? "+" : ""}${(selected.indirect_effect * 100).toFixed(2)} pp`} /><Metric label="Recovery of clean signal" value={recovery === null ? "Not defined" : percent(recovery)} /></div>
            <p className="mt-3 text-[11px] leading-5 text-ink/55">Indirect effect is restored minus corrupted probability. Recovery divides that change by the clean–corrupted gap and can fall outside 0–100%. Probabilities measure the first token of “{result.target}”.</p>
            <details className="mt-4 text-xs"><summary className="text-ink/55">View run prompt and raw cell data</summary><p className="mt-3 break-words leading-5 text-ink/65">{result.prompt}</p><pre className="mt-2 overflow-x-auto rounded-lg bg-panel p-3 text-[11px]">{JSON.stringify(selected, null, 2)}</pre></details>
          </> : <p className="text-xs leading-6 text-ink/55">After running a trace, select a cell or drag its marker to compare restored probabilities at different token and layer positions.</p>}
        </section>
        <div className="flex items-start gap-3 px-1 text-[11px] leading-5 text-ink/50"><Info size={15} className="mt-0.5 shrink-0 text-cyan" /><p>Each cell restores the clean activation from that same token and layer. Moving the marker inspects evaluated results without running inference. Causal tracing is a temporary intervention; ROME weight edits are managed separately in Knowledge edits.</p></div>
      </div>
    </div>
  );
}

function percent(value: number) { return `${(value * 100).toFixed(2)}%`; }
function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div className={`min-w-0 rounded-lg border p-3 ${accent ? "border-cyan/15 bg-cyan/5" : "border-line bg-panel/60"}`}><div className="text-[10px] font-medium text-ink/55">{label}</div><div className={`mt-2 text-xl font-semibold tracking-tight tabular-nums ${accent ? "text-cyan" : "text-ink"}`}>{value}</div></div>;
}
