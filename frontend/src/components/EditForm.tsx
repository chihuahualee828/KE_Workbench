"use client";

import { BarChart3, FlaskConical, Layers, LocateFixed, Save, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { EditJob, Patch, SupportedModel } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { PatchHistory } from "./PatchHistory";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export function EditForm({ models, patches, onPatchesChanged }: { models: SupportedModel[]; patches: Patch[]; onPatchesChanged?: () => void | Promise<void> }) {
  const [tab, setTab] = usePersistentState<"rome" | "memit" | "patches">("kew:edit:tab", "rome");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [job, setJob] = useState<EditJob | null>(null);

  return (
    <div className="grid gap-4">
      <div className="inline-flex w-fit rounded-md border border-line bg-white p-1">
        {[
          ["rome", "ROME"],
          ["memit", "MEMIT"],
          ["patches", "Patches"]
        ].map(([id, label]) => (
          <button
            key={id}
            className={`rounded px-3 py-2 text-sm ${tab === id ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`}
            onClick={() => setTab(id as "rome" | "memit" | "patches")}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "rome" ? <RomeForm models={models} result={result} job={job} setResult={setResult} setJob={setJob} onPatchesChanged={onPatchesChanged} /> : null}
      {tab === "memit" ? <MemitForm models={models} result={result} job={job} setResult={setResult} setJob={setJob} onPatchesChanged={onPatchesChanged} /> : null}
      {tab === "patches" ? (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,0.85fr)]">
          <PatchHistory patches={patches} onPatchesChanged={onPatchesChanged} />
          <ResultPanel result={result} job={job} />
        </div>
      ) : null}
    </div>
  );
}

async function watchEditJob(initial: EditJob, onUpdate: (job: EditJob) => void) {
  let current = initial;
  onUpdate(current);
  while (current.status === "running") {
    current = await waitEditJob(current.job_id, current.logs.length, 60);
    onUpdate(current);
  }
  if (current.status === "failed") {
    throw new Error(current.logs[current.logs.length - 1] ?? "Edit job failed");
  }
  return current;
}

async function waitEditJob(jobId: string, afterLogCount: number, timeoutSeconds: number) {
  const response = await fetch(`${API_BASE}/api/edit/jobs/${jobId}/wait?after_log_count=${afterLogCount}&timeout=${timeoutSeconds}`, {
    cache: "no-store"
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      detail = response.statusText;
    }
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json() as Promise<EditJob>;
}

function RomeForm({
  models,
  result,
  job,
  setResult,
  setJob,
  onPatchesChanged
}: {
  models: SupportedModel[];
  result: Record<string, unknown> | null;
  job: EditJob | null;
  setResult: (result: Record<string, unknown> | null) => void;
  setJob: (job: EditJob | null) => void;
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const romeModels = useMemo(() => models.filter((model) => model.supports_rome), [models]);
  const [modelId, setModelId] = usePersistentState("kew:edit:rome:modelId", romeModels[0]?.id ?? "gpt2-xl");
  const [subject, setSubject] = usePersistentState("kew:edit:rome:subject", "The Space Needle");
  const [relationPrompt, setRelationPrompt] = usePersistentState("kew:edit:rome:relationPrompt", "{} is located in the city of");
  const [newObject, setNewObject] = usePersistentState("kew:edit:rome:newObject", "Paris");
  const [groundTruth, setGroundTruth] = usePersistentState("kew:edit:rome:groundTruth", "Seattle");
  const [targetLayer, setTargetLayer] = usePersistentState("kew:edit:rome:targetLayer", "");
  const [evaluationPrompts, setEvaluationPrompts] = usePersistentState("kew:edit:rome:evaluationPrompts", "The landmark called the Space Needle can be found in");
  const [sequentialEdit, setSequentialEdit] = usePersistentState("kew:edit:rome:sequentialEdit", false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [evaluating, setEvaluating] = useState(false);

  useEffect(() => {
    if (romeModels.length === 0) return;
    if (!romeModels.some((model) => model.id === modelId)) {
      setModelId(romeModels[0].id);
    }
  }, [modelId, romeModels, setModelId]);

  async function run() {
    setBusy(true);
    setError("");
    setResult(null);
    setJob(null);
    try {
      const response = await api.romeEdit({
        model_id: modelId,
        subject,
        relation_prompt: relationPrompt,
        new_object: newObject,
        ground_truth: groundTruth || null,
        target_layer: targetLayer.trim() === "" ? null : Number(targetLayer),
        evaluation_prompts: evaluationPrompts.split("\n").map((line) => line.trim()).filter(Boolean),
        sequential_edit: sequentialEdit
      });
      const completed = await watchEditJob(response, setJob);
      setResult(completed.result ?? null);
      await onPatchesChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function evaluatePrompts() {
    const prompts = evaluationPrompts.split("\n").map((line) => line.trim()).filter(Boolean);
    if (!prompts.length) return;
    setEvaluating(true);
    setError("");
    try {
      const generations = [];
      for (const prompt of prompts) {
        const response = await api.generate({
          model_id: modelId,
          prompt,
          max_new_tokens: 32,
          decoding_strategy: "greedy"
        });
        generations.push({ prompt, text: response.text });
      }
      setResult({ generations, evaluation_only: true, model_id: modelId });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setEvaluating(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={romeModels.length ? romeModels : models} value={modelId} onChange={setModelId} />
        <TextInput label="Subject" value={subject} onChange={setSubject} />
        <TextInput label="Relation Prompt" value={relationPrompt} onChange={setRelationPrompt} />
        <TextInput label="New Object" value={newObject} onChange={setNewObject} />
        <TextInput label="Ground Truth" value={groundTruth} onChange={setGroundTruth} />
        <label className="grid gap-2">
          <span className="label">Layer Override</span>
          <div className="grid grid-cols-[minmax(0,1fr)_40px] gap-2">
            <input className="control" type="number" value={targetLayer} min={0} placeholder="Use method default" onChange={(event) => setTargetLayer(event.target.value)} />
            <button
              type="button"
              className="ghost-button h-10 min-h-10 px-0"
              onClick={() => setTargetLayer("")}
              title="Use causal tracing for GPT-2 models or the EasyEdit hparams default otherwise"
            >
              <LocateFixed size={16} />
            </button>
          </div>
        </label>
        <div className="grid gap-2">
          <span className="flex items-center justify-between gap-2">
            <span className="label">Evaluation Prompts</span>
            <button
              type="button"
              className="ghost-button min-h-8 px-2 py-1 text-xs"
              onClick={evaluatePrompts}
              disabled={busy || evaluating || !evaluationPrompts.trim()}
              title="Generate from the current active model without applying a new edit"
            >
              <BarChart3 size={14} />
              {evaluating ? "Evaluating" : "Evaluate"}
            </button>
          </span>
          <textarea className="control min-h-28" value={evaluationPrompts} onChange={(event) => setEvaluationPrompts(event.target.value)} />
        </div>
        <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
          <span>Sequential edit</span>
          <input type="checkbox" checked={sequentialEdit} onChange={(event) => setSequentialEdit(event.target.checked)} />
        </label>
        <button className="tool-button w-full" onClick={run} disabled={busy} title="Apply ROME edit">
          <FlaskConical size={16} />
          {busy ? "Editing" : "Apply ROME"}
        </button>
        <ExportActiveModelControl modelId={modelId} disabled={busy || evaluating} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} />
    </div>
  );
}

function MemitForm({
  models,
  result,
  job,
  setResult,
  setJob,
  onPatchesChanged
}: {
  models: SupportedModel[];
  result: Record<string, unknown> | null;
  job: EditJob | null;
  setResult: (result: Record<string, unknown> | null) => void;
  setJob: (job: EditJob | null) => void;
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const memitModels = useMemo(() => models.filter((model) => model.supports_memit), [models]);
  const [modelId, setModelId] = usePersistentState("kew:edit:memit:modelId", memitModels[0]?.id ?? "gpt2-xl");
  const [edits, setEdits] = usePersistentState("kew:edit:memit:edits", '[{"subject":"Michael Jordan","relation_prompt":"{} plays the sport of","ground_truth":"basketball","new_object":"baseball"},{"subject":"The Space Needle","relation_prompt":"{} is located in the city of","ground_truth":"Seattle","new_object":"Paris"}]');
  const [layers, setLayers] = usePersistentState("kew:edit:memit:layers", "3,4,5,6,7,8");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (memitModels.length === 0) return;
    if (!memitModels.some((model) => model.id === modelId)) {
      setModelId(memitModels[0].id);
    }
  }, [memitModels, modelId, setModelId]);

  async function run() {
    setBusy(true);
    setError("");
    setResult(null);
    setJob(null);
    try {
      const response = await api.memitEdit({
        model_id: modelId,
        edits: JSON.parse(edits),
        layers: layers.split(",").map((item) => Number(item.trim())).filter((item) => Number.isFinite(item)),
        sequential_edit: false
      });
      const completed = await watchEditJob(response, setJob);
      setResult(completed.result ?? null);
      await onPatchesChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function uploadEditsJson(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) {
        throw new Error("MEMIT edits JSON must be an array of edit objects.");
      }
      setEdits(JSON.stringify(parsed, null, 2));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={memitModels.length ? memitModels : models} value={modelId} onChange={setModelId} />
        <div className="grid gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <label className="label" htmlFor="memit-edits-json">
              Batch Edits JSON
            </label>
            <label className="ghost-button min-h-8 shrink-0 cursor-pointer px-2 py-1 text-xs" title="Upload MEMIT edits JSON">
              <Upload size={14} />
              Upload JSON
              <input
                className="sr-only"
                type="file"
                accept="application/json,.json"
                onChange={(event) => {
                  void uploadEditsJson(event.target.files?.[0] ?? null);
                  event.target.value = "";
                }}
              />
            </label>
          </div>
          <textarea id="memit-edits-json" className="control min-h-48 font-mono text-xs" value={edits} onChange={(event) => setEdits(event.target.value)} />
        </div>
        <TextInput label="Layers" value={layers} onChange={setLayers} />
        <button className="tool-button w-full" onClick={run} disabled={busy} title="Apply MEMIT edit">
          <Layers size={16} />
          {busy ? "Editing" : "Apply MEMIT"}
        </button>
        <ExportActiveModelControl modelId={modelId} disabled={busy} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} />
    </div>
  );
}

function ExportActiveModelControl({ modelId, disabled = false }: { modelId: string; disabled?: boolean }) {
  const [exportDirectory] = usePersistentState("kew:settings:modelExportPath", "");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const targetPath = exportDirectory.trim() ? buildExportPath(exportDirectory, modelId, "model") : "";

  async function saveModel() {
    setMessage("");
    setError("");
    setBusy(true);
    try {
      const result = await api.exportModel({ model_id: modelId, output_path: targetPath });
      setMessage(`Saved model to ${result.output_path}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-3 border-t border-line pt-3">
      <div className="label">Save Current Model</div>
      <div className="rounded-md border border-line bg-panel px-3 py-2">
        <div className="truncate font-mono text-xs text-ink/60" title={targetPath || "Set default save directory in Settings"}>
          {targetPath || "Set default save directory in Settings"}
        </div>
      </div>
      <div className="flex justify-end">
        <button
          type="button"
          className="ghost-button min-h-9 px-3 py-1"
          onClick={saveModel}
          disabled={disabled || busy || !targetPath}
          title="Save current active model weights and lineage"
        >
          <Save size={15} />
          {busy ? "Saving" : "Save"}
        </button>
      </div>
      {message ? <div className="text-xs text-cyan">{message}</div> : null}
      {error ? <div className="text-xs text-rust">{error}</div> : null}
    </div>
  );
}

function buildExportPath(directory: string, id: string, mode: "model" | "patch") {
  const base = directory.trim().replace(/\/+$/, "");
  const safeId = id.replace(/[^a-zA-Z0-9._-]/g, "_");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${base}/${safeId}-${mode}-${timestamp}`;
}

function TextInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="grid gap-2">
      <span className="label">{label}</span>
      <input className="control" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function ResultPanel({ result, job }: { result: Record<string, unknown> | null; job: EditJob | null }) {
  const logsRef = useRef<HTMLPreElement | null>(null);
  const generations = useMemo(() => parseGenerations(result), [result]);

  useEffect(() => {
    if (!logsRef.current) return;
    logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [job?.logs]);

  return (
    <section className="panel">
      <div className="mb-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="label">Edit Logs</span>
          <span className="font-mono text-xs text-ink/50">{job?.status ?? "idle"}</span>
        </div>
        <pre ref={logsRef} className="max-h-72 min-h-40 overflow-auto rounded-md border border-line bg-ink p-3 text-xs leading-5 text-white">
          {job?.logs.length ? job.logs.join("\n") : "No edit job is running."}
        </pre>
      </div>
      <div className="mb-2 flex items-center gap-2">
        <Save size={16} />
        <span className="label">Generations</span>
      </div>
      <div className="min-h-96 rounded-md border border-line bg-panel p-4">
        {generations.length ? (
          <div className="grid gap-3">
            {generations.map((generation, index) => (
              <div key={`${generation.prompt}-${index}`} className="grid gap-2 border-b border-line pb-3 last:border-b-0 last:pb-0">
                <div className="font-mono text-xs text-ink/50">{generation.prompt}</div>
                <div className="whitespace-pre-wrap text-sm leading-6 text-ink">{generation.text}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-sm text-ink/60">Evaluation generations appear here after an edit finishes.</div>
        )}
      </div>
    </section>
  );
}

function parseGenerations(result: Record<string, unknown> | null) {
  const value = result?.generations;
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const record = item as Record<string, unknown>;
      return {
        prompt: typeof record.prompt === "string" ? record.prompt : "",
        text: typeof record.text === "string" ? record.text : ""
      };
    })
    .filter((item): item is { prompt: string; text: string } => Boolean(item?.text));
}

