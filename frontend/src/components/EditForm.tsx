"use client";

import { BarChart3, Download, FlaskConical, Layers, Loader2, LocateFixed, Plus, Save, Trash2, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { EditJob, ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { PatchHistory } from "./PatchHistory";
import { useWorkbenchSession } from "./WorkbenchSession";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const EMPTY_LAYER_VALUES: Record<string, string> = {};

type MemitEditDraft = {
  subject: string;
  relation_prompt: string;
  ground_truth: string;
  new_object: string;
  rephrase_prompts?: string[];
};

type MemitEditableField = "subject" | "relation_prompt" | "ground_truth" | "new_object";

export function EditForm({ models, patches, runtime = [], onPatchesChanged }: { models: SupportedModel[]; patches: Patch[]; runtime?: ModelRuntimeStatus[]; onPatchesChanged?: () => void | Promise<void> }) {
  const [tab, setTab] = usePersistentState<"rome" | "memit" | "patches">("kew:edit:tab", "rome");
  const { edit, setEdit } = useWorkbenchSession();
  const setResult = (result: Record<string, unknown> | null) => setEdit((current) => ({ ...current, result }));
  const setJob = (job: EditJob | null) => setEdit((current) => ({ ...current, job }));

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
      {tab === "rome" ? <RomeForm models={models} runtime={runtime} result={edit.result} job={edit.job} setResult={setResult} setJob={setJob} onPatchesChanged={onPatchesChanged} /> : null}
      {tab === "memit" ? <MemitForm models={models} runtime={runtime} result={edit.result} job={edit.job} setResult={setResult} setJob={setJob} onPatchesChanged={onPatchesChanged} /> : null}
      {tab === "patches" ? (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,0.85fr)]">
          <PatchHistory patches={patches} runtime={runtime} onPatchesChanged={onPatchesChanged} />
          <div className="min-w-0">
            <ResultPanel result={edit.result} job={edit.job} />
          </div>
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
  runtime,
  result,
  job,
  setResult,
  setJob,
  onPatchesChanged
}: {
  models: SupportedModel[];
  runtime: ModelRuntimeStatus[];
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
  const [targetLayers, setTargetLayers] = usePersistentState<Record<string, string>>("kew:edit:rome:targetLayers", EMPTY_LAYER_VALUES);
  const [easyEditDefaultLayers, setEasyEditDefaultLayers] = useState<Record<string, number>>({});
  const [evaluationPrompts, setEvaluationPrompts] = usePersistentState("kew:edit:rome:evaluationPrompts", "The landmark called the Space Needle can be found in");
  const [sequentialEdit, setSequentialEdit] = usePersistentState("kew:edit:rome:sequentialEdit", false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [locatingLayer, setLocatingLayer] = useState(false);
  const [layerNotice, setLayerNotice] = useState("");
  const selectedModel = useMemo(() => romeModels.find((model) => model.id === modelId), [modelId, romeModels]);
  const selectedModelId = selectedModel?.id;
  const maxLayer = Math.max(0, (selectedModel?.n_layers ?? 1) - 1);
  const targetLayer = targetLayers[modelId] ?? "";
  const targetLayerNumber = parseLayer(targetLayer, maxLayer);
  const easyEditDefaultLayer = easyEditDefaultLayers[modelId];
  const modelLoaded = runtime.some((runtimeStatus) => runtimeStatus.model_id === modelId && runtimeStatus.loaded);

  useLoadedModelSelection(romeModels, runtime, setModelId);

  useEffect(() => {
    if (romeModels.length === 0) return;
    if (!romeModels.some((model) => model.id === modelId)) {
      setModelId(romeModels[0].id);
    }
  }, [modelId, romeModels, setModelId]);

  useEffect(() => {
    let active = true;
    if (!selectedModelId) return;
    setLayerNotice("");
    api.editDefaultLayers({ model_id: selectedModelId, method: "rome" })
      .then((result) => {
        const defaultLayer = result.layers[0];
        if (!active || !Number.isInteger(defaultLayer) || defaultLayer < 0 || defaultLayer > maxLayer) return;
        setEasyEditDefaultLayers((current) => ({ ...current, [selectedModelId]: defaultLayer }));
        setTargetLayers((current) => {
          const storedLayer = parseLayer(current[selectedModelId] ?? "", maxLayer);
          if (storedLayer !== null) return current;
          return { ...current, [selectedModelId]: String(defaultLayer) };
        });
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      active = false;
    };
  }, [maxLayer, selectedModelId, setTargetLayers]);

  async function run() {
    if (!modelLoaded) {
      setError("Load the selected model before editing.");
      return;
    }
    if (targetLayerNumber === null) {
      setError(`Enter a layer from 0 to ${maxLayer}.`);
      return;
    }
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
        target_layer: targetLayerNumber,
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

  async function locateLayer() {
    if (!selectedModel?.supports_causal_trace || !subject.trim() || !relationPrompt.trim() || !groundTruth.trim()) return;
    setLocatingLayer(true);
    setError("");
    setLayerNotice("");
    try {
      const response = await api.causalTrace({
        model_id: modelId,
        prompt: buildEditPrompt(subject, relationPrompt),
        subject,
        target: groundTruth,
        component: "mlp",
        corruption: { type: "gaussian_noise", noise_level: 3.0 }
      });
      const locatedLayer = response.result.recommended_layer;
      if (locatedLayer < 0 || locatedLayer > maxLayer) {
        throw new Error(`Causal trace selected an invalid layer: ${locatedLayer}.`);
      }
      setTargetLayers((current) => ({ ...current, [modelId]: String(locatedLayer) }));
      setLayerNotice(`Causal trace selected layer ${locatedLayer}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLocatingLayer(false);
    }
  }

  async function evaluatePrompts() {
    const prompts = evaluationPrompts.split("\n").map((line) => line.trim()).filter(Boolean);
    if (!modelLoaded || !prompts.length) return;
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
        <ModelSelector models={romeModels.length ? romeModels : models} value={modelId} onChange={setModelId} onRuntimeUnloaded={onPatchesChanged} />
        <TextInput label="Subject" value={subject} onChange={setSubject} />
        <TextInput label="Relation Prompt" value={relationPrompt} onChange={setRelationPrompt} />
        <TextInput label="New Object" value={newObject} onChange={setNewObject} />
        <TextInput label="Ground Truth" value={groundTruth} onChange={setGroundTruth} />
        <label className="grid gap-2">
          <span className="label">Edit Layer</span>
          <div className="grid grid-cols-[minmax(0,1fr)_40px] gap-2">
            <input
              className="control"
              type="number"
              value={targetLayer}
              min={0}
              max={maxLayer}
              onChange={(event) => {
                const value = event.target.value;
                if (value === "" || parseLayer(value, maxLayer) !== null) {
                  setTargetLayers((current) => ({ ...current, [modelId]: value }));
                  setLayerNotice("");
                }
              }}
            />
            <button
              type="button"
              className="ghost-button h-10 min-h-10 px-0"
              onClick={locateLayer}
              disabled={busy || locatingLayer || !modelLoaded || !selectedModel?.supports_causal_trace || !subject.trim() || !relationPrompt.trim() || !groundTruth.trim()}
              title={locatingLayer ? "Running causal trace" : !modelLoaded ? "Load the selected model before causal tracing" : selectedModel?.supports_causal_trace ? "Run causal tracing on the current fact and fill the selected edit layer" : "Causal tracing is not available for this model"}
            >
              {locatingLayer ? <Loader2 className="animate-spin" size={16} /> : <LocateFixed size={16} />}
            </button>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 font-mono text-xs text-ink/50">
            <span>Valid layers: 0-{maxLayer}</span>
            <span>EasyEdit default: {easyEditDefaultLayer ?? "Loading..."}</span>
          </div>
          {layerNotice ? <span className="text-xs text-cyan">{layerNotice}</span> : null}
        </label>
        <div className="grid gap-2">
          <span className="flex items-center justify-between gap-2">
            <span className="label">Evaluation Prompts</span>
            <button
              type="button"
              className="ghost-button min-h-8 px-2 py-1 text-xs"
              onClick={evaluatePrompts}
              disabled={busy || evaluating || !evaluationPrompts.trim() || !modelLoaded}
              title={modelLoaded ? "Generate from the current active model without applying a new edit" : "Load the selected model before evaluating"}
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
        <button className="tool-button w-full" onClick={run} disabled={busy || locatingLayer || targetLayerNumber === null || !modelLoaded} title={modelLoaded ? "Apply ROME edit at the selected layer" : "Load the selected model before editing"}>
          <FlaskConical size={16} />
          {busy ? "Editing" : "Apply ROME"}
        </button>
        <ExportActiveModelControl modelId={modelId} modelLoaded={modelLoaded} disabled={busy || evaluating} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} />
    </div>
  );
}

function MemitForm({
  models,
  runtime,
  result,
  job,
  setResult,
  setJob,
  onPatchesChanged
}: {
  models: SupportedModel[];
  runtime: ModelRuntimeStatus[];
  result: Record<string, unknown> | null;
  job: EditJob | null;
  setResult: (result: Record<string, unknown> | null) => void;
  setJob: (job: EditJob | null) => void;
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const memitModels = useMemo(() => models.filter((model) => model.supports_memit), [models]);
  const [modelId, setModelId] = usePersistentState("kew:edit:memit:modelId", memitModels[0]?.id ?? "gpt2-xl");
  const [edits, setEdits] = usePersistentState("kew:edit:memit:edits", '[{"subject":"Michael Jordan","relation_prompt":"{} plays the sport of","ground_truth":"basketball","new_object":"baseball"},{"subject":"The Space Needle","relation_prompt":"{} is located in the city of","ground_truth":"Seattle","new_object":"Paris"}]');
  const [inputMode, setInputMode] = usePersistentState<"form" | "json">("kew:edit:memit:inputMode", "form");
  const [layersByModel, setLayersByModel] = usePersistentState<Record<string, string>>("kew:edit:memit:layersByModel", EMPTY_LAYER_VALUES);
  const [easyEditDefaultLayers, setEasyEditDefaultLayers] = useState<Record<string, number[]>>({});
  const [evaluationPrompts, setEvaluationPrompts] = usePersistentState("kew:edit:memit:evaluationPrompts", "Michael Jordan plays the sport of\nThe Space Needle is located in the city of");
  const [sequentialEdit, setSequentialEdit] = usePersistentState("kew:edit:memit:sequentialEdit", false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const selectedModel = useMemo(() => memitModels.find((model) => model.id === modelId), [memitModels, modelId]);
  const selectedModelId = selectedModel?.id;
  const maxLayer = Math.max(0, (selectedModel?.n_layers ?? 1) - 1);
  const layers = layersByModel[modelId] ?? "";
  const selectedLayers = parseMemitLayers(layers, maxLayer);
  const easyEditDefaultLayerValues = easyEditDefaultLayers[modelId];
  const modelLoaded = runtime.some((runtimeStatus) => runtimeStatus.model_id === modelId && runtimeStatus.loaded);
  const editDrafts = useMemo(() => parseMemitEditDrafts(edits), [edits]);

  useLoadedModelSelection(memitModels, runtime, setModelId);

  useEffect(() => {
    if (memitModels.length === 0) return;
    if (!memitModels.some((model) => model.id === modelId)) {
      setModelId(memitModels[0].id);
    }
  }, [memitModels, modelId, setModelId]);

  useEffect(() => {
    let active = true;
    if (!selectedModelId) return;
    api.editDefaultLayers({ model_id: selectedModelId, method: "memit" })
      .then((result) => {
        const defaults = result.layers.filter((layer) => Number.isInteger(layer) && layer >= 0 && layer <= maxLayer);
        if (!active || defaults.length === 0) return;
        const defaultValue = defaults.join(",");
        setEasyEditDefaultLayers((current) => ({ ...current, [selectedModelId]: defaults }));
        setLayersByModel((current) => {
          if (parseMemitLayers(current[selectedModelId] ?? "", maxLayer)) return current;
          return { ...current, [selectedModelId]: defaultValue };
        });
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      active = false;
    };
  }, [maxLayer, selectedModelId, setLayersByModel]);

  async function run() {
    if (!modelLoaded || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    setJob(null);
    try {
      const parsedEdits = parseMemitEditDrafts(edits);
      if (!parsedEdits?.length) {
        throw new Error("Add at least one MEMIT fact before editing.");
      }
      const invalidEdit = parsedEdits.findIndex((edit) => !edit.subject.trim() || !edit.relation_prompt.trim() || !edit.new_object.trim());
      if (invalidEdit >= 0) {
        throw new Error(`Fact ${invalidEdit + 1} needs a subject, relation prompt, and new object.`);
      }
      if (!selectedLayers?.length) {
        throw new Error(`Enter one or more layers from 0 to ${maxLayer}.`);
      }
      const response = await api.memitEdit({
        model_id: modelId,
        edits: parsedEdits.map((edit) => ({
          ...edit,
          ground_truth: edit.ground_truth.trim() || null
        })),
        layers: selectedLayers,
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
      setInputMode("form");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function updateEditDraft(index: number, field: MemitEditableField, value: string) {
    if (!editDrafts) return;
    setEdits(JSON.stringify(editDrafts.map((edit, editIndex) => editIndex === index ? { ...edit, [field]: value } : edit), null, 2));
  }

  function addEditDraft() {
    const nextEdits = editDrafts ?? [];
    setEdits(JSON.stringify([...nextEdits, emptyMemitEditDraft()], null, 2));
  }

  function removeEditDraft(index: number) {
    if (!editDrafts) return;
    setEdits(JSON.stringify(editDrafts.filter((_, editIndex) => editIndex !== index), null, 2));
  }

  function showFormEditor() {
    if (!editDrafts) {
      setError("Fix the JSON before switching to the form editor.");
      return;
    }
    setError("");
    setInputMode("form");
  }

  function formatEditsJson() {
    try {
      const parsed = JSON.parse(edits);
      if (!Array.isArray(parsed)) throw new Error("MEMIT edits JSON must be an array of edit objects.");
      setEdits(JSON.stringify(parsed, null, 2));
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function evaluatePrompts() {
    const prompts = evaluationPrompts.split("\n").map((line) => line.trim()).filter(Boolean);
    if (!modelLoaded || !prompts.length || busy || evaluating) return;
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
        <ModelSelector models={memitModels.length ? memitModels : models} value={modelId} onChange={setModelId} onRuntimeUnloaded={onPatchesChanged} />
        <div className="grid gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="label">Batch Edits</span>
            <div className="flex items-center gap-2">
              <div className="inline-flex rounded-md border border-line bg-white p-1">
                <button className={`rounded px-2 py-1 text-xs ${inputMode === "form" ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} onClick={showFormEditor} type="button">Form</button>
                <button className={`rounded px-2 py-1 text-xs ${inputMode === "json" ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} onClick={() => setInputMode("json")} type="button">JSON</button>
              </div>
              <label className="ghost-button grid size-8 shrink-0 cursor-pointer place-items-center p-0" title="Upload MEMIT edits JSON">
                <Upload size={14} />
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
          </div>
          {inputMode === "form" && editDrafts ? (
            <div className="grid max-h-96 gap-3 overflow-y-auto pr-1">
              {editDrafts.map((edit, index) => (
                <section key={index} className="grid gap-3 border border-line bg-panel p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-medium">Fact {index + 1}</span>
                    <button aria-label={`Remove fact ${index + 1}`} className="ghost-button grid size-8 place-items-center p-0" disabled={editDrafts.length === 1} onClick={() => removeEditDraft(index)} title="Remove fact" type="button">
                      <Trash2 size={15} />
                    </button>
                  </div>
                  <MemitFactInput label="Subject" value={edit.subject} onChange={(value) => updateEditDraft(index, "subject", value)} />
                  <MemitFactInput label="Relation Prompt" value={edit.relation_prompt} onChange={(value) => updateEditDraft(index, "relation_prompt", value)} />
                  <MemitFactInput label="New Object" value={edit.new_object} onChange={(value) => updateEditDraft(index, "new_object", value)} />
                  <MemitFactInput label="Ground Truth" value={edit.ground_truth} onChange={(value) => updateEditDraft(index, "ground_truth", value)} />
                </section>
              ))}
              <button className="ghost-button min-h-9 justify-self-start px-2 py-1 text-xs" onClick={addEditDraft} type="button">
                <Plus size={14} />
                Add fact
              </button>
            </div>
          ) : (
            <>
              <div className="flex flex-wrap justify-end gap-2">
                <button className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={formatEditsJson} type="button">Format JSON</button>
              </div>
              <textarea id="memit-edits-json" className="control min-h-64 font-mono text-xs" value={edits} onChange={(event) => setEdits(event.target.value)} />
            </>
          )}
        </div>
        <label className="grid gap-2">
          <span className="label">Edit Layers</span>
          <input
            className="control"
            inputMode="numeric"
            value={layers}
            onChange={(event) => {
              const value = event.target.value;
              if (isEditableMemitLayerInput(value, maxLayer)) {
                setLayersByModel((current) => ({ ...current, [modelId]: value }));
              }
            }}
          />
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 font-mono text-xs text-ink/50">
            <span>Valid layers: 0-{maxLayer}</span>
            <span>EasyEdit default: {easyEditDefaultLayerValues?.join(",") ?? "Loading..."}</span>
          </div>
        </label>
        <div className="grid gap-2">
          <span className="flex items-center justify-between gap-2">
            <span className="label">Evaluation Prompts</span>
            <button
              type="button"
              className="ghost-button min-h-8 px-2 py-1 text-xs"
              onClick={evaluatePrompts}
              disabled={busy || evaluating || !evaluationPrompts.trim() || !modelLoaded}
              title={modelLoaded ? "Generate from the current active model without applying a new edit" : "Load the selected model before evaluating"}
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
        <button className="tool-button w-full" onClick={run} disabled={busy || evaluating || !modelLoaded || !selectedLayers?.length} title={modelLoaded ? "Apply MEMIT edit at the selected layers" : "Load the selected model before editing"}>
          <Layers size={16} />
          {busy ? "Editing" : "Apply MEMIT"}
        </button>
        <ExportActiveModelControl modelId={modelId} modelLoaded={modelLoaded} disabled={busy || evaluating} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} />
    </div>
  );
}

function ExportActiveModelControl({ modelId, modelLoaded, disabled = false }: { modelId: string; modelLoaded: boolean; disabled?: boolean }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function saveModel() {
    setMessage("");
    setError("");
    const suggestedName = `${safeFilename(modelId)}-edited.zip`;
    const pickerWindow = window as SaveFilePickerWindow;

    setBusy(true);
    try {
      if (!pickerWindow.showSaveFilePicker) {
        const response = await api.downloadModelArchive({ model_id: modelId });
        const archive = await response.blob();
        const url = URL.createObjectURL(archive);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = suggestedName;
        anchor.click();
        URL.revokeObjectURL(url);
        setMessage("Started model archive download.");
        return;
      }

      const handle = await pickerWindow.showSaveFilePicker({
        suggestedName,
        types: [{ description: "ZIP archive", accept: { "application/zip": [".zip"] } }]
      });
      const writable = await handle.createWritable();
      const response = await api.downloadModelArchive({ model_id: modelId });
      if (!response.body) {
        await writable.abort();
        throw new Error("The browser did not provide a download stream.");
      }
      await response.body.pipeTo(writable);
      setMessage(`Saved ${suggestedName}.`);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-3 border-t border-line pt-3">
      <div className="label">Export Current Model</div>
      <div className="rounded-md border border-line bg-panel px-3 py-2">
        <div className="text-xs text-ink/60">Downloads a standard model ZIP with the current edited weights.</div>
      </div>
      <div className="flex justify-end">
        <button
          type="button"
          className="ghost-button min-h-9 px-3 py-1"
          onClick={saveModel}
          disabled={disabled || busy || !modelLoaded}
          title={modelLoaded ? "Choose a location and download the current edited model as a ZIP archive" : "Load the selected model before exporting"}
        >
          <Download size={15} />
          {busy ? "Preparing" : "Save Model ZIP"}
        </button>
      </div>
      {message ? <div className="text-xs text-cyan">{message}</div> : null}
      {error ? <div className="text-xs text-rust">{error}</div> : null}
    </div>
  );
}

function parseLayer(value: string, maxLayer: number) {
  if (!/^\d+$/.test(value)) return null;
  const layer = Number(value);
  return Number.isSafeInteger(layer) && layer >= 0 && layer <= maxLayer ? layer : null;
}

function buildEditPrompt(subject: string, relationPrompt: string) {
  return relationPrompt.includes("{}") ? relationPrompt.replace("{}", subject) : `${subject} ${relationPrompt}`.trim();
}

type ModelArchiveWritable = WritableStream<Uint8Array> & {
  abort: () => Promise<void>;
};

type SaveFilePickerWindow = Window & {
  showSaveFilePicker?: (options: {
    suggestedName: string;
    types: Array<{ description: string; accept: Record<string, string[]> }>;
  }) => Promise<{
    createWritable: () => Promise<ModelArchiveWritable>;
  }>;
};

function safeFilename(value: string) {
  return value.replace(/[^a-zA-Z0-9._-]/g, "_") || "edited-model";
}

function emptyMemitEditDraft(): MemitEditDraft {
  return {
    subject: "",
    relation_prompt: "{} is",
    ground_truth: "",
    new_object: ""
  };
}

function parseMemitEditDrafts(value: string): MemitEditDraft[] | null {
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) return null;
    return parsed.map((item) => {
      if (!item || typeof item !== "object") return emptyMemitEditDraft();
      const edit = item as Record<string, unknown>;
      return {
        subject: typeof edit.subject === "string" ? edit.subject : "",
        relation_prompt: typeof edit.relation_prompt === "string" ? edit.relation_prompt : "",
        ground_truth: typeof edit.ground_truth === "string" ? edit.ground_truth : "",
        new_object: typeof edit.new_object === "string" ? edit.new_object : "",
        rephrase_prompts: Array.isArray(edit.rephrase_prompts)
          ? edit.rephrase_prompts.filter((prompt): prompt is string => typeof prompt === "string")
          : undefined
      };
    });
  } catch {
    return null;
  }
}

function MemitFactInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="grid gap-1">
      <span className="label">{label}</span>
      <input className="control" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
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
