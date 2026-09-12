"use client";

import { Database, Download, FlaskConical, Layers, Loader2, LocateFixed, Plus, Save, Trash2, Upload } from "lucide-react";
import Link from "next/link";
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { announceExperimentCanceled, announceExperimentStarted, listenForExperimentCanceled } from "@/lib/experimentEvents";
import type { EditJob, ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { EvaluationProgressIndicator } from "./EvaluationProgressIndicator";
import { PatchHistory } from "./PatchHistory";
import { SecondMomentCacheDialog } from "./SecondMomentCacheDialog";
import { useWorkbenchSession } from "./WorkbenchSession";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const EMPTY_LAYER_VALUES: Record<string, string> = {};

type FactEditDraft = {
  subject: string;
  relation_prompt: string;
  ground_truth: string;
  new_object: string;
  rephrase_prompts?: string[];
};

type FactEditableField = "subject" | "relation_prompt" | "ground_truth" | "new_object";
type ReliabilityTestProgress = { completed: number; total: number } | null;

export function EditForm({ models, patches, runtime = [], onPatchesChanged }: { models: SupportedModel[]; patches: Patch[]; runtime?: ModelRuntimeStatus[]; onPatchesChanged?: () => void | Promise<void> }) {
  const [tab, setTab] = usePersistentState<"rome" | "r_rome" | "kn" | "memit" | "patches">("kew:edit:tab", "rome");
  const { edit, setEdit } = useWorkbenchSession();
  const setResult = (result: Record<string, unknown> | null) => setEdit((current) => ({ ...current, result }));
  const setJob = (job: EditJob | null) => setEdit((current) => ({ ...current, job }));

  return (
    <div className="grid gap-4">
      <div className="inline-flex max-w-full flex-wrap gap-1 rounded-xl border border-line bg-white p-1.5">
        {[
          ["rome", "ROME"],
          ["r_rome", "r-ROME"],
          ["kn", "KN"],
          ["memit", "MEMIT"],
          ["patches", "Patches"]
        ].map(([id, label]) => (
          <button
            key={id}
            aria-pressed={tab === id}
            className={`rounded-lg px-4 py-2 text-xs font-medium transition ${tab === id ? "bg-cyan/10 text-cyan" : "text-ink/55 hover:bg-panel hover:text-ink"}`}
            onClick={() => setTab(id as "rome" | "r_rome" | "kn" | "memit" | "patches")}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "rome" ? <RomeForm models={models} runtime={runtime} result={edit.result} job={edit.job} setResult={setResult} setJob={setJob} onPatchesChanged={onPatchesChanged} /> : null}
      {tab === "r_rome" ? <RomeForm method="r_rome" models={models} runtime={runtime} result={edit.result} job={edit.job} setResult={setResult} setJob={setJob} onPatchesChanged={onPatchesChanged} /> : null}
      {tab === "kn" ? <KnForm models={models} runtime={runtime} result={edit.result} job={edit.job} setResult={setResult} setJob={setJob} onPatchesChanged={onPatchesChanged} /> : null}
      {tab === "memit" ? <MemitForm models={models} runtime={runtime} result={edit.result} job={edit.job} setResult={setResult} setJob={setJob} onPatchesChanged={onPatchesChanged} /> : null}
      {tab === "patches" ? (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,0.85fr)]">
          <PatchHistory patches={patches} models={models} runtime={runtime} onPatchesChanged={onPatchesChanged} />
          <div className="min-w-0">
            <ResultPanel result={edit.result} job={edit.job} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function KnForm({
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
  const knModels = useMemo(() => models.filter((model) => model.supports_kn), [models]);
  const [modelId, setModelId] = usePersistentState("kew:edit:kn:modelId", knModels[0]?.id ?? "gpt2-xl");
  const [subject, setSubject] = usePersistentState("kew:edit:kn:subject", "The Space Needle");
  const [relationPrompt, setRelationPrompt] = usePersistentState("kew:edit:kn:relationPrompt", "{} is located in the city of");
  const [newObject, setNewObject] = usePersistentState("kew:edit:kn:newObject", "Paris");
  const [groundTruth, setGroundTruth] = usePersistentState("kew:edit:kn:groundTruth", "Seattle");
  const [sequentialEdit, setSequentialEdit] = usePersistentState("kew:edit:kn:sequentialEdit", false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [testingReliability, setTestingReliability] = useState(false);
  const [reliabilityTestProgress, setReliabilityTestProgress] = useState<ReliabilityTestProgress>(null);
  const [reliabilityResult, setReliabilityResult] = useState<Record<string, unknown> | null>(null);
  const reliabilityTestCancellation = useReliabilityTestCancellation();
  const modelLoaded = runtime.some((runtimeStatus) => runtimeStatus.model_id === modelId && runtimeStatus.loaded);
  const evaluationPrompts = useMemo(
    () => buildFactEvaluationPrompts([{ subject, relation_prompt: relationPrompt }]),
    [relationPrompt, subject]
  );

  useLoadedModelSelection(knModels, runtime, setModelId);

  useEffect(() => {
    if (knModels.length && !knModels.some((model) => model.id === modelId)) {
      setModelId(knModels[0].id);
    }
  }, [knModels, modelId, setModelId]);

  useEffect(() => {
    setReliabilityResult(null);
  }, [evaluationPrompts, modelId, newObject]);

  async function run() {
    if (!modelLoaded || busy) return;
    if (!subject.trim() || !relationPrompt.trim() || !newObject.trim() || !groundTruth.trim()) {
      setError("KN needs a subject, relation prompt, new object, and original object.");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    setJob(null);
    try {
      const response = await api.knEdit({
        model_id: modelId,
        subject,
        relation_prompt: relationPrompt,
        new_object: newObject,
        ground_truth: groundTruth,
        evaluation_prompts: evaluationPrompts,
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

  async function testReliability() {
    if (!modelLoaded || !evaluationPrompts.length || busy || testingReliability) return;
    setTestingReliability(true);
    setError("");
    setReliabilityResult(null);
    setReliabilityTestProgress(null);
    reliabilityTestCancellation.reset();
    try {
      const response = await api.easyEditReliability({
        model_id: modelId,
        method: "kn",
        edits: [{ subject, relation_prompt: relationPrompt, new_object: newObject, ground_truth: groundTruth || null }],
        include_generations: true
      });
      reliabilityTestCancellation.setJobId(response.job_id);
      await watchReliabilityTestJob(response, (job) => {
        if (!job.result) return;
        setReliabilityResult(job.result);
        setReliabilityTestProgress(reliabilityTestJobProgress(job));
        setResult({
          reliability: job.result,
          reliability_test_job_id: job.job_id,
          reliability_test_item_count: job.result_item_count ?? 0,
          reliability_test_page_size: job.result_page_size ?? null,
          evaluation_only: true,
          model_id: modelId
        });
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setTestingReliability(false);
      reliabilityTestCancellation.reset();
    }
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={knModels} value={modelId} onChange={setModelId} onRuntimeUnloaded={onPatchesChanged} />
        <div className="rounded-md border border-line bg-panel p-3 text-sm leading-6 text-ink/70">
          Knowledge Neurons finds fact-specific MLP neurons, then updates their output vectors. The original object is required to identify the neurons.
        </div>
        <TextInput label="Subject" value={subject} onChange={setSubject} />
        <TextInput label="Relation Prompt" value={relationPrompt} onChange={setRelationPrompt} />
        <TextInput label="New Object" value={newObject} onChange={setNewObject} />
        <TextInput label="Original Object" value={groundTruth} onChange={setGroundTruth} />
        <DerivedEvaluationPrompts
          prompts={evaluationPrompts}
          onTest={testReliability}
          onCancel={() => void reliabilityTestCancellation.cancel().catch((error) => setError(error instanceof Error ? error.message : String(error)))}
          reliabilityResult={reliabilityResult}
          testProgress={reliabilityTestProgress}
          testing={testingReliability}
          testCancelable={Boolean(reliabilityTestCancellation.jobId)}
          testCanceling={reliabilityTestCancellation.canceling}
          testDisabled={busy || !modelLoaded || !evaluationPrompts.length}
        />
        <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
          <span>Sequential edit</span>
          <input type="checkbox" checked={sequentialEdit} onChange={(event) => setSequentialEdit(event.target.checked)} />
        </label>
        <button className="tool-button w-full" onClick={run} disabled={busy || !modelLoaded} title={modelLoaded ? "Identify knowledge neurons and apply the KN edit" : "Load the selected model before editing"}>
          <FlaskConical size={16} />
          {busy ? "Editing" : "Apply KN"}
        </button>
        <ExportActiveModelControl modelId={modelId} modelLoaded={modelLoaded} disabled={busy} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} />
    </div>
  );
}

async function watchEditJob(initial: EditJob, onUpdate: (job: EditJob) => void) {
  announceExperimentStarted(initial.experiment_id);
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
  method = "rome",
  models,
  runtime,
  result,
  job,
  setResult,
  setJob,
  onPatchesChanged
}: {
  method?: "rome" | "r_rome";
  models: SupportedModel[];
  runtime: ModelRuntimeStatus[];
  result: Record<string, unknown> | null;
  job: EditJob | null;
  setResult: (result: Record<string, unknown> | null) => void;
  setJob: (job: EditJob | null) => void;
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const isRRome = method === "r_rome";
  const methodLabel = isRRome ? "R-ROME" : "ROME";
  const supportsSeries = !isRRome;
  const romeModels = useMemo(() => models.filter((model) => isRRome ? model.supports_r_rome : model.supports_rome), [isRRome, models]);
  const [modelId, setModelId] = usePersistentState(`kew:edit:${method}:modelId`, romeModels[0]?.id ?? "gpt2-xl");
  const [subject, setSubject] = usePersistentState(`kew:edit:${method}:subject`, "The Space Needle");
  const [relationPrompt, setRelationPrompt] = usePersistentState(`kew:edit:${method}:relationPrompt`, "{} is located in the city of");
  const [newObject, setNewObject] = usePersistentState(`kew:edit:${method}:newObject`, "Paris");
  const [groundTruth, setGroundTruth] = usePersistentState(`kew:edit:${method}:groundTruth`, "Seattle");
  const [editMode, setEditMode] = usePersistentState<"single" | "series">(`kew:edit:${method}:editMode`, "single");
  const [seriesEdits, setSeriesEdits] = usePersistentState(`kew:edit:${method}:seriesEdits`, '[{"subject":"Michael Jordan","relation_prompt":"{} plays the sport of","ground_truth":"basketball","new_object":"baseball"},{"subject":"The Space Needle","relation_prompt":"{} is located in the city of","ground_truth":"Seattle","new_object":"Paris"}]');
  const [seriesEditorMode, setSeriesEditorMode] = usePersistentState<"form" | "json">(`kew:edit:${method}:seriesEditorMode`, "form");
  const [targetLayers, setTargetLayers] = usePersistentState<Record<string, string>>(`kew:edit:${method}:targetLayers`, EMPTY_LAYER_VALUES);
  const [easyEditDefaultLayers, setEasyEditDefaultLayers] = useState<Record<string, number>>({});
  const [sequentialEdit, setSequentialEdit] = usePersistentState(`kew:edit:${method}:sequentialEdit`, false);
  const [mom2Adjustment, setMom2Adjustment] = usePersistentState(`kew:edit:${method}:mom2Adjustment`, false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [testingReliability, setTestingReliability] = useState(false);
  const [reliabilityTestProgress, setReliabilityTestProgress] = useState<ReliabilityTestProgress>(null);
  const [reliabilityResult, setReliabilityResult] = useState<Record<string, unknown> | null>(null);
  const reliabilityTestCancellation = useReliabilityTestCancellation();
  const [locatingLayer, setLocatingLayer] = useState(false);
  const [missingMom2Layers, setMissingMom2Layers] = useState<number[] | null>(null);
  const [mom2CacheOpen, setMom2CacheOpen] = useState(false);
  const [layerNotice, setLayerNotice] = useState("");
  const selectedModel = useMemo(() => romeModels.find((model) => model.id === modelId), [modelId, romeModels]);
  const selectedModelId = selectedModel?.id;
  const maxLayer = Math.max(0, (selectedModel?.n_layers ?? 1) - 1);
  const targetLayer = targetLayers[modelId] ?? "";
  const targetLayerNumber = parseLayer(targetLayer, maxLayer);
  const easyEditDefaultLayer = easyEditDefaultLayers[modelId];
  const modelLoaded = runtime.some((runtimeStatus) => runtimeStatus.model_id === modelId && runtimeStatus.loaded);
  const seriesEditDrafts = useMemo(() => parseFactEditDrafts(seriesEdits), [seriesEdits]);
  const isSeriesEdit = supportsSeries && editMode === "series";
  const evaluationPrompts = useMemo(
    () => buildFactEvaluationPrompts(
      isSeriesEdit
        ? seriesEditDrafts ?? []
        : [{ subject, relation_prompt: relationPrompt }]
    ),
    [isSeriesEdit, relationPrompt, seriesEditDrafts, subject]
  );
  const reliabilityEdits = useMemo(
    () => isSeriesEdit
      ? seriesEditDrafts ?? []
      : [{ subject, relation_prompt: relationPrompt, new_object: newObject, ground_truth: groundTruth }],
    [groundTruth, isSeriesEdit, newObject, relationPrompt, seriesEditDrafts, subject]
  );

  useLoadedModelSelection(romeModels, runtime, setModelId);

  useEffect(() => {
    if (romeModels.length === 0) return;
    if (!romeModels.some((model) => model.id === modelId)) {
      setModelId(romeModels[0].id);
    }
  }, [modelId, romeModels, setModelId]);

  useEffect(() => {
    setReliabilityResult(null);
  }, [modelId, reliabilityEdits]);

  useEffect(() => {
    let active = true;
    if (!selectedModelId) return;
    setLayerNotice("");
    api.editDefaultLayers({ model_id: selectedModelId, method })
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
  }, [maxLayer, method, selectedModelId, setTargetLayers]);

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
      const response = isSeriesEdit
        ? await startRomeSequentialSeries(seriesEditDrafts, {
            modelId,
            targetLayer: targetLayerNumber,
            evaluationPrompts,
            mom2Adjustment
          })
        : await (isRRome ? api.rRomeEdit : api.romeEdit)({
            model_id: modelId,
            subject,
            relation_prompt: relationPrompt,
            new_object: newObject,
            ground_truth: groundTruth || null,
            target_layer: targetLayerNumber,
            evaluation_prompts: evaluationPrompts,
            sequential_edit: sequentialEdit,
            mom2_adjustment: mom2Adjustment
          });
      const completed = await watchEditJob(response, setJob);
      setResult(completed.result ?? null);
      await onPatchesChanged?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const missingLayers = missingMom2LayersFromMessage(message);
      if (missingLayers) {
        setMissingMom2Layers(missingLayers);
        return;
      }
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  async function testReliability() {
    if (!modelLoaded || !evaluationPrompts.length || busy || testingReliability) return;
    setTestingReliability(true);
    setError("");
    setReliabilityResult(null);
    setReliabilityTestProgress(null);
    reliabilityTestCancellation.reset();
    try {
      const response = await api.easyEditReliability({
        model_id: modelId,
        method,
        edits: reliabilityEdits.map((edit) => ({
          subject: edit.subject,
          relation_prompt: edit.relation_prompt,
          new_object: edit.new_object,
          ground_truth: edit.ground_truth || null
        })),
        include_generations: true
      });
      reliabilityTestCancellation.setJobId(response.job_id);
      await watchReliabilityTestJob(response, (job) => {
        if (!job.result) return;
        setReliabilityResult(job.result);
        setReliabilityTestProgress(reliabilityTestJobProgress(job));
        setResult({
          reliability: job.result,
          reliability_test_job_id: job.job_id,
          reliability_test_item_count: job.result_item_count ?? 0,
          reliability_test_page_size: job.result_page_size ?? null,
          evaluation_only: true,
          model_id: modelId
        });
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setTestingReliability(false);
      reliabilityTestCancellation.reset();
    }
  }

  async function locateLayer() {
    if (isRRome || isSeriesEdit || !selectedModel?.supports_causal_trace || !subject.trim() || !relationPrompt.trim() || !groundTruth.trim()) return;
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

  return (
    <div className="grid gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={romeModels} value={modelId} onChange={setModelId} onRuntimeUnloaded={onPatchesChanged} />
        {supportsSeries ? <div className="grid gap-2">
          <span className="label">Edit Mode</span>
          <div className="inline-flex w-fit rounded-md border border-line bg-white p-1">
            <button aria-pressed={!isSeriesEdit} className={`rounded px-2 py-1 text-xs ${!isSeriesEdit ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} onClick={() => setEditMode("single")} type="button">Single fact</button>
            <button aria-pressed={isSeriesEdit} className={`rounded px-2 py-1 text-xs ${isSeriesEdit ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} onClick={() => setEditMode("series")} type="button">Sequential series</button>
          </div>
        </div> : null}
        {isSeriesEdit ? (
          <RomeSeriesEditor
            drafts={seriesEditDrafts}
            editorMode={seriesEditorMode}
            value={seriesEdits}
            onChange={setSeriesEdits}
            onEditorModeChange={setSeriesEditorMode}
            onError={setError}
          />
        ) : (
          <>
            <TextInput label="Subject" value={subject} onChange={setSubject} />
            <TextInput label="Relation Prompt" value={relationPrompt} onChange={setRelationPrompt} />
            <TextInput label="New Object" value={newObject} onChange={setNewObject} />
            <TextInput label="Ground Truth" value={groundTruth} onChange={setGroundTruth} />
          </>
        )}
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
              disabled={isRRome || busy || locatingLayer || isSeriesEdit || !modelLoaded || !selectedModel?.supports_causal_trace || !subject.trim() || !relationPrompt.trim() || !groundTruth.trim()}
              title={isRRome ? "R-ROME uses its selected EasyEdit layer; causal-trace auto-location is only available for ROME" : isSeriesEdit ? "Causal trace a fact in single-fact mode, then use that layer for the series" : locatingLayer ? "Running causal trace" : !modelLoaded ? "Load the selected model before causal tracing" : selectedModel?.supports_causal_trace ? "Run causal tracing on the current fact and fill the selected edit layer" : "Causal tracing is not available for this model"}
            >
              {locatingLayer ? <Loader2 className="animate-spin" size={16} /> : <LocateFixed size={16} />}
            </button>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 font-mono text-xs text-ink/50">
            <span>Valid layers: 0-{maxLayer}</span>
            <span>EasyEdit default: {easyEditDefaultLayer ?? "Loading..."}</span>
          </div>
          {isSeriesEdit ? <span className="text-xs text-ink/60">Each fact in this series uses the selected layer.</span> : null}
          {layerNotice ? <span className="text-xs text-cyan">{layerNotice}</span> : null}
        </label>
        <DerivedEvaluationPrompts
          prompts={evaluationPrompts}
          onTest={testReliability}
          onCancel={() => void reliabilityTestCancellation.cancel().catch((error) => setError(error instanceof Error ? error.message : String(error)))}
          reliabilityResult={reliabilityResult}
          testProgress={reliabilityTestProgress}
          testing={testingReliability}
          testCancelable={Boolean(reliabilityTestCancellation.jobId)}
          testCanceling={reliabilityTestCancellation.canceling}
          testDisabled={busy || !modelLoaded || !evaluationPrompts.length}
        />
        <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
          <span>{isSeriesEdit ? "Sequential edit" : "Sequential edit"}</span>
          <input type="checkbox" checked={isSeriesEdit || sequentialEdit} disabled={isSeriesEdit} onChange={(event) => setSequentialEdit(event.target.checked)} />
        </label>
        <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
          <span>Second-Moment Adjustment</span>
          <input type="checkbox" checked={mom2Adjustment} onChange={(event) => setMom2Adjustment(event.target.checked)} />
        </label>
        <div className="flex items-center justify-between gap-3">
          {mom2Adjustment ? <p className="text-xs text-ink/60">Requires prepared Wikipedia statistics for the selected layer.</p> : <span />}
          <button className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={() => setMom2CacheOpen(true)} title="Peek cached second-moment statistics" type="button">
            <Database size={14} />
            Cache
          </button>
        </div>
        <button className="tool-button w-full" onClick={run} disabled={busy || locatingLayer || targetLayerNumber === null || !modelLoaded} title={modelLoaded ? isSeriesEdit ? "Apply each fact as an ordered ROME edit at the selected layer" : `Apply ${methodLabel} edit at the selected layer` : "Load the selected model before editing"}>
          <FlaskConical size={16} />
          {busy ? "Editing" : isSeriesEdit ? "Apply ROME Series" : `Apply ${methodLabel}`}
        </button>
        <ExportActiveModelControl modelId={modelId} modelLoaded={modelLoaded} disabled={busy} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} />
      {missingMom2Layers ? <MissingMom2StatisticsDialog layers={missingMom2Layers} onClose={() => setMissingMom2Layers(null)} /> : null}
      {mom2CacheOpen ? <SecondMomentCacheDialog modelId={modelId} method={method} onClose={() => setMom2CacheOpen(false)} /> : null}
    </div>
  );
}

function RomeSeriesEditor({
  drafts,
  editorMode,
  value,
  onChange,
  onEditorModeChange,
  onError
}: {
  drafts: FactEditDraft[] | null;
  editorMode: "form" | "json";
  value: string;
  onChange: (value: string) => void;
  onEditorModeChange: (value: "form" | "json") => void;
  onError: (value: string) => void;
}) {
  function updateDraft(index: number, field: FactEditableField, fieldValue: string) {
    if (!drafts) return;
    onChange(JSON.stringify(drafts.map((edit, editIndex) => editIndex === index ? { ...edit, [field]: fieldValue } : edit), null, 2));
  }

  function addDraft() {
    onChange(JSON.stringify([...(drafts ?? []), emptyFactEditDraft()], null, 2));
  }

  function removeDraft(index: number) {
    if (!drafts) return;
    onChange(JSON.stringify(drafts.filter((_, editIndex) => editIndex !== index), null, 2));
  }

  async function uploadJson(file: File | null) {
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!Array.isArray(parsed)) throw new Error("ROME series JSON must be an array of fact objects.");
      onChange(JSON.stringify(parsed, null, 2));
      onEditorModeChange("form");
      onError("");
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  }

  function showFormEditor() {
    if (!drafts) {
      onError("Fix the sequential series JSON before switching to the form editor.");
      return;
    }
    onError("");
    onEditorModeChange("form");
  }

  function formatJson() {
    try {
      const parsed = JSON.parse(value);
      if (!Array.isArray(parsed)) throw new Error("ROME series JSON must be an array of fact objects.");
      onChange(JSON.stringify(parsed, null, 2));
      onError("");
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="label">Sequential Facts</span>
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-md border border-line bg-white p-1">
            <button aria-pressed={editorMode === "form"} className={`rounded px-2 py-1 text-xs ${editorMode === "form" ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} onClick={showFormEditor} type="button">Form</button>
            <button aria-pressed={editorMode === "json"} className={`rounded px-2 py-1 text-xs ${editorMode === "json" ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} onClick={() => onEditorModeChange("json")} type="button">JSON</button>
          </div>
          <label className="ghost-button grid size-8 shrink-0 cursor-pointer place-items-center p-0" title="Upload ROME sequential facts JSON">
            <Upload size={14} />
            <input className="sr-only" type="file" accept="application/json,.json" onChange={(event) => {
              void uploadJson(event.target.files?.[0] ?? null);
              event.target.value = "";
            }} />
          </label>
        </div>
      </div>
      {editorMode === "form" && drafts ? (
        <div className="grid max-h-96 gap-3 overflow-y-auto pr-1">
          {drafts.map((edit, index) => (
            <section key={index} className="grid gap-3 border border-line bg-panel p-3">
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium">Fact {index + 1}</span>
                <button aria-label={`Remove fact ${index + 1}`} className="ghost-button grid size-8 place-items-center p-0" disabled={drafts.length === 1} onClick={() => removeDraft(index)} title="Remove fact" type="button"><Trash2 size={15} /></button>
              </div>
              <FactInput label="Subject" value={edit.subject} onChange={(fieldValue) => updateDraft(index, "subject", fieldValue)} />
              <FactInput label="Relation Prompt" value={edit.relation_prompt} onChange={(fieldValue) => updateDraft(index, "relation_prompt", fieldValue)} />
              <FactInput label="New Object" value={edit.new_object} onChange={(fieldValue) => updateDraft(index, "new_object", fieldValue)} />
              <FactInput label="Ground Truth" value={edit.ground_truth} onChange={(fieldValue) => updateDraft(index, "ground_truth", fieldValue)} />
            </section>
          ))}
          <button className="ghost-button min-h-9 justify-self-start px-2 py-1 text-xs" onClick={addDraft} type="button"><Plus size={14} />Add fact</button>
        </div>
      ) : (
        <>
          <div className="flex justify-end"><button className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={formatJson} type="button">Format JSON</button></div>
          <textarea className="control min-h-64 font-mono text-xs" value={value} onChange={(event) => onChange(event.target.value)} />
        </>
      )}
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
  const [sequentialEdit, setSequentialEdit] = usePersistentState("kew:edit:memit:sequentialEdit", false);
  const mom2Adjustment = true;
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [testingReliability, setTestingReliability] = useState(false);
  const [reliabilityTestProgress, setReliabilityTestProgress] = useState<ReliabilityTestProgress>(null);
  const [reliabilityResult, setReliabilityResult] = useState<Record<string, unknown> | null>(null);
  const reliabilityTestCancellation = useReliabilityTestCancellation();
  const [missingMom2Layers, setMissingMom2Layers] = useState<number[] | null>(null);
  const [mom2CacheOpen, setMom2CacheOpen] = useState(false);
  const selectedModel = useMemo(() => memitModels.find((model) => model.id === modelId), [memitModels, modelId]);
  const selectedModelId = selectedModel?.id;
  const maxLayer = Math.max(0, (selectedModel?.n_layers ?? 1) - 1);
  const layers = layersByModel[modelId] ?? "";
  const selectedLayers = parseMemitLayers(layers, maxLayer);
  const easyEditDefaultLayerValues = easyEditDefaultLayers[modelId];
  const modelLoaded = runtime.some((runtimeStatus) => runtimeStatus.model_id === modelId && runtimeStatus.loaded);
  const editDrafts = useMemo(() => parseFactEditDrafts(edits), [edits]);
  const evaluationPrompts = useMemo(
    () => buildFactEvaluationPrompts(editDrafts ?? []),
    [editDrafts]
  );

  useLoadedModelSelection(memitModels, runtime, setModelId);

  useEffect(() => {
    if (memitModels.length === 0) return;
    if (!memitModels.some((model) => model.id === modelId)) {
      setModelId(memitModels[0].id);
    }
  }, [memitModels, modelId, setModelId]);

  useEffect(() => {
    setReliabilityResult(null);
  }, [editDrafts, modelId]);

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
      if (!edits.trim()) {
        throw new Error("Batch Edits JSON is empty. Add at least one MEMIT fact.");
      }
      const parsedEdits = parseFactEditDrafts(edits);
      if (!parsedEdits) {
        throw new Error("Batch Edits JSON is invalid. Use a JSON array of MEMIT fact objects.");
      }
      if (!parsedEdits.length) {
        throw new Error("Batch Edits JSON must include at least one MEMIT fact.");
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
        evaluation_prompts: evaluationPrompts,
        sequential_edit: sequentialEdit,
        mom2_adjustment: mom2Adjustment
      });
      const completed = await watchEditJob(response, setJob);
      setResult(completed.result ?? null);
      await onPatchesChanged?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const missingLayers = missingMom2LayersFromMessage(message);
      if (missingLayers) {
        setMissingMom2Layers(missingLayers);
        return;
      }
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  async function testReliability() {
    if (!modelLoaded || !evaluationPrompts.length || busy || testingReliability || !editDrafts?.length) return;
    setTestingReliability(true);
    setError("");
    setReliabilityResult(null);
    setReliabilityTestProgress(null);
    reliabilityTestCancellation.reset();
    try {
      const response = await api.easyEditReliability({
        model_id: modelId,
        method: "memit",
        edits: editDrafts.map((edit) => ({
          subject: edit.subject,
          relation_prompt: edit.relation_prompt,
          new_object: edit.new_object,
          ground_truth: edit.ground_truth || null
        })),
        include_generations: true
      });
      reliabilityTestCancellation.setJobId(response.job_id);
      await watchReliabilityTestJob(response, (job) => {
        if (!job.result) return;
        setReliabilityResult(job.result);
        setReliabilityTestProgress(reliabilityTestJobProgress(job));
        setResult({
          reliability: job.result,
          reliability_test_job_id: job.job_id,
          reliability_test_item_count: job.result_item_count ?? 0,
          reliability_test_page_size: job.result_page_size ?? null,
          evaluation_only: true,
          model_id: modelId
        });
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setTestingReliability(false);
      reliabilityTestCancellation.reset();
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

  function updateEditDraft(index: number, field: FactEditableField, value: string) {
    if (!editDrafts) return;
    setEdits(JSON.stringify(editDrafts.map((edit, editIndex) => editIndex === index ? { ...edit, [field]: value } : edit), null, 2));
  }

  function addEditDraft() {
    const nextEdits = editDrafts ?? [];
    setEdits(JSON.stringify([...nextEdits, emptyFactEditDraft()], null, 2));
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

  return (
    <div className="grid gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={memitModels} value={modelId} onChange={setModelId} onRuntimeUnloaded={onPatchesChanged} />
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
                  <FactInput label="Subject" value={edit.subject} onChange={(value) => updateEditDraft(index, "subject", value)} />
                  <FactInput label="Relation Prompt" value={edit.relation_prompt} onChange={(value) => updateEditDraft(index, "relation_prompt", value)} />
                  <FactInput label="New Object" value={edit.new_object} onChange={(value) => updateEditDraft(index, "new_object", value)} />
                  <FactInput label="Ground Truth" value={edit.ground_truth} onChange={(value) => updateEditDraft(index, "ground_truth", value)} />
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
        <DerivedEvaluationPrompts
          prompts={evaluationPrompts}
          onTest={testReliability}
          onCancel={() => void reliabilityTestCancellation.cancel().catch((error) => setError(error instanceof Error ? error.message : String(error)))}
          reliabilityResult={reliabilityResult}
          testProgress={reliabilityTestProgress}
          testing={testingReliability}
          testCancelable={Boolean(reliabilityTestCancellation.jobId)}
          testCanceling={reliabilityTestCancellation.canceling}
          testDisabled={busy || !modelLoaded || !evaluationPrompts.length || !editDrafts?.length}
        />
        <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
          <span>Sequential edit</span>
          <input type="checkbox" checked={sequentialEdit} onChange={(event) => setSequentialEdit(event.target.checked)} />
        </label>
        <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
          <span>Second-Moment Adjustment</span>
          <input type="checkbox" checked={mom2Adjustment} disabled title="Required for MEMIT" />
        </label>
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-ink/60">Required for MEMIT. Prepare Wikipedia statistics for every selected layer.</p>
          <button className="ghost-button min-h-8 shrink-0 px-2 py-1 text-xs" onClick={() => setMom2CacheOpen(true)} title="Peek cached second-moment statistics" type="button">
            <Database size={14} />
            Cache
          </button>
        </div>
        <button className="tool-button w-full" onClick={run} disabled={busy || !modelLoaded || !selectedLayers?.length} title={modelLoaded ? "Apply MEMIT edit at the selected layers" : "Load the selected model before editing"}>
          <Layers size={16} />
          {busy ? "Editing" : "Apply MEMIT"}
        </button>
        <ExportActiveModelControl modelId={modelId} modelLoaded={modelLoaded} disabled={busy} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} />
      {missingMom2Layers ? <MissingMom2StatisticsDialog layers={missingMom2Layers} onClose={() => setMissingMom2Layers(null)} /> : null}
      {mom2CacheOpen ? <SecondMomentCacheDialog modelId={modelId} method="memit" onClose={() => setMom2CacheOpen(false)} /> : null}
    </div>
  );
}

function MissingMom2StatisticsDialog({ layers, onClose }: { layers: number[]; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/30 p-4 sm:items-center" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }} role="presentation">
      <section aria-labelledby="missing-mom2-title" aria-modal="true" className="my-auto w-full max-w-md rounded-md border border-line bg-white p-5 shadow-xl" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <span className="label">Second-Moment Statistics Required</span>
        <h2 id="missing-mom2-title" className="mt-1 text-base font-semibold">Statistics are missing for layer{layers.length === 1 ? "" : "s"} {layers.join(", ")}</h2>
        <p className="mt-3 text-sm leading-6 text-ink/70">This edit was not started. Prepare the selected layers locally or import compatible NPZ files in Settings, then apply the edit again.</p>
        <div className="mt-5 flex justify-end gap-2">
          <button className="ghost-button min-h-9 px-3 py-1" onClick={onClose} type="button">Close</button>
          <Link className="tool-button min-h-9 px-3 py-1" href="/settings" onClick={onClose}>Open Settings</Link>
        </div>
      </section>
    </div>
  );
}

function missingMom2LayersFromMessage(message: string): number[] | null {
  const match = message.match(/MOM2 statistics are missing for .*? layer\(s\)\s+([\d,\s]+)(?:\.|$)/i);
  if (!match) return null;
  const layers = match[1]
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value >= 0);
  return layers.length ? layers : null;
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

function parseMemitLayers(value: string, maxLayer: number): number[] | null {
  const entries = value.split(",").map((entry) => entry.trim());
  if (!entries.length || entries.some((entry) => !/^\d+$/.test(entry))) return null;
  const layers = entries.map(Number);
  if (layers.some((layer) => !Number.isSafeInteger(layer) || layer < 0 || layer > maxLayer)) return null;
  return new Set(layers).size === layers.length ? layers : null;
}

function isEditableMemitLayerInput(value: string, maxLayer: number) {
  if (value === "") return true;
  if (!/^\s*\d+\s*(?:,\s*\d+\s*)*,?\s*$/.test(value)) return false;
  const completeValue = value.trim().replace(/,$/, "").trim();
  return !completeValue || completeValue.split(",").every((entry) => {
    const layer = Number(entry.trim());
    return Number.isSafeInteger(layer) && layer >= 0 && layer <= maxLayer;
  });
}

function buildEditPrompt(subject: string, relationPrompt: string) {
  return relationPrompt.includes("{}") ? relationPrompt.replace("{}", subject) : `${subject} ${relationPrompt}`.trim();
}

function buildFactEvaluationPrompts(edits: Array<Pick<FactEditDraft, "subject" | "relation_prompt">>) {
  return edits.flatMap((edit) => {
    if (!edit.subject.trim() || !edit.relation_prompt.trim()) return [];
    return [buildEditPrompt(edit.subject, edit.relation_prompt)];
  });
}

async function generateReliabilityTestOutputs(
  modelId: string,
  edits: Array<Pick<FactEditDraft, "subject" | "relation_prompt" | "new_object">> | null
) {
  const generations = [];
  for (const edit of edits ?? []) {
    if (!edit.subject.trim() || !edit.relation_prompt.trim()) continue;
    const prompt = buildEditPrompt(edit.subject, edit.relation_prompt);
    const response = await api.generate({
      model_id: modelId,
      prompt,
      max_new_tokens: 32,
      decoding_strategy: "greedy"
    });
    generations.push({ prompt, text: response.text, target_new: edit.new_object });
  }
  return generations;
}

async function watchReliabilityTestJob(initial: EditJob, onUpdate: (job: EditJob) => void): Promise<EditJob> {
  announceExperimentStarted(initial.experiment_id);
  return new Promise<EditJob>((resolve, reject) => {
    let current = initial;
    let timer: number | null = null;
    let settled = false;
    let stopCanceledListener: () => void = () => undefined;
    const finish = () => {
      if (timer !== null) window.clearTimeout(timer);
      stopCanceledListener();
    };
    const succeed = (job: EditJob) => {
      if (settled) return;
      settled = true;
      finish();
      resolve(job);
    };
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      finish();
      reject(error);
    };
    stopCanceledListener = listenForExperimentCanceled(({ jobId }) => {
      if (jobId !== initial.job_id) return;
      current = { ...current, status: "canceled" };
      onUpdate(current);
      succeed(current);
    });
    const poll = () => {
      if (settled) return;
      onUpdate(current);
      if (current.status !== "running") {
        if (current.status === "failed") fail(new Error(current.logs[current.logs.length - 1] ?? "Reliability evaluation failed."));
        else succeed(current);
        return;
      }
      timer = window.setTimeout(() => {
        void api.easyEditReliabilityJob(current.job_id)
          .then((job) => {
            if (settled) return;
            current = job;
            poll();
          })
          .catch((error) => fail(error instanceof Error ? error : new Error(String(error))));
      }, 500);
    };
    poll();
  });
}

function useReliabilityTestCancellation() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [canceling, setCanceling] = useState(false);
  const reset = useCallback(() => {
    setJobId(null);
    setCanceling(false);
  }, []);
  const cancel = useCallback(async () => {
    if (!jobId || canceling) return;
    setCanceling(true);
    try {
      const canceledJob = await api.cancelEvaluationJob(jobId);
      if (canceledJob.status !== "canceled") throw new Error("This evaluation is no longer running.");
      announceExperimentCanceled({ experimentId: canceledJob.experiment_id, jobId: canceledJob.job_id });
    } catch (error) {
      setCanceling(false);
      throw error;
    }
  }, [canceling, jobId]);
  return { cancel, canceling, jobId, reset, setJobId };
}

function reliabilityTestJobProgress(job: EditJob): ReliabilityTestProgress {
  const completed = job.progress_current;
  const total = job.progress_total;
  return typeof completed === "number" && typeof total === "number" && total > 0
    ? { completed, total }
    : null;
}

function DerivedEvaluationPrompts({
  prompts,
  onTest,
  onCancel,
  reliabilityResult,
  testProgress,
  testing,
  testCancelable,
  testCanceling,
  testDisabled
}: {
  prompts: string[];
  onTest: () => void;
  onCancel: () => void;
  reliabilityResult: Record<string, unknown> | null;
  testProgress: ReliabilityTestProgress;
  testing: boolean;
  testCancelable: boolean;
  testCanceling: boolean;
  testDisabled: boolean;
}) {
  const meanReliability = typeof reliabilityResult?.mean_reliability === "number"
    ? reliabilityResult.mean_reliability
    : null;
  const cases = Array.isArray(reliabilityResult?.cases)
    ? reliabilityResult.cases.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
    : [];
  const completed = testProgress?.completed ?? 0;
  const total = testProgress?.total ?? 0;

  return (
    <div className="grid min-w-0 max-w-full gap-2">
      <span className="flex min-w-0 items-center justify-between gap-2">
        <span className="label min-w-0">Edit Evaluation Prompts</span>
        <button
          type="button"
          className="ghost-button min-h-8 px-2 py-1 text-xs"
          disabled={testDisabled || testing}
          onClick={onTest}
          title="Evaluate EasyEdit rewrite reliability on the current active model without applying an edit"
        >
          <FlaskConical size={14} />
          {testing ? "Testing" : "Test"}
        </button>
      </span>
      <textarea
        aria-label="Edit evaluation prompts derived from the current facts"
        className="control h-28 resize-y overflow-y-auto"
        placeholder="Add an edit fact to create its evaluation prompt."
        readOnly
        value={prompts.join("\n")}
      />
      {testing ? (
        <EvaluationProgressIndicator cancelDisabled={!testCancelable} canceling={testCanceling} current={completed} multiFactLabel="Testing edit facts…" onCancel={onCancel} preparingLabel="Preparing EasyEdit…" singleFactLabel="Evaluating fact…" total={total} />
      ) : null}
      {reliabilityResult ? (
        <div className="grid min-w-0 max-w-full gap-2 overflow-hidden rounded-md border border-cyan/30 bg-cyan/5 p-3 text-sm">
          <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-baseline gap-2">
            <span className="min-w-0 break-words font-medium">Reliability · EasyEdit rewrite_acc</span>
            <span className="font-mono text-cyan">{meanReliability === null ? "Metric unavailable" : formatReliabilityScore(meanReliability)}</span>
          </div>
          {cases.length ? (
            <div className="grid min-w-0 max-w-full max-h-36 gap-2 overflow-x-hidden overflow-y-auto pr-1 text-xs">
              {cases.map((testCase, index) => {
                const prompt = typeof testCase.prompt === "string" ? testCase.prompt : "";
                const target = typeof testCase.target_new === "string" ? testCase.target_new : "";
                const score = typeof testCase.reliability_score === "number" ? testCase.reliability_score : null;
                return (
                  <div key={`${prompt}-${index}`} className="grid min-w-0 max-w-full grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 gap-y-1 border-t border-cyan/20 pt-2 first:border-t-0 first:pt-0">
                    <span className="min-w-0 max-w-full break-words font-mono text-ink/65">{prompt} <span className="text-ink/40">→</span> {target}</span>
                    <span className="shrink-0 font-mono text-ink">{score === null ? "—" : formatReliabilityScore(score)}</span>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function formatReliabilityScore(score: number) {
  return `${(score * 100).toFixed(1)}%`;
}

async function startRomeSequentialSeries(
  drafts: FactEditDraft[] | null,
  {
    modelId,
    targetLayer,
    evaluationPrompts,
    mom2Adjustment
  }: {
    modelId: string;
    targetLayer: number;
    evaluationPrompts: string[];
    mom2Adjustment: boolean;
  }
) {
  if (!drafts) {
    throw new Error("Sequential Facts JSON is invalid. Use a JSON array of ROME fact objects.");
  }
  if (!drafts.length) {
    throw new Error("Sequential Facts JSON must include at least one ROME fact.");
  }
  if (drafts.length > 100) {
    throw new Error("ROME sequential series supports at most 100 facts at once.");
  }
  const invalidEdit = drafts.findIndex((edit) => !edit.subject.trim() || !edit.relation_prompt.trim() || !edit.new_object.trim());
  if (invalidEdit >= 0) {
    throw new Error(`Fact ${invalidEdit + 1} needs a subject, relation prompt, and new object.`);
  }
  return api.romeSequentialEdits({
    model_id: modelId,
    edits: drafts.map((edit) => ({
      ...edit,
      ground_truth: edit.ground_truth.trim() || null
    })),
    target_layer: targetLayer,
    evaluation_prompts: evaluationPrompts,
    mom2_adjustment: mom2Adjustment
  });
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

function emptyFactEditDraft(): FactEditDraft {
  return {
    subject: "",
    relation_prompt: "{} is",
    ground_truth: "",
    new_object: ""
  };
}

function parseFactEditDrafts(value: string): FactEditDraft[] | null {
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) return null;
    return parsed.map((item) => {
      if (!item || typeof item !== "object") return emptyFactEditDraft();
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

function FactInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
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
                <div className="whitespace-pre-wrap break-words text-sm leading-6 text-ink">
                  <HighlightedGeneration text={generation.text} target={generation.targetNew} />
                </div>
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
  const targetByPrompt = new Map<string, string>();
  const patch = result?.patch;
  if (patch && typeof patch === "object") {
    const edits = (patch as Record<string, unknown>).edits;
    if (Array.isArray(edits)) {
      for (const edit of edits) {
        if (!edit || typeof edit !== "object") continue;
        const record = edit as Record<string, unknown>;
        if (typeof record.subject !== "string" || typeof record.relation_prompt !== "string" || typeof record.new_object !== "string") continue;
        targetByPrompt.set(buildEditPrompt(record.subject, record.relation_prompt), record.new_object);
      }
    }
  }
  return value
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const record = item as Record<string, unknown>;
      const prompt = typeof record.prompt === "string" ? record.prompt : "";
      return {
        prompt,
        text: typeof record.text === "string" ? record.text : "",
        targetNew: typeof record.target_new === "string" ? record.target_new : targetByPrompt.get(prompt) ?? ""
      };
    })
    .filter((item): item is { prompt: string; text: string; targetNew: string } => Boolean(item?.text));
}

function HighlightedGeneration({ text, target }: { text: string; target: string }) {
  const normalizedTarget = target.trim();
  if (!normalizedTarget) return <>{text}</>;

  const normalizedText = text.toLocaleLowerCase();
  const needle = normalizedTarget.toLocaleLowerCase();
  const fragments: ReactNode[] = [];
  let cursor = 0;
  let matchIndex = normalizedText.indexOf(needle, cursor);

  while (matchIndex >= 0) {
    if (matchIndex > cursor) fragments.push(text.slice(cursor, matchIndex));
    fragments.push(
      <mark key={`${matchIndex}-${cursor}`} className="rounded bg-cyan/20 px-0.5 font-medium text-cyan">
        {text.slice(matchIndex, matchIndex + normalizedTarget.length)}
      </mark>
    );
    cursor = matchIndex + normalizedTarget.length;
    matchIndex = normalizedText.indexOf(needle, cursor);
  }

  if (cursor === 0) return <>{text}</>;
  if (cursor < text.length) fragments.push(text.slice(cursor));
  return <>{fragments}</>;
}
