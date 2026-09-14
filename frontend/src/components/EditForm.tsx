"use client";

import { Database, Download, FlaskConical, Layers, Loader2, LocateFixed, Plus, Save, Trash2, Upload, X } from "lucide-react";
import Link from "next/link";
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { announceEditJobCanceling, announceExperimentCanceled, announceExperimentStarted, listenForEditJobCanceling, listenForExperimentCanceled } from "@/lib/experimentEvents";
import { type FactEditDraft, type IndexedFactDataset, useIndexedFactDataset } from "@/lib/factDataset";
import { isModelRuntimeConnected, listenForModelRuntimeConnection } from "@/lib/modelRuntimeEvents";
import type { EditJob, ModelRuntimeStatus, Patch, ReliabilityTestResultItem, ReliabilityTestResultPage, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelSelector } from "./ModelSelector";
import { EvaluationProgressIndicator } from "./EvaluationProgressIndicator";
import { PatchHistory } from "./PatchHistory";
import { SecondMomentCacheDialog } from "./SecondMomentCacheDialog";
import { VirtualizedList } from "./VirtualizedList";
import { useWorkbenchSession } from "./WorkbenchSession";

const EMPTY_LAYER_VALUES: Record<string, string> = {};
const DEFAULT_FACT_EDIT_DRAFTS: FactEditDraft[] = [
  { subject: "Michael Jordan", relation_prompt: "{} plays the sport of", ground_truth: "basketball", new_object: "baseball" },
  { subject: "The Space Needle", relation_prompt: "{} is located in the city of", ground_truth: "Seattle", new_object: "Paris" }
];
const VIRTUAL_FACT_CARD_HEIGHT = 412;
const VIRTUAL_FACT_OVERSCAN = 3;
const ACTIVE_EDIT_JOB_STORAGE_KEY = "kew:edit:activeJob";

function useRuntimeConnection() {
  const [connected, setConnected] = useState(() => isModelRuntimeConnected());
  useEffect(() => listenForModelRuntimeConnection(setConnected), []);
  return connected;
}

type FactEditableField = "subject" | "relation_prompt" | "ground_truth" | "new_object";
type ReliabilityTestProgress = { completed: number; total: number } | null;

export function EditForm({ models, patches, runtime = [], onPatchesChanged }: { models: SupportedModel[]; patches: Patch[]; runtime?: ModelRuntimeStatus[]; onPatchesChanged?: () => void | Promise<void> }) {
  const [tab, setTab] = usePersistentState<"rome" | "r_rome" | "kn" | "memit" | "patches">("kew:edit:tab", "rome");
  const { edit, setEdit } = useWorkbenchSession();
  const setResult = useCallback((result: Record<string, unknown> | null) => setEdit((current) => ({ ...current, result })), [setEdit]);
  const setJob = useCallback((job: EditJob | null) => setEdit((current) => ({ ...current, job })), [setEdit]);
  const resumeAttempted = useRef(false);
  const resumeAbortController = useRef<AbortController | null>(null);

  useEffect(() => {
    if (resumeAttempted.current || edit.job) return;
    const jobId = readActiveEditJobId();
    if (!jobId) return;
    resumeAttempted.current = true;
    const controller = new AbortController();
    resumeAbortController.current = controller;

    async function resume() {
      try {
        const current = await api.editJob(jobId);
        if (controller.signal.aborted) return;
        setJob(current);
        if (current.status !== "running") {
          clearActiveEditJob();
          setResult(current.result ?? null);
          await onPatchesChanged?.();
          return;
        }
        const completed = await watchEditJob(current, setJob, { announceStart: false, signal: controller.signal });
        if (controller.signal.aborted) return;
        setResult(completed.result ?? null);
        await onPatchesChanged?.();
      } catch (error) {
        if (controller.signal.aborted) return;
        console.warn("Unable to reattach to the active edit job.", error);
        resumeAttempted.current = false;
      }
    }

    void resume();
  }, [edit.job, onPatchesChanged, setJob, setResult]);

  useEffect(() => () => {
    resumeAbortController.current?.abort();
    resumeAbortController.current = null;
    resumeAttempted.current = false;
  }, []);

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
            <ResultPanel result={edit.result} job={edit.job} onPatchesChanged={onPatchesChanged} />
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
      if (!sequentialEdit) await onPatchesChanged?.();
      const completed = await watchEditJob(response, setJob);
      const completedResult = completed.result ?? null;
      setResult(completedResult);
      setReliabilityResult(reliabilityFromEditResult(completedResult));
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
          <span>Continue from active patches</span>
          <input type="checkbox" checked={sequentialEdit} onChange={(event) => setSequentialEdit(event.target.checked)} title="Keep the active patch lineage for this edit" />
        </label>
        <button className="tool-button w-full" onClick={run} disabled={busy || !modelLoaded} title={modelLoaded ? "Identify knowledge neurons and apply the KN edit" : "Load the selected model before editing"}>
          <FlaskConical size={16} />
          {busy ? "Editing" : "Apply KN"}
        </button>
        <ExportActiveModelControl modelId={modelId} modelLoaded={modelLoaded} disabled={busy} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} onPatchesChanged={onPatchesChanged} />
    </div>
  );
}

async function watchEditJob(
  initial: EditJob,
  onUpdate: (job: EditJob) => void,
  options: { announceStart?: boolean; signal?: AbortSignal } = {}
) {
  if (options.announceStart !== false) announceExperimentStarted(initial.experiment_id);
  persistActiveEditJob(initial.job_id);
  onUpdate(initial);
  return new Promise<EditJob>((resolve, reject) => {
    let current = initial;
    let settled = false;
    let stopSubscription = () => undefined;
    const finish = () => {
      stopSubscription();
      options.signal?.removeEventListener("abort", stop);
    };
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      finish();
      reject(error);
    };
    const succeed = (job: EditJob) => {
      if (settled) return;
      settled = true;
      finish();
      resolve(job);
    };
    const stop = () => fail(new DOMException("Edit job watcher stopped.", "AbortError"));
    if (options.signal?.aborted) {
      stop();
      return;
    }
    options.signal?.addEventListener("abort", stop, { once: true });
    stopSubscription = api.subscribeToEvents(`/api/edit/jobs/${initial.job_id}/events`, {
      onEvent: (event, data) => {
        if (event === "job") {
          try {
            current = mergeEditJobLogs(current, JSON.parse(data) as EditJob);
            onUpdate(current);
            if (current.status === "running") return;
            clearActiveEditJob();
            if (current.status === "failed") fail(new Error(current.logs[current.logs.length - 1] ?? "Edit job failed"));
            else succeed(current);
          } catch (error) {
            fail(error instanceof Error ? error : new Error(String(error)));
          }
        }
        if (event === "complete" && current.status === "running") {
          fail(new Error("Edit job stream closed before the job completed."));
        }
      }
    });
  });
}

function persistActiveEditJob(jobId: string) {
  window.sessionStorage.setItem(ACTIVE_EDIT_JOB_STORAGE_KEY, jobId);
}

function readActiveEditJobId() {
  return window.sessionStorage.getItem(ACTIVE_EDIT_JOB_STORAGE_KEY)?.trim() || null;
}

function clearActiveEditJob() {
  window.sessionStorage.removeItem(ACTIVE_EDIT_JOB_STORAGE_KEY);
}

function mergeEditJobLogs(current: EditJob, update: EditJob): EditJob {
  const expectedStart = current.log_count ?? current.logs.length;
  const logs = update.log_start_count === expectedStart
    ? [...current.logs, ...update.logs].slice(-500)
    : update.logs.slice(-500);
  return { ...update, logs };
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
  const supportsSeries = true;
  const romeModels = useMemo(() => models.filter((model) => isRRome ? model.supports_r_rome : model.supports_rome), [isRRome, models]);
  const [modelId, setModelId] = usePersistentState(`kew:edit:${method}:modelId`, romeModels[0]?.id ?? "gpt2-xl");
  const [subject, setSubject] = usePersistentState(`kew:edit:${method}:subject`, "The Space Needle");
  const [relationPrompt, setRelationPrompt] = usePersistentState(`kew:edit:${method}:relationPrompt`, "{} is located in the city of");
  const [newObject, setNewObject] = usePersistentState(`kew:edit:${method}:newObject`, "Paris");
  const [groundTruth, setGroundTruth] = usePersistentState(`kew:edit:${method}:groundTruth`, "Seattle");
  const [editMode, setEditMode] = usePersistentState<"single" | "series">(`kew:edit:${method}:editMode`, "single");
  const [seriesEditorMode, setSeriesEditorMode] = usePersistentState<"form" | "json">(`kew:edit:${method}:seriesEditorMode`, "form");
  const [targetLayers, setTargetLayers] = usePersistentState<Record<string, string>>(`kew:edit:${method}:targetLayers`, EMPTY_LAYER_VALUES);
  const [easyEditDefaultLayers, setEasyEditDefaultLayers] = useState<Record<string, number>>({});
  const [sequentialEdit, setSequentialEdit] = usePersistentState(`kew:edit:${method}:sequentialEdit`, false);
  const [seriesContinuationConfigured, setSeriesContinuationConfigured] = usePersistentState(`kew:edit:${method}:seriesContinuationConfigured`, false);
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
  const seriesDataset = useIndexedFactDataset({
    datasetId: `kew:edit:${method}:sequential-facts`,
    initialValues: DEFAULT_FACT_EDIT_DRAFTS,
    legacyStorageKey: `kew:edit:${method}:seriesEdits`,
    normalize: normalizeFactEditDraft
  });
  const selectedModel = useMemo(() => romeModels.find((model) => model.id === modelId), [modelId, romeModels]);
  const selectedModelId = selectedModel?.id;
  const maxLayer = Math.max(0, (selectedModel?.n_layers ?? 1) - 1);
  const targetLayer = targetLayers[modelId] ?? "";
  const targetLayerNumber = parseLayer(targetLayer, maxLayer);
  const easyEditDefaultLayer = easyEditDefaultLayers[modelId];
  const modelLoaded = runtime.some((runtimeStatus) => runtimeStatus.model_id === modelId && runtimeStatus.loaded);
  const runtimeConnected = useRuntimeConnection();
  const isSeriesEdit = supportsSeries && editMode === "series";
  const seriesPromptPreview = useFactEvaluationPromptPreview(seriesDataset, isSeriesEdit);
  const singleFact = useMemo(
    () => ({ subject, relation_prompt: relationPrompt, new_object: newObject, ground_truth: groundTruth }),
    [groundTruth, newObject, relationPrompt, subject]
  );
  const evaluationPrompts = useMemo(
    () => isSeriesEdit ? seriesPromptPreview : buildFactEvaluationPrompts([singleFact]),
    [isSeriesEdit, seriesPromptPreview, singleFact]
  );

  useLoadedModelSelection(romeModels, runtime, setModelId);

  useEffect(() => {
    if (!isSeriesEdit || seriesContinuationConfigured) return;
    setSequentialEdit(true);
    setSeriesContinuationConfigured(true);
  }, [isSeriesEdit, seriesContinuationConfigured, setSequentialEdit, setSeriesContinuationConfigured]);

  useEffect(() => {
    if (romeModels.length === 0) return;
    if (!romeModels.some((model) => model.id === modelId)) {
      setModelId(romeModels[0].id);
    }
  }, [modelId, romeModels, setModelId]);

  useEffect(() => {
    setReliabilityResult(null);
  }, [isSeriesEdit, modelId, seriesDataset.revision, singleFact]);

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
      const seriesFacts = isSeriesEdit ? await seriesDataset.getAll() : null;
      const fullEvaluationPrompts = isSeriesEdit
        ? buildFactEvaluationPrompts(seriesFacts ?? [])
        : evaluationPrompts;
      const response = isSeriesEdit
        ? await startRomeSequentialSeries(seriesFacts, {
            modelId,
            method,
            targetLayer: targetLayerNumber,
            evaluationPrompts: fullEvaluationPrompts,
            sequentialEdit,
            mom2Adjustment
          })
        : await (isRRome ? api.rRomeEdit : api.romeEdit)({
            model_id: modelId,
            subject,
            relation_prompt: relationPrompt,
            new_object: newObject,
            ground_truth: groundTruth || null,
            target_layer: targetLayerNumber,
            evaluation_prompts: fullEvaluationPrompts,
            sequential_edit: sequentialEdit,
            mom2_adjustment: mom2Adjustment
          });
      if (!sequentialEdit) await onPatchesChanged?.();
      const completed = await watchEditJob(response, setJob);
      const completedResult = completed.result ?? null;
      setResult(completedResult);
      setReliabilityResult(reliabilityFromEditResult(completedResult));
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
      const facts = isSeriesEdit ? await seriesDataset.getAll() : [singleFact];
      if (!facts.length) throw new Error("Add at least one fact before testing reliability.");
      const response = await api.easyEditReliability({
        model_id: modelId,
        method,
        edits: facts.map((edit) => ({
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
            <button aria-pressed={isSeriesEdit} className={`rounded px-2 py-1 text-xs ${isSeriesEdit ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} onClick={() => { if (!isSeriesEdit) { setSequentialEdit(true); setSeriesContinuationConfigured(true); } setEditMode("series"); }} type="button">Sequential series</button>
          </div>
        </div> : null}
        {isSeriesEdit ? (
          <FactDatasetEditor
            label="Sequential Facts"
            dataset={seriesDataset}
            editorMode={seriesEditorMode}
            onEditorModeChange={setSeriesEditorMode}
            onError={setError}
            uploadTitle={`${methodLabel} sequential facts JSON`}
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
            <span>EasyEdit default: {runtimeConnected ? easyEditDefaultLayer ?? "Loading..." : "Unknown"}</span>
          </div>
          {isSeriesEdit ? <span className="text-xs text-ink/60">Each fact in this series uses the selected layer.</span> : null}
          {layerNotice ? <span className="text-xs text-cyan">{layerNotice}</span> : null}
        </label>
        <DerivedEvaluationPrompts
          prompts={evaluationPrompts}
          totalPromptCount={isSeriesEdit ? seriesDataset.count : undefined}
          onTest={testReliability}
          onCancel={() => void reliabilityTestCancellation.cancel().catch((error) => setError(error instanceof Error ? error.message : String(error)))}
          reliabilityResult={reliabilityResult}
          testProgress={reliabilityTestProgress}
          testing={testingReliability}
          testCancelable={Boolean(reliabilityTestCancellation.jobId)}
          testCanceling={reliabilityTestCancellation.canceling}
          testDisabled={busy || !modelLoaded || (isSeriesEdit ? seriesDataset.status !== "ready" || seriesDataset.count === 0 : !evaluationPrompts.length)}
        />
        <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
          <span>Continue from active patches</span>
          <input type="checkbox" checked={sequentialEdit} onChange={(event) => setSequentialEdit(event.target.checked)} title="Keep the active patch lineage for this edit" />
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
        <button className="tool-button w-full" onClick={run} disabled={busy || locatingLayer || targetLayerNumber === null || !modelLoaded || (isSeriesEdit && seriesDataset.status !== "ready")} title={modelLoaded ? isSeriesEdit ? `Apply each fact as an ordered ${methodLabel} edit at the selected layer` : `Apply ${methodLabel} edit at the selected layer` : "Load the selected model before editing"}>
          <FlaskConical size={16} />
          {busy ? "Editing" : isSeriesEdit ? `Apply ${methodLabel} Series` : `Apply ${methodLabel}`}
        </button>
        <ExportActiveModelControl modelId={modelId} modelLoaded={modelLoaded} disabled={busy} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} onPatchesChanged={onPatchesChanged} />
      {missingMom2Layers ? <MissingMom2StatisticsDialog layers={missingMom2Layers} onClose={() => setMissingMom2Layers(null)} /> : null}
      {mom2CacheOpen ? <SecondMomentCacheDialog modelId={modelId} method={method} onClose={() => setMom2CacheOpen(false)} /> : null}
    </div>
  );
}

function FactDatasetEditor({
  dataset,
  editorMode,
  label,
  onEditorModeChange,
  onError,
  uploadTitle
}: {
  dataset: IndexedFactDataset<FactEditDraft>;
  editorMode: "form" | "json";
  label: string;
  onEditorModeChange: (value: "form" | "json") => void;
  onError: (value: string) => void;
  uploadTitle: string;
}) {
  const [jsonValue, setJsonValue] = useState<string | null>(null);
  const [loadingJson, setLoadingJson] = useState(false);
  const [savingJson, setSavingJson] = useState(false);

  useEffect(() => {
    if (editorMode !== "json") {
      setJsonValue(null);
      return;
    }
    let active = true;
    setLoadingJson(true);
    void dataset.getJson()
      .then((value) => {
        if (active) setJsonValue(value);
      })
      .catch((reason) => {
        if (active) onError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (active) setLoadingJson(false);
      });
    return () => {
      active = false;
    };
  }, [dataset.getJson, dataset.revision, editorMode, onError]);

  async function showFormEditor() {
    if (editorMode !== "json") {
      onEditorModeChange("form");
      return;
    }
    if (jsonValue === null) return;
    setSavingJson(true);
    try {
      const drafts = parseFactEditDrafts(jsonValue);
      if (!drafts) throw new Error(`${label} JSON must be an array of fact objects.`);
      await dataset.replace(drafts);
      onEditorModeChange("form");
      onError("");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSavingJson(false);
    }
  }

  async function uploadJson(file: File | null) {
    if (!file) return;
    try {
      const drafts = parseFactEditDrafts(await file.text());
      if (!drafts) throw new Error(`${label} JSON must be an array of fact objects.`);
      if (!drafts.length) throw new Error(`${label} JSON must include at least one fact.`);
      await dataset.replace(drafts);
      onEditorModeChange("form");
      onError("");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function clearFacts() {
    try {
      await dataset.clear();
      setJsonValue(null);
      onEditorModeChange("form");
      onError("");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function formatJson() {
    if (jsonValue === null) return;
    const drafts = parseFactEditDrafts(jsonValue);
    if (!drafts) {
      onError(`${label} JSON must be an array of fact objects.`);
      return;
    }
    setJsonValue(JSON.stringify(drafts, null, 2));
    onError("");
  }

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="label">{label}</span>
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-md border border-line bg-white p-1">
            <button aria-pressed={editorMode === "form"} className={`rounded px-2 py-1 text-xs ${editorMode === "form" ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} disabled={savingJson} onClick={() => void showFormEditor()} type="button">Form</button>
            <button aria-pressed={editorMode === "json"} className={`rounded px-2 py-1 text-xs ${editorMode === "json" ? "bg-ink text-white" : "text-ink/70 hover:text-ink"}`} disabled={dataset.status !== "ready" || savingJson} onClick={() => onEditorModeChange("json")} type="button">JSON</button>
          </div>
          <label className="ghost-button grid size-8 shrink-0 cursor-pointer place-items-center p-0" title={`Upload ${uploadTitle}`}>
            <Upload size={14} />
            <input className="sr-only" disabled={dataset.status !== "ready" || savingJson} type="file" accept="application/json,.json" onChange={(event) => {
              void uploadJson(event.target.files?.[0] ?? null);
              event.target.value = "";
            }} />
          </label>
          <button aria-label={`Clear all ${label.toLocaleLowerCase()}`} className="ghost-button grid size-8 shrink-0 place-items-center p-0 text-rust hover:text-rust" disabled={dataset.status !== "ready" || dataset.count === 0 || savingJson} onClick={() => void clearFacts()} title={`Clear all ${label.toLocaleLowerCase()}`} type="button"><Trash2 size={14} /></button>
        </div>
      </div>
      {editorMode === "form" ? (
        dataset.status === "loading" ? <div className="rounded-md border border-line bg-panel p-3 text-xs text-ink/60">Loading saved facts…</div>
          : dataset.status === "error" ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-xs text-rust">{dataset.error}</div>
            : <>
                <VirtualFactDraftList dataset={dataset} onError={onError} />
                <button className="ghost-button min-h-9 justify-self-start px-2 py-1 text-xs" onClick={() => void dataset.append(emptyFactEditDraft())} type="button"><Plus size={14} />Add fact</button>
              </>
      ) : loadingJson ? <div className="rounded-md border border-line bg-panel p-3 text-xs text-ink/60">Loading JSON from local browser storage…</div>
        : <>
            <div className="flex justify-end"><button className="ghost-button min-h-8 px-2 py-1 text-xs" disabled={savingJson || jsonValue === null} onClick={formatJson} type="button">Format JSON</button></div>
            <textarea className="control min-h-64 font-mono text-xs" value={jsonValue ?? ""} onChange={(event) => setJsonValue(event.target.value)} />
          </>}
    </div>
  );
}

function VirtualFactDraftList({ dataset, onError }: { dataset: IndexedFactDataset<FactEditDraft>; onError: (value: string) => void }) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(384);
  const [facts, setFacts] = useState<FactEditDraft[]>([]);
  const visibleCount = Math.max(1, Math.ceil(viewportHeight / VIRTUAL_FACT_CARD_HEIGHT));
  const startIndex = Math.max(0, Math.floor(scrollTop / VIRTUAL_FACT_CARD_HEIGHT) - VIRTUAL_FACT_OVERSCAN);
  const endIndex = Math.min(dataset.count, startIndex + visibleCount + VIRTUAL_FACT_OVERSCAN * 2);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const updateViewport = () => setViewportHeight(container.clientHeight || 384);
    updateViewport();
    const observer = new ResizeObserver(updateViewport);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let active = true;
    void dataset.getRange(startIndex, endIndex)
      .then((values) => {
        if (active) setFacts(values);
      })
      .catch((reason) => {
        if (active) onError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      active = false;
    };
  }, [dataset.getRange, dataset.revision, endIndex, onError, startIndex]);

  if (dataset.count === 0) {
    return <div className="rounded-md border border-dashed border-line bg-panel/50 p-4 text-xs text-ink/60">No facts yet. Add a fact or upload a JSON file.</div>;
  }

  return (
    <div ref={scrollRef} className="max-h-96 overflow-y-auto pr-1" onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}>
      <div className="relative" style={{ height: dataset.count * VIRTUAL_FACT_CARD_HEIGHT }}>
        <div className="absolute inset-x-0 grid gap-3" style={{ transform: `translateY(${startIndex * VIRTUAL_FACT_CARD_HEIGHT}px)` }}>
          {Array.from({ length: Math.max(0, endIndex - startIndex) }, (_, offset) => {
            const index = startIndex + offset;
            const fact = facts[offset];
            return fact ? (
              <VirtualFactDraftCard
                key={index}
                count={dataset.count}
                fact={fact}
                index={index}
                onError={onError}
                onRemove={() => void dataset.remove(index).catch((reason) => onError(reason instanceof Error ? reason.message : String(reason)))}
                onUpdate={(field, value) => void dataset.update(index, (current) => ({ ...current, [field]: value })).catch((reason) => onError(reason instanceof Error ? reason.message : String(reason)))}
              />
            ) : <div key={index} className="animate-pulse border border-line bg-panel" style={{ height: VIRTUAL_FACT_CARD_HEIGHT - 12 }} />;
          })}
        </div>
      </div>
    </div>
  );
}

function VirtualFactDraftCard({
  count,
  fact,
  index,
  onError,
  onRemove,
  onUpdate
}: {
  count: number;
  fact: FactEditDraft;
  index: number;
  onError: (value: string) => void;
  onRemove: () => void;
  onUpdate: (field: FactEditableField, value: string) => void;
}) {
  const [value, setValue] = useState(fact);

  useEffect(() => {
    setValue(fact);
  }, [fact]);

  function update(field: FactEditableField, fieldValue: string) {
    setValue((current) => ({ ...current, [field]: fieldValue }));
    onError("");
    onUpdate(field, fieldValue);
  }

  return (
    <section className="grid gap-3 border border-line bg-panel p-3" style={{ height: VIRTUAL_FACT_CARD_HEIGHT - 12 }}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium">Fact {index + 1}</span>
        <button aria-label={`Remove fact ${index + 1}`} className="ghost-button grid size-8 place-items-center p-0" disabled={count === 1} onClick={onRemove} title="Remove fact" type="button"><Trash2 size={15} /></button>
      </div>
      <FactInput label="Subject" value={value.subject} onChange={(fieldValue) => update("subject", fieldValue)} />
      <FactInput label="Relation Prompt" value={value.relation_prompt} onChange={(fieldValue) => update("relation_prompt", fieldValue)} />
      <FactInput label="New Object" value={value.new_object} onChange={(fieldValue) => update("new_object", fieldValue)} />
      <FactInput label="Ground Truth" value={value.ground_truth} onChange={(fieldValue) => update("ground_truth", fieldValue)} />
    </section>
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
  const batchDataset = useIndexedFactDataset({
    datasetId: "kew:edit:memit:batch-facts",
    initialValues: DEFAULT_FACT_EDIT_DRAFTS,
    legacyStorageKey: "kew:edit:memit:edits",
    normalize: normalizeFactEditDraft
  });
  const selectedModel = useMemo(() => memitModels.find((model) => model.id === modelId), [memitModels, modelId]);
  const selectedModelId = selectedModel?.id;
  const maxLayer = Math.max(0, (selectedModel?.n_layers ?? 1) - 1);
  const layers = layersByModel[modelId] ?? "";
  const selectedLayers = parseMemitLayers(layers, maxLayer);
  const easyEditDefaultLayerValues = easyEditDefaultLayers[modelId];
  const modelLoaded = runtime.some((runtimeStatus) => runtimeStatus.model_id === modelId && runtimeStatus.loaded);
  const runtimeConnected = useRuntimeConnection();
  const evaluationPrompts = useFactEvaluationPromptPreview(batchDataset, true);

  useLoadedModelSelection(memitModels, runtime, setModelId);

  useEffect(() => {
    if (memitModels.length === 0) return;
    if (!memitModels.some((model) => model.id === modelId)) {
      setModelId(memitModels[0].id);
    }
  }, [memitModels, modelId, setModelId]);

  useEffect(() => {
    setReliabilityResult(null);
  }, [batchDataset.revision, modelId]);

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
      const parsedEdits = await batchDataset.getAll();
      if (!parsedEdits.length) {
        throw new Error("Batch Edits needs at least one MEMIT fact.");
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
        evaluation_prompts: buildFactEvaluationPrompts(parsedEdits),
        sequential_edit: sequentialEdit,
        mom2_adjustment: mom2Adjustment
      });
      if (!sequentialEdit) await onPatchesChanged?.();
      const completed = await watchEditJob(response, setJob);
      const completedResult = completed.result ?? null;
      setResult(completedResult);
      setReliabilityResult(reliabilityFromEditResult(completedResult));
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
    if (!modelLoaded || batchDataset.count === 0 || busy || testingReliability) return;
    setTestingReliability(true);
    setError("");
    setReliabilityResult(null);
    setReliabilityTestProgress(null);
    reliabilityTestCancellation.reset();
    try {
      const edits = await batchDataset.getAll();
      const response = await api.easyEditReliability({
        model_id: modelId,
        method: "memit",
        edits: edits.map((edit) => ({
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

  return (
    <div className="grid gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
      <section className="panel grid gap-4">
        <ModelSelector models={memitModels} value={modelId} onChange={setModelId} onRuntimeUnloaded={onPatchesChanged} />
        <FactDatasetEditor
          label="Batch Edits"
          dataset={batchDataset}
          editorMode={inputMode}
          onEditorModeChange={setInputMode}
          onError={setError}
          uploadTitle="MEMIT edits JSON"
        />
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
            <span>EasyEdit default: {runtimeConnected ? easyEditDefaultLayerValues?.join(",") ?? "Loading..." : "Unknown"}</span>
          </div>
        </label>
        <DerivedEvaluationPrompts
          prompts={evaluationPrompts}
          totalPromptCount={batchDataset.count}
          onTest={testReliability}
          onCancel={() => void reliabilityTestCancellation.cancel().catch((error) => setError(error instanceof Error ? error.message : String(error)))}
          reliabilityResult={reliabilityResult}
          testProgress={reliabilityTestProgress}
          testing={testingReliability}
          testCancelable={Boolean(reliabilityTestCancellation.jobId)}
          testCanceling={reliabilityTestCancellation.canceling}
          testDisabled={busy || !modelLoaded || batchDataset.status !== "ready" || batchDataset.count === 0}
        />
        <label className="flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
          <span>Continue from active patches</span>
          <input type="checkbox" checked={sequentialEdit} onChange={(event) => setSequentialEdit(event.target.checked)} title="Keep the active patch lineage for this edit" />
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
        <button className="tool-button w-full" onClick={run} disabled={busy || !modelLoaded || batchDataset.status !== "ready" || !selectedLayers?.length} title={modelLoaded ? "Apply MEMIT edit at the selected layers" : "Load the selected model before editing"}>
          <Layers size={16} />
          {busy ? "Editing" : "Apply MEMIT"}
        </button>
        <ExportActiveModelControl modelId={modelId} modelLoaded={modelLoaded} disabled={busy} />
        {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
      <ResultPanel result={result} job={job} onPatchesChanged={onPatchesChanged} />
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

function useFactEvaluationPromptPreview(dataset: IndexedFactDataset<FactEditDraft>, active: boolean) {
  const [prompts, setPrompts] = useState<string[]>([]);

  useEffect(() => {
    if (!active || dataset.status !== "ready" || dataset.count === 0) {
      setPrompts([]);
      return;
    }
    let current = true;
    void dataset.getRange(0, Math.min(dataset.count, 20))
      .then((facts) => {
        if (current) setPrompts(buildFactEvaluationPrompts(facts));
      })
      .catch(() => {
        if (current) setPrompts([]);
      });
    return () => {
      current = false;
    };
  }, [active, dataset.count, dataset.getRange, dataset.revision, dataset.status]);

  return prompts;
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

const RELIABILITY_TEST_PAGE_SIZE = 25;

function useReliabilityTestResultPages(
  jobId: string | null,
  itemCount: number,
  configuredPageSize: number | null,
  refreshKey: number
) {
  const pageSize = configuredPageSize && configuredPageSize > 0 ? configuredPageSize : RELIABILITY_TEST_PAGE_SIZE;
  const [pages, setPages] = useState<Record<number, ReliabilityTestResultPage>>({});
  const requestKeys = useRef(new Set<string>());
  const wantedOffsets = useRef(new Set<number>());
  const generation = useRef(0);

  useEffect(() => {
    generation.current += 1;
    requestKeys.current.clear();
    wantedOffsets.current.clear();
    setPages({});
  }, [jobId]);

  const ensureRange = useCallback((start: number, end: number) => {
    if (!jobId || itemCount <= 0 || end < start) return;
    const firstVisiblePage = Math.floor(Math.max(0, start) / pageSize) * pageSize;
    const lastVisiblePage = Math.floor(Math.min(itemCount - 1, Math.max(start, end)) / pageSize) * pageSize;
    const firstWantedPage = Math.max(0, firstVisiblePage - pageSize);
    const lastWantedPage = Math.min(
      Math.floor((itemCount - 1) / pageSize) * pageSize,
      lastVisiblePage + pageSize
    );
    const nextOffsets = new Set<number>();
    for (let offset = firstWantedPage; offset <= lastWantedPage; offset += pageSize) nextOffsets.add(offset);
    wantedOffsets.current = nextOffsets;

    setPages((current) => Object.fromEntries(
      Object.entries(current).filter(([offset]) => nextOffsets.has(Number(offset)))
    ));

    const currentGeneration = generation.current;
    for (const offset of nextOffsets) {
      const requestKey = `${jobId}:${offset}`;
      if (requestKeys.current.has(requestKey)) continue;
      requestKeys.current.add(requestKey);
      void api.easyEditReliabilityPage(jobId, offset)
        .then((page) => {
          if (generation.current !== currentGeneration || !wantedOffsets.current.has(offset)) return;
          setPages((current) => {
            const retained = Object.fromEntries(
              Object.entries(current).filter(([currentOffset]) => wantedOffsets.current.has(Number(currentOffset)))
            );
            return { ...retained, [offset]: page };
          });
        })
        .catch(() => undefined)
        .finally(() => requestKeys.current.delete(requestKey));
    }
  }, [itemCount, jobId, pageSize, refreshKey]);

  const getItem = useCallback((index: number) => {
    const page = pages[Math.floor(index / pageSize) * pageSize];
    return page?.items.find((item) => item.index === index);
  }, [pageSize, pages]);

  return { ensureRange, getItem };
}

function DerivedEvaluationPrompts({
  prompts,
  totalPromptCount,
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
  totalPromptCount?: number;
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
  const hasAdditionalPrompts = totalPromptCount !== undefined && totalPromptCount > prompts.length;

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
        value={hasAdditionalPrompts ? `${prompts.join("\n")}\n…` : prompts.join("\n")}
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
    method,
    targetLayer,
    evaluationPrompts,
    sequentialEdit,
    mom2Adjustment
  }: {
    modelId: string;
    method: "rome" | "r_rome";
    targetLayer: number;
    evaluationPrompts: string[];
    sequentialEdit: boolean;
    mom2Adjustment: boolean;
  }
) {
  if (!drafts) {
    throw new Error("Sequential Facts JSON is invalid. Use a JSON array of fact objects.");
  }
  if (!drafts.length) {
    throw new Error("Sequential Facts must include at least one fact.");
  }
  const invalidEdit = drafts.findIndex((edit) => !edit.subject.trim() || !edit.relation_prompt.trim() || !edit.new_object.trim());
  if (invalidEdit >= 0) {
    throw new Error(`Fact ${invalidEdit + 1} needs a subject, relation prompt, and new object.`);
  }
  const request = {
    model_id: modelId,
    edits: drafts.map((edit) => ({
      ...edit,
      ground_truth: edit.ground_truth.trim() || null
    })),
    target_layer: targetLayer,
    evaluation_prompts: evaluationPrompts,
    sequential_edit: sequentialEdit,
    mom2_adjustment: mom2Adjustment
  };
  return method === "r_rome" ? api.rRomeSequentialEdits(request) : api.romeSequentialEdits(request);
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

function normalizeFactEditDraft(value: unknown): FactEditDraft {
  if (!value || typeof value !== "object" || Array.isArray(value)) return emptyFactEditDraft();
  const edit = value as Record<string, unknown>;
  return {
    subject: typeof edit.subject === "string" ? edit.subject : "",
    relation_prompt: typeof edit.relation_prompt === "string" ? edit.relation_prompt : "",
    ground_truth: typeof edit.ground_truth === "string" ? edit.ground_truth : "",
    new_object: typeof edit.new_object === "string" ? edit.new_object : "",
    rephrase_prompts: Array.isArray(edit.rephrase_prompts)
      ? edit.rephrase_prompts.filter((prompt): prompt is string => typeof prompt === "string")
      : undefined
  };
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

function ResultPanel({
  result,
  job,
  onPatchesChanged
}: {
  result: Record<string, unknown> | null;
  job: EditJob | null;
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const logsRef = useRef<HTMLPreElement | null>(null);
  const generations = useMemo(() => parseGenerations(result), [result]);
  const reliabilityTestJobId = typeof result?.reliability_test_job_id === "string"
    ? result.reliability_test_job_id
    : null;
  const reliabilityTestItemCount = typeof result?.reliability_test_item_count === "number"
    ? result.reliability_test_item_count
    : 0;
  const reliabilityTestPageSize = typeof result?.reliability_test_page_size === "number"
    ? result.reliability_test_page_size
    : null;
  const reliabilitySummary = result?.reliability;
  const completedFactCount = reliabilitySummary && typeof reliabilitySummary === "object"
    ? (reliabilitySummary as Record<string, unknown>).completed_fact_count
    : undefined;
  const reliabilityTestCompletedCount = typeof completedFactCount === "number" ? completedFactCount : 0;
  const reliabilityTestPages = useReliabilityTestResultPages(
    reliabilityTestJobId,
    reliabilityTestItemCount,
    reliabilityTestPageSize,
    reliabilityTestCompletedCount
  );
  const [canceling, setCanceling] = useState(false);

  useEffect(() => {
    if (!logsRef.current) return;
    logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [job?.logs]);

  useEffect(() => {
    if (!job?.job_id) return;
    return listenForEditJobCanceling((jobId) => {
      if (jobId === job.job_id) setCanceling(true);
    });
  }, [job?.job_id]);

  useEffect(() => {
    if (job?.status !== "running") setCanceling(false);
  }, [job?.status]);

  async function cancel() {
    if (!job || job.status !== "running" || canceling) return;
    setCanceling(true);
    announceEditJobCanceling(job.job_id);
    let cancelAccepted = false;
    try {
      const canceledJob = await api.cancelEditJob(job.job_id);
      if (canceledJob.status !== "canceled") {
        throw new Error("This edit is no longer running.");
      }
      announceExperimentCanceled({ experimentId: canceledJob.experiment_id, jobId: canceledJob.job_id });
      cancelAccepted = true;
      clearActiveEditJob();
      void Promise.resolve(onPatchesChanged?.()).catch(() => undefined);
    } finally {
      if (!cancelAccepted) setCanceling(false);
    }
  }

  return (
    <section className="panel">
      <div className="mb-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="label">Edit Logs</span>
          <div className="flex items-center gap-2">
            {job?.status === "running" ? (
              <button className="ghost-button min-h-8 px-2 py-1 text-xs text-rust" disabled={canceling} onClick={() => void cancel()} title="Stop this edit and unload the active model" type="button">
                {canceling ? <Loader2 className="animate-spin" size={14} /> : <X size={14} />}
                {canceling ? "Canceling" : "Cancel edit"}
              </button>
            ) : null}
            <span className="font-mono text-xs text-ink/50">{job?.status ?? "idle"}</span>
          </div>
        </div>
        <EditJobProgress job={job} />
        <pre ref={logsRef} className="max-h-72 min-h-40 overflow-auto rounded-md border border-line bg-ink p-3 text-xs leading-5 text-white">
          {job?.logs.length ? job.logs.join("\n") : "No edit job is running."}
        </pre>
      </div>
      <div className="mb-2 flex items-center gap-2">
        <Save size={16} />
        <span className="label">Generations</span>
      </div>
      <div className={reliabilityTestJobId ? "rounded-md border border-line bg-panel p-1" : "min-h-96 rounded-md border border-line bg-panel p-4"}>
        {reliabilityTestJobId ? (
          reliabilityTestItemCount > 0 ? (
            <VirtualizedList
              className="h-[32rem] overflow-y-auto overscroll-contain p-3"
              estimatedItemHeight={150}
              getItem={reliabilityTestPages.getItem}
              itemCount={reliabilityTestItemCount}
              onVisibleRangeChange={reliabilityTestPages.ensureRange}
              renderItem={(item) => <ReliabilityTestGenerationCard item={item} />}
            />
          ) : (
            <div className="min-h-40 p-3 text-sm text-ink/60">Waiting for the first generation…</div>
          )
        ) : generations.length ? (
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

function ReliabilityTestGenerationCard({ item }: { item: ReliabilityTestResultItem }) {
  const prompt = item.generation?.prompt ?? item.case?.prompt ?? "";
  const generation = item.generation?.active_model ?? "";
  const target = item.case?.target_new ?? "";
  const score = item.case?.reliability_score;

  return (
    <div className="grid gap-2 rounded-md border border-line bg-surface p-3">
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-xs text-ink/50">Fact {item.index + 1}</span>
        {typeof score === "number" ? <span className="shrink-0 font-mono text-xs text-cyan">rewrite_acc {formatReliabilityScore(score)}</span> : <span className="shrink-0 font-mono text-xs text-ink/45">Evaluating…</span>}
      </div>
      <div className="font-mono text-xs text-ink/50">{prompt}</div>
      {generation ? (
        <div className="whitespace-pre-wrap break-words text-sm leading-6 text-ink">
          <HighlightedGeneration text={generation} target={target} />
        </div>
      ) : (
        <div className="h-6 animate-pulse rounded bg-line/60" />
      )}
    </div>
  );
}

function EditJobProgress({ job }: { job: EditJob | null }) {
  if (!job || job.status !== "running") return null;
  const completed = job.progress_current ?? 0;
  const total = job.progress_total ?? 0;
  const determinate = total > 0;
  const percent = Math.max(0, Math.min(100, Math.round((determinate ? completed / total : job.progress) * 100)));
  const phase = job.progress >= 0.1 ? "Editing" : "Preparing";

  return (
    <div className="mb-2 grid gap-1.5" aria-label={determinate ? `${completed} of ${total} sequential edits complete` : `${phase} edit`} role="status">
      <div className="h-1.5 overflow-hidden rounded-full bg-line">
        <div className={determinate ? "h-full bg-cyan transition-[width] duration-300" : "edit-progress-indeterminate h-full bg-cyan"} style={determinate ? { width: `${percent}%` } : undefined} />
      </div>
      <span className="font-mono text-xs text-ink/60">{determinate ? `${completed} / ${total} facts (${percent}%)` : phase}</span>
    </div>
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

function reliabilityFromEditResult(result: Record<string, unknown> | null) {
  const value = result?.reliability;
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
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
