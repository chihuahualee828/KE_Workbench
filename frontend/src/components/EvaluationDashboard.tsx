"use client";

import { BarChart3, FlaskConical, Loader2, SlidersHorizontal, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { getBackendConnection } from "@/lib/backendConnection";
import { announceExperimentCanceled, announceExperimentStarted, listenForExperimentCanceled } from "@/lib/experimentEvents";
import { readFactOverrides, writeFactOverride } from "@/lib/factDataset";
import type { EvaluationFact, EvaluationResultItem, EvaluationResultPage, EvaluationResultOverview, ExperimentDetails, ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelCacheManager } from "./ModelCacheManager";
import { ModelSelector } from "./ModelSelector";
import { PatchHistory } from "./PatchHistory";
import { EvaluationProgressIndicator } from "./EvaluationProgressIndicator";
import { VirtualizedList } from "./VirtualizedList";
import { useWorkbenchSession } from "./WorkbenchSession";

type EvaluationMode = "reliability" | "generalization" | "locality" | "portability";
type EvaluationGenerationMetric = Exclude<EvaluationMode, "portability">;

const EVALUATION_METRICS: EvaluationMode[] = ["reliability", "generalization", "locality", "portability"];
const PAGE_SIZE = 50;
const DEFAULT_SELECTED_METRICS: EvaluationMode[] = ["reliability"];

type StoredEvaluationFact = EvaluationFact & { rephrase_override?: string };

const EVALUATION_FACT_DATASET_PREFIX = "kew:evaluate:active-facts:session:";
const EVALUATION_FACT_SESSION_KEY = "kew:evaluate:active-facts:session-id";
let evaluationFactSessionId: string | null = null;

function getEvaluationFactSessionId(): string {
  if (typeof window === "undefined") return "server";
  if (evaluationFactSessionId) return evaluationFactSessionId;
  try {
    const current = window.sessionStorage.getItem(EVALUATION_FACT_SESSION_KEY);
    if (current) {
      evaluationFactSessionId = current;
      return current;
    }
    const next = window.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    window.sessionStorage.setItem(EVALUATION_FACT_SESSION_KEY, next);
    evaluationFactSessionId = next;
    return next;
  } catch {
    evaluationFactSessionId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    return evaluationFactSessionId;
  }
}

function defaultRephrasePrompt(fact: StoredEvaluationFact) {
  return fact.rephrase_prompt?.trim() || fact.prompt;
}

function isCurrentEvaluationExperiment(experiment: ExperimentDetails, modelId: string, activePatchSignature: string) {
  const activePatchIds = experiment.input_json?.active_patch_ids;
  return experiment.type === "evaluation"
    && experiment.input_json?.source === "evaluation_page"
    && experiment.model_id === modelId
    && Array.isArray(activePatchIds)
    && activePatchIds.every((patchId): patchId is string => typeof patchId === "string")
    && JSON.stringify(activePatchIds) === activePatchSignature;
}

export function EvaluationDashboard({ models, patches, runtime = [], apiStatus, onPatchesChanged }: {
  models: SupportedModel[];
  patches: Patch[];
  runtime?: ModelRuntimeStatus[];
  apiStatus: "loading" | "ready" | "offline";
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const [modelId, setModelId] = usePersistentState("kew:evaluate:modelId", "gpt2-xl");
  const [selectedMetrics, setSelectedMetrics] = usePersistentState<EvaluationMode[]>("kew:evaluate:selectedMetrics", DEFAULT_SELECTED_METRICS);
  const [editingRephrase, setEditingRephrase] = useState<{ index: number; key: string; value: string } | null>(null);
  const { evaluationExperimentId, setEvaluationExperimentId } = useWorkbenchSession();
  const [overview, setOverview] = useState<EvaluationResultOverview | null>(null);
  const [busy, setBusy] = useState(false);
  const [canceling, setCanceling] = useState(false);
  const [evaluationJobId, setEvaluationJobId] = useState<string | null>(null);
  const [evaluationProgress, setEvaluationProgress] = useState<{ current: number; total: number } | null>(null);
  const [error, setError] = useState("");
  const [rawDataOpen, setRawDataOpen] = useState(false);
  const [rawData, setRawData] = useState<Record<string, unknown> | null>(null);
  const [rawDataLoading, setRawDataLoading] = useState(false);
  const [generationMetric, setGenerationMetric] = useState<EvaluationGenerationMetric>("reliability");
  const [focusedResultIndex, setFocusedResultIndex] = useState<number | null>(null);
  const [factSearch, setFactSearch] = useState("");
  const [factSearchTotal, setFactSearchTotal] = useState(0);
  const [factSearchPosition, setFactSearchPosition] = useState(-1);
  const [focusedFactIndex, setFocusedFactIndex] = useState<number | null>(null);
  const modelLoaded = runtime.some((status) => status.model_id === modelId && status.loaded);
  const activePatchIds = useMemo(() => patches
    .filter((patch) => patch.active && patch.base_model === modelId)
    .sort((left, right) => (left.active_index ?? Number.MAX_SAFE_INTEGER) - (right.active_index ?? Number.MAX_SAFE_INTEGER))
    .map((patch) => patch.patch_id), [modelId, patches]);
  const activePatchSignature = JSON.stringify(activePatchIds);
  const activePatchMethod = useMemo(() => patches
    .filter((patch) => patch.active && patch.base_model === modelId)
    .sort((left, right) => (left.active_index ?? Number.MAX_SAFE_INTEGER) - (right.active_index ?? Number.MAX_SAFE_INTEGER))
    .at(-1)?.method ?? "", [modelId, patches]);
  const facts = useActiveFactDataset(modelId, modelLoaded, activePatchSignature, activePatchMethod);
  const resultPages = useEvaluationResultPages(evaluationExperimentId, overview?.total ?? 0);
  const selectedMetricOrder = EVALUATION_METRICS.filter((metric) => selectedMetrics.includes(metric));
  const needsEditQuality = selectedMetrics.includes("reliability") || selectedMetrics.includes("generalization");
  const hasRunnableMetric = selectedMetrics.some((metric) => metric === "locality" || metric === "portability" || facts.total > 0);
  const selectedMetricsHaveInputs = selectedMetrics.every((metric) => metric === "locality" || metric === "portability" || facts.total > 0);
  const resultIncludesReliability = overview?.evaluation_modes.includes("reliability") ?? false;
  const resultIncludesGeneralization = overview?.evaluation_modes.includes("generalization") ?? false;
  const resultIncludesLocality = overview?.evaluation_modes.includes("locality") ?? false;
  const availableGenerationMetrics = overview?.available_generation_metrics ?? [];
  const activeGenerationMetric = availableGenerationMetrics.includes(generationMetric) ? generationMetric : availableGenerationMetrics[0] ?? "reliability";

  useLoadedModelSelection(models, runtime, setModelId);

  useEffect(() => listenForExperimentCanceled(({ experimentId, jobId }) => {
    if (experimentId !== evaluationExperimentId && jobId !== evaluationJobId) return;
    setBusy(false);
    setCanceling(false);
    setEvaluationProgress(null);
    setOverview(null);
    setEvaluationExperimentId(null);
    setEvaluationJobId(null);
    setError("");
  }), [evaluationExperimentId, evaluationJobId, setEvaluationExperimentId]);

  useEffect(() => {
    setFocusedResultIndex(null);
    setFocusedFactIndex(null);
    setFactSearchTotal(0);
    setFactSearchPosition(-1);
    setRawData(null);
    setRawDataOpen(false);
    setEvaluationProgress(null);
    setEvaluationJobId(null);
    setCanceling(false);
  }, [activePatchSignature, modelId]);

  useEffect(() => {
    setFactSearchPosition(-1);
  }, [factSearch, facts.sourceKey]);

  useEffect(() => {
    const query = factSearch.trim();
    if (!query || !facts.ready) {
      setFactSearchTotal(0);
      setFocusedFactIndex(null);
      return;
    }
    let active = true;
    const timer = setTimeout(() => {
      void api.activeEvaluationFacts(modelId, Math.max(0, factSearchPosition), 1, query).then((page) => {
        if (!active) return;
        setFactSearchTotal(page.match_total);
        setFocusedFactIndex(page.matches[0] ?? null);
      }).catch((error) => { if (active) setError(String(error)); });
    }, 200);
    return () => { active = false; clearTimeout(timer); };
  }, [factSearch, factSearchPosition, facts.ready, facts.sourceKey, modelId]);

  useEffect(() => {
    if (!evaluationExperimentId) {
      setOverview(null);
      setBusy(false);
      setEvaluationProgress(null);
      setEvaluationJobId(null);
      return;
    }
    // Wait for the shared patch/runtime state before comparing an evaluation's
    // saved active stack.  Otherwise a browser refresh briefly looks like a
    // clean model and would discard the tab-scoped experiment id.
    if (apiStatus !== "ready") return;
    let cancelled = false;
    let retryTimer: number | null = null;
    const restoreEvaluation = async () => {
      try {
        const experiment = await api.experimentStatus(evaluationExperimentId);
        if (cancelled) return;
        if (!isCurrentEvaluationExperiment(experiment, modelId, activePatchSignature)) {
          setEvaluationExperimentId(null);
          setOverview(null);
          setBusy(false);
          setEvaluationProgress(null);
          setEvaluationJobId(null);
          return;
        }
        if (experiment.status === "running") {
          setBusy(true);
          setEvaluationJobId(experiment.job?.id ?? null);
          setOverview(null);
          setEvaluationProgress({
            current: experiment.job?.progress_current ?? 0,
            total: experiment.job?.progress_total ?? 0,
          });
          retryTimer = window.setTimeout(() => void restoreEvaluation(), 1500);
          return;
        }
        setBusy(false);
        setEvaluationProgress(null);
        if (experiment.status === "canceled") {
          setEvaluationExperimentId(null);
          setEvaluationJobId(null);
          setOverview(null);
          setCanceling(false);
          setError("");
          return;
        }
        if (experiment.status !== "completed") {
          setEvaluationExperimentId(null);
          setEvaluationJobId(null);
          setOverview(null);
          setError(experiment.error_message || "Evaluation did not complete.");
          return;
        }
        const nextOverview = await api.evaluationResultOverview(evaluationExperimentId);
        if (cancelled) return;
        setOverview(nextOverview);
        setGenerationMetric(nextOverview.available_generation_metrics[0] ?? "reliability");
      } catch (err) {
        if (!cancelled) {
          setBusy(false);
          setEvaluationProgress(null);
          setError(err instanceof Error ? err.message : "Unable to restore the evaluation result.");
        }
      }
    };
    void restoreEvaluation();
    return () => {
      cancelled = true;
      if (retryTimer !== null) window.clearTimeout(retryTimer);
    };
  }, [activePatchSignature, apiStatus, evaluationExperimentId, modelId, setEvaluationExperimentId]);

  async function run() {
    if (!modelLoaded || busy || !facts.ready || !selectedMetrics.length || !hasRunnableMetric || !selectedMetricsHaveInputs || (needsEditQuality && !facts.method) || (!needsEditQuality && !activePatchIds.length)) return;
    setError("");
    setCanceling(false);
    setBusy(true);
    setEvaluationProgress(null);
    try {
      const rephraseOverrides = facts.overrides;
      const response = await api.evaluateActiveModel({
        model_id: modelId,
        evaluation_modes: selectedMetricOrder,
        rephrase_overrides: rephraseOverrides,
        include_generations: true,
      });
      announceExperimentStarted(response.experiment_id);
      setEvaluationExperimentId(response.experiment_id);
      setEvaluationJobId(response.job_id);
      setOverview(null);
      setBusy(response.status === "running");
      if (response.status !== "running") setEvaluationProgress(null);
      if (response.result) {
        const nextOverview = response.result as unknown as EvaluationResultOverview;
        setOverview(nextOverview);
        setGenerationMetric(nextOverview.available_generation_metrics[0] ?? "reliability");
      }
      setFocusedResultIndex(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed. Please try again.");
      setBusy(false);
      setEvaluationProgress(null);
    }
  }

  async function cancel() {
    if (!evaluationJobId || !busy || canceling) return;
    setCanceling(true);
    setError("");
    try {
      const canceledJob = await api.cancelEvaluationJob(evaluationJobId);
      if (canceledJob.status !== "canceled") {
        throw new Error("This evaluation is no longer running.");
      }
      announceExperimentCanceled({ experimentId: canceledJob.experiment_id, jobId: canceledJob.job_id });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to cancel evaluation.");
    } finally {
      setCanceling(false);
    }
  }

  async function saveRephrase(index: number, fact: StoredEvaluationFact) {
    if (editingRephrase?.key !== fact.key || editingRephrase.index !== index) return;
    const value = editingRephrase.value.trim();
    const rephraseOverride = value === defaultRephrasePrompt(fact) ? "" : value;
    try { await facts.saveOverride(fact.key, rephraseOverride); }
    catch (error) { setError(String(error)); return; }
    setEditingRephrase(null);
  }

  function focusResult(index: number, metric: EvaluationGenerationMetric = "reliability") {
    setGenerationMetric(metric);
    setFocusedResultIndex(index);
  }

  async function openRawData() {
    if (!evaluationExperimentId) return;
    setRawDataOpen(true);
    if (rawData) return;
    setRawDataLoading(true);
    try {
      setRawData(await api.evaluationResultRaw(evaluationExperimentId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load raw evaluation data.");
      setRawDataOpen(false);
    } finally {
      setRawDataLoading(false);
    }
  }

  return <div className="grid gap-5">
    <div className="grid items-start gap-5 xl:grid-cols-[300px_minmax(0,1fr)]">
      <section className="panel">
        <div className="panel-heading"><h2 className="flex items-center gap-2 text-sm font-semibold"><SlidersHorizontal size={16} className="text-cyan" />Evaluation setup</h2></div>
        <fieldset disabled={busy} className="grid min-w-0 gap-4">
          <ModelSelector models={models} value={modelId} onChange={setModelId} />
          <div className="grid gap-3">
            <div className="inline-flex max-w-full flex-wrap gap-1 rounded-lg border border-line bg-white p-1" role="group" aria-label="Evaluation metrics">
              {EVALUATION_METRICS.map((tab) => <button key={tab} type="button" aria-pressed={selectedMetrics.includes(tab)} className={`rounded-md px-2.5 py-1.5 text-xs font-medium capitalize transition ${selectedMetrics.includes(tab) ? "bg-cyan/10 text-cyan" : "text-ink/55 hover:bg-panel hover:text-ink"}`} onClick={() => setSelectedMetrics((current) => current.includes(tab) ? current.filter((metric) => metric !== tab) : EVALUATION_METRICS.filter((metric) => [...current, tab].includes(metric)))}>{tab}</button>)}
            </div>
            {selectedMetrics.some((metric) => metric === "reliability" || metric === "generalization") ? <label className="grid gap-1.5"><span className="label">Find an edit fact</span><input className="control" value={factSearch} onChange={(event) => setFactSearch(event.target.value)} onKeyDown={(event) => {
              if (event.key !== "Enter" || !factSearchTotal) return;
              event.preventDefault();
              const next = (Math.max(0, factSearchPosition) + 1) % factSearchTotal;
              setFactSearchPosition(next);
            }} placeholder="Search subject, original value, and edited target" type="search" />{factSearch.trim() ? <span className="text-[11px] text-ink/55">{factSearchTotal ? `${Math.max(0, factSearchPosition) + 1} of ${factSearchTotal} matches · Enter for next` : "No matching edit facts"}</span> : null}</label> : null}
            {selectedMetrics.includes("reliability") ? <EvaluationFactList facts={facts} focusedIndex={focusedFactIndex} title="Original edit prompts" emptyMessage="No active patches are loaded for this model. Load a patch in Patch History to evaluate its edit facts." onFactClick={(resultIncludesReliability || resultIncludesGeneralization) && overview?.total ? (index) => focusResult(index) : undefined} /> : null}
            {selectedMetrics.includes("generalization") ? <ManualRephraseList facts={facts} editing={editingRephrase} focusedIndex={focusedFactIndex} resultAvailable={resultIncludesGeneralization && Boolean(overview?.total)} onClick={(index) => focusResult(index, "generalization")} onEdit={setEditingRephrase} onChange={setEditingRephrase} onSave={saveRephrase} /> : null}
            {selectedMetrics.includes("locality") ? <EvaluationMetricPromptList facts={facts} focusedIndex={focusedFactIndex} promptField="locality_prompt" groundTruthField="locality_ground_truth" promptCount={facts.localityPromptCount} title="Unrelated fact prompts" emptyMessage="No locality prompts are included in the active patches." onFactClick={resultIncludesLocality && Boolean(overview?.total) ? (index) => focusResult(index, "locality") : undefined} /> : null}
            {selectedMetrics.includes("portability") ? <EvaluationMetricPromptList facts={facts} focusedIndex={focusedFactIndex} promptField="portability_prompt" groundTruthField="portability_ground_truth" promptCount={facts.portabilityPromptCount} title="Derived-fact prompts" emptyMessage="No portability prompts are included in the active patches." /> : null}
          </div>
          {!patches.length ? <p className="text-xs leading-5 text-ink/55">Create a knowledge edit first, then select its patch here.</p> : !modelLoaded ? <p className="text-xs leading-5 text-ink/55">Load the selected model to enable evaluation.</p> : null}
          <button className="tool-button" onClick={run} disabled={busy || !modelLoaded || !facts.ready || !selectedMetrics.length || !hasRunnableMetric || !selectedMetricsHaveInputs || (needsEditQuality && !facts.method) || (!needsEditQuality && !activePatchIds.length)} title={modelLoaded ? "Run all highlighted metrics" : "Load the selected model before evaluating"}>{busy ? <Loader2 size={16} className="animate-spin" /> : <BarChart3 size={16} />}{busy ? "Evaluating patch…" : "Run evaluation"}</button>
        </fieldset>
        {error || facts.error ? <div role="alert" className="inline-error mt-4">{error || facts.error}</div> : null}
      </section>
      <section className="panel grid gap-4" aria-busy={busy}>
        <h2 className="flex items-center gap-2 text-sm font-semibold"><FlaskConical size={17} className="text-cyan" />Evaluation results</h2>
        {busy ? <EvaluationProgressIndicator cancelDisabled={!evaluationJobId} canceling={canceling} current={evaluationProgress?.current} multiFactLabel="Running highlighted metrics…" onCancel={() => void cancel()} preparingLabel={needsEditQuality ? "Preparing EasyEdit…" : "Preparing evaluation…"} singleFactLabel="Evaluating fact…" total={evaluationProgress?.total} /> : null}
        {overview ? <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{Object.entries(overview.metrics).map(([key, value]) => {
            const scoreGenerationMetric: EvaluationGenerationMetric | null = key === "reliability_rewrite_acc" ? "reliability" : key === "generalization_rephrase_acc" ? "generalization" : key === "locality_preservation_acc" ? "locality" : null;
            const canSelectGeneration = scoreGenerationMetric !== null && availableGenerationMetrics.includes(scoreGenerationMetric);
            const selected = canSelectGeneration && activeGenerationMetric === scoreGenerationMetric;
            return <button key={key} type="button" disabled={!canSelectGeneration} aria-pressed={canSelectGeneration ? selected : undefined} title={canSelectGeneration ? `Show ${scoreGenerationMetric} prompt generations below` : undefined} onClick={() => scoreGenerationMetric && setGenerationMetric(scoreGenerationMetric)} className={`rounded-md border p-3 text-left transition ${canSelectGeneration ? "cursor-pointer hover:border-cyan/60 focus:outline-none focus:ring-2 focus:ring-cyan/40" : "cursor-default"} ${selected ? "border-cyan bg-cyan/5" : "border-line bg-panel"}`}><div className="label">{key.replace(/_/g, " ")}</div><div className="mt-2 text-2xl font-semibold tabular-nums">{typeof value === "number" ? value.toFixed(3) : value == null ? "—" : String(value)}</div></button>;
          })}</div>
          {Object.keys(overview.metrics).some((key) => key.startsWith("portability_")) ? <p className="text-xs leading-5 text-ink/55">Portability scores are temporary placeholders and are not formal metrics yet.</p> : null}
          {overview.total ? <VirtualizedList itemCount={overview.total} estimatedItemHeight={310} scrollToIndex={focusedResultIndex} className="h-[520px] overflow-y-auto overscroll-contain p-1" getItem={resultPages.getItem} onVisibleRangeChange={resultPages.ensureRange} renderItem={(item, _index, focused) => <EvaluationResultCard item={item} generationMetric={activeGenerationMetric} showReliability={resultIncludesReliability} showGeneralization={resultIncludesGeneralization} showLocality={resultIncludesLocality} focused={focused} />} /> : null}
          <button className="ghost-button justify-self-start text-xs" onClick={() => void openRawData()} type="button">View raw evaluation data</button>
        </> : busy ? <div className="empty-state min-h-[360px] rounded-lg border border-dashed border-line bg-panel/50"><span className="icon-tile bg-white shadow-sm"><Loader2 size={22} className="animate-spin text-cyan" /></span><h3 className="text-sm font-medium">Evaluation is running</h3><p className="max-w-sm text-xs leading-6 text-ink/55">Results will appear here automatically when the evaluation finishes.</p></div> : <div className="empty-state min-h-[360px] rounded-lg border border-dashed border-line bg-panel/50"><span className="icon-tile bg-white shadow-sm"><BarChart3 size={22} strokeWidth={1.5} /></span><h3 className="text-sm font-medium">Put your edit to the test</h3><p className="max-w-sm text-xs leading-6 text-ink/55">Load an edited patch, then run prompts to compare the clean model with the active edited model.</p></div>}
      </section>
    </div>
    <PatchHistory patches={patches} models={models} runtime={runtime} onPatchesChanged={onPatchesChanged} />
    <ModelCacheManager models={models} apiStatus={apiStatus} onModelsChanged={onPatchesChanged} />
    {rawDataOpen ? <RawEvaluationDataDialog result={rawData} loading={rawDataLoading} onClose={() => setRawDataOpen(false)} /> : null}
  </div>;
}

function useActiveFactDataset(modelId: string, enabled: boolean, patchSignature: string, activeMethod: string) {
  const sessionId = getEvaluationFactSessionId();
  const legacyId = `${EVALUATION_FACT_DATASET_PREFIX}${sessionId}:${modelId}:${patchSignature || "none"}`;
  const connection = getBackendConnection();
  const datasetId = `${legacyId}:${connection.baseUrl}`;
  const sourceKey = `${datasetId}:${enabled}`;
  const [pages, setPages] = useState<Record<number, EvaluationFact[]>>({});
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [meta, setMeta] = useState({ total: 0, locality: 0, portability: 0, method: activeMethod, key: "" });
  const [error, setError] = useState("");
  const pending = useRef(new Set<string>());
  const version = useRef(0);
  const ranges = useRef(new Map<string, [number, number]>());
  const loaded = useRef(new Set<number>());
  const wantedPages = useRef(new Set<number>([0]));

  useEffect(() => {
    const current = ++version.current;
    setPages({}); loaded.current.clear(); ranges.current.clear(); pending.current.clear(); wantedPages.current = new Set([0]);
    setMeta({ total: 0, locality: 0, portability: 0, method: activeMethod, key: "" });
    setOverrides({}); setError("");
    if (!enabled) return;
    void (async () => {
      const [page, saved, legacy] = await Promise.all([
        api.activeEvaluationFacts(modelId), readFactOverrides(datasetId), readFactOverrides(legacyId)
      ]);
      if (current !== version.current) return;
      const legacyLocal = readLegacyGeneralizationOverrides();
      const combined = { ...legacyLocal, ...legacy, ...saved };
      await Promise.all(Object.entries({ ...legacyLocal, ...legacy }).filter(([key]) => !(key in saved)).map(([key, value]) => writeFactOverride(datasetId, key, value)));
      if (current !== version.current) return;
      setOverrides(combined);
      setPages({ 0: page.facts }); loaded.current.add(0);
      setMeta({ total: page.total, locality: page.locality_count, portability: page.portability_count, method: page.method, key: sourceKey });
    })().catch((reason) => { if (current === version.current) setError(String(reason)); });
    return () => { version.current += 1; };
  }, [sourceKey, datasetId, legacyId, enabled, modelId, activeMethod]);

  const ensureRange = useCallback((owner: string, start: number, end: number) => {
    if (!enabled || meta.key !== sourceKey || end < start) return;
    ranges.current.set(owner, [start, end]);
    const wanted = new Set<number>();
    ranges.current.forEach(([first, last]) => {
      for (let offset = Math.max(0, Math.floor(first / PAGE_SIZE) * PAGE_SIZE - PAGE_SIZE); offset <= Math.min(meta.total - 1, last + PAGE_SIZE); offset += PAGE_SIZE) wanted.add(offset);
    });
    wantedPages.current = wanted;
    for (const offset of loaded.current) if (!wanted.has(offset)) loaded.current.delete(offset);
    setPages((current) => {
      if (Object.keys(current).every((offset) => wanted.has(Number(offset)))) return current;
      return Object.fromEntries(Object.entries(current).filter(([offset]) => wanted.has(Number(offset))));
    });
    const current = version.current;
    for (const offset of wanted) {
      const key = `${current}:${offset}`;
      if (loaded.current.has(offset) || pending.current.has(key)) continue;
      pending.current.add(key);
      void api.activeEvaluationFacts(modelId, offset).then((page) => {
        if (current !== version.current || !wantedPages.current.has(offset)) return;
        loaded.current.add(offset);
        setPages((previous) => ({ ...previous, [offset]: page.facts }));
      }).catch((reason) => { if (current === version.current) setError(String(reason)); })
        .finally(() => pending.current.delete(key));
    }
  }, [enabled, meta.key, meta.total, modelId, sourceKey]);
  const saveOverride = async (key: string, value: string) => {
    await writeFactOverride(datasetId, key, value);
    setOverrides((current) => ({ ...current, [key]: value }));
  };
  const getItem = useCallback((index: number): StoredEvaluationFact | undefined => {
    const fact = pages[Math.floor(index / PAGE_SIZE) * PAGE_SIZE]?.[index % PAGE_SIZE];
    return fact ? { ...fact, rephrase_override: overrides[fact.key] ?? "" } : undefined;
  }, [pages, overrides]);
  return { getItem, ensureRange, overrides, saveOverride, error, localityPromptCount: meta.locality, method: meta.method, portabilityPromptCount: meta.portability, ready: enabled && meta.key === sourceKey, sourceKey, total: enabled && meta.key === sourceKey ? meta.total : 0 };
}

function readLegacyGeneralizationOverrides(): Record<string, string> {
  try {
    const key = "kew:evaluate:generalizationPrompts";
    const raw = window.localStorage.getItem(key);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Record<string, string> : {};
  } catch {
    return {};
  }
}

function useEvaluationResultPages(experimentId: string | null, total: number) {
  const [pages, setPages] = useState<Record<number, EvaluationResultPage>>({});
  const requestKeyRef = useRef(experimentId);
  const pending = useRef(new Set<number>());
  const loadPage = useCallback(async (offset: number, key = requestKeyRef.current) => {
    if (!experimentId || pending.current.has(offset)) return;
    pending.current.add(offset);
    try {
      const page = await api.evaluationResultPage(experimentId, offset);
      if (requestKeyRef.current !== key) return;
      setPages((current) => ({ ...current, [offset]: page }));
    } finally { pending.current.delete(offset); }
  }, [experimentId]);
  useEffect(() => {
    requestKeyRef.current = experimentId;
    pending.current.clear();
    setPages({});
    if (experimentId && total) void loadPage(0, experimentId);
  }, [experimentId, loadPage, total]);
  const ensureRange = useCallback((start: number, end: number) => {
    if (!experimentId || end < start) return;
    for (let offset = Math.floor(start / PAGE_SIZE) * PAGE_SIZE; offset <= end; offset += PAGE_SIZE) if (!pages[offset]) void loadPage(offset);
  }, [experimentId, loadPage, pages]);
  const getItem = useCallback((index: number) => pages[Math.floor(index / PAGE_SIZE) * PAGE_SIZE]?.items[index % PAGE_SIZE], [pages]);
  return { getItem, ensureRange };
}

function EvaluationFactList({ facts, emptyMessage, focusedIndex, title, onFactClick }: { facts: ReturnType<typeof useActiveFactDataset>; emptyMessage: string; focusedIndex: number | null; title: string; onFactClick?: (index: number) => void }) {
  return <div className="grid gap-2"><span className="label">{title}</span>{facts.total ? <VirtualizedList key={facts.sourceKey} itemCount={facts.total} estimatedItemHeight={84} scrollToIndex={focusedIndex} className="max-h-56 overflow-y-auto overscroll-contain pr-1" getItem={facts.getItem} onVisibleRangeChange={(start, end) => facts.ensureRange(title, start, end)} renderItem={(fact, index, focused) => <button type="button" className={`grid w-full rounded-md border bg-panel p-3 text-left ${focused ? "border-cyan ring-2 ring-cyan/50" : "border-line"} ${onFactClick ? "transition hover:border-cyan/50 hover:bg-cyan/5 focus:outline-none focus:ring-2 focus:ring-cyan/40" : "cursor-default"}`} onClick={() => onFactClick?.(index)} title={onFactClick ? "Show this fact's evaluation result" : undefined}><div className="break-words font-mono text-xs text-ink">{fact.prompt}</div><div className="mt-1 break-words text-xs text-ink/55">Target: <span className="font-medium text-cyan">{fact.new_object}</span></div></button>} /> : <div className="rounded-md border border-line bg-panel p-3 text-xs leading-5 text-ink/60">{emptyMessage}</div>}</div>;
}

function ManualRephraseList({ facts, editing, focusedIndex, resultAvailable, onClick, onEdit, onChange, onSave }: { facts: ReturnType<typeof useActiveFactDataset>; editing: { index: number; key: string; value: string } | null; focusedIndex: number | null; resultAvailable: boolean; onClick: (index: number) => void; onEdit: (value: { index: number; key: string; value: string }) => void; onChange: (value: { index: number; key: string; value: string } | null) => void; onSave: (index: number, fact: StoredEvaluationFact) => Promise<void> }) {
  return <div className="grid gap-2"><span className="label">Manual rephrases</span>{facts.total ? <VirtualizedList key={facts.sourceKey} itemCount={facts.total} estimatedItemHeight={104} scrollToIndex={focusedIndex} className="max-h-56 overflow-y-auto overscroll-contain pr-1" getItem={facts.getItem} onVisibleRangeChange={(start, end) => facts.ensureRange("rephrases", start, end)} renderItem={(fact, index, focused) => <div className={`rounded-md border bg-panel p-3 transition hover:border-cyan/50 hover:bg-cyan/5 ${focused ? "border-cyan ring-2 ring-cyan/50" : "border-line"} ${resultAvailable ? "cursor-pointer" : ""}`} onClick={() => resultAvailable && onClick(index)} onDoubleClick={(event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!editing) onEdit({ index, key: fact.key, value: fact.rephrase_override?.trim() || defaultRephrasePrompt(fact) });
  }}>{editing?.key === fact.key && editing.index === index ? <input autoFocus className="control" value={editing.value} aria-label={`Rephrased prompt for ${fact.prompt}`} onClick={(event) => event.stopPropagation()} onBlur={() => void onSave(index, fact)} onChange={(event) => onEdit({ index, key: fact.key, value: event.target.value })} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); if (event.key === "Escape") onChange(null); }} /> : <div className="break-words font-mono text-xs text-ink">{fact.rephrase_override?.trim() || defaultRephrasePrompt(fact)}</div>}<div className="mt-1 break-words text-xs text-ink/55">Target: <span className="font-medium text-cyan">{fact.new_object}</span></div></div>} /> : <div className="rounded-md border border-line bg-panel p-3 text-xs leading-5 text-ink/60">Load a patch to add rephrases for its edit facts.</div>}</div>;
}

function EvaluationMetricPromptList({ facts, focusedIndex, promptField, groundTruthField, promptCount, title, emptyMessage, onFactClick }: { facts: ReturnType<typeof useActiveFactDataset>; focusedIndex: number | null; promptField: "locality_prompt" | "portability_prompt"; groundTruthField: "locality_ground_truth" | "portability_ground_truth"; promptCount: number; title: string; emptyMessage: string; onFactClick?: (index: number) => void }) {
  return <div className="grid gap-2"><span className="label">{title}</span>{facts.total && promptCount ? <VirtualizedList key={`${facts.sourceKey}:${promptField}`} itemCount={facts.total} estimatedItemHeight={84} scrollToIndex={focusedIndex} className="max-h-56 overflow-y-auto overscroll-contain pr-1" getItem={facts.getItem} onVisibleRangeChange={(start, end) => facts.ensureRange(promptField, start, end)} renderItem={(fact, index, focused) => {
    const prompt = fact[promptField]?.trim() || "";
    const groundTruth = fact[groundTruthField]?.trim() || "";
    const canFocus = Boolean(prompt && onFactClick);
    return <button type="button" disabled={!canFocus} onClick={() => onFactClick?.(index)} title={canFocus ? "Show this fact's locality evaluation result" : undefined} className={`grid w-full rounded-md border bg-panel p-3 text-left ${focused ? "border-cyan ring-2 ring-cyan/50" : "border-line"} ${canFocus ? "cursor-pointer transition hover:border-cyan/50 hover:bg-cyan/5 focus:outline-none focus:ring-2 focus:ring-cyan/40" : "cursor-default"}`}><div className={`min-h-4 break-words font-mono text-xs ${prompt ? "text-ink" : "text-transparent"}`}>{prompt || " "}</div>{groundTruth ? <div className="mt-1 break-words text-xs text-ink/55">Target: <span className="font-medium text-cyan">{groundTruth}</span></div> : null}</button>;
  }} /> : <div className="rounded-md border border-line bg-panel p-3 text-xs leading-5 text-ink/60">{emptyMessage}</div>}</div>;
}

function EvaluationResultCard({ item, generationMetric, showReliability, showGeneralization, showLocality, focused }: { item: EvaluationResultItem; generationMetric: EvaluationGenerationMetric; showReliability: boolean; showGeneralization: boolean; showLocality: boolean; focused: boolean }) {
  const generation = generationMetric === "generalization" ? item.generalization_generation : generationMetric === "locality" ? item.locality_generation : item.reliability_generation;
  const generationLabel = generationMetric === "generalization" ? "Generalization prompt" : generationMetric === "locality" ? "Locality prompt" : "Reliability prompt";
  const hasGenerationToggle = Number(showReliability) + Number(showGeneralization) + Number(showLocality) > 1;
  const fact = item.fact;
  const displayedGroundTruth = generationMetric === "locality" ? fact.locality_ground_truth : fact.ground_truth;
  const highlightedEditedTarget = generationMetric === "locality" ? "" : fact.new_object;
  return <div className={`h-max overflow-hidden rounded-lg border outline-none ${focused ? "border-cyan ring-2 ring-cyan/50" : "border-line"}`}><p className="border-b border-line bg-panel px-4 py-3 text-xs font-medium leading-5">{generation?.prompt ?? fact.prompt}</p><div className="flex flex-wrap gap-x-4 gap-y-1 border-b border-line bg-panel/60 px-4 py-2 text-xs leading-5 text-ink/60">{displayedGroundTruth ? <span>Ground truth: <span className="font-medium text-rust">{displayedGroundTruth}</span></span> : null}{generationMetric !== "locality" ? <span>Edited target: <span className="font-medium text-cyan">{fact.new_object}</span></span> : null}{showReliability && typeof item.reliability_score === "number" ? <span>Rewrite accuracy: <span className="font-medium text-cyan">{item.reliability_score.toFixed(3)}</span></span> : null}{showGeneralization && typeof item.generalization_score === "number" ? <span>Rephrase accuracy: <span className="font-medium text-cyan">{item.generalization_score.toFixed(3)}</span></span> : null}{showLocality && typeof item.locality_score === "number" ? <span>Locality preservation: <span className="font-medium text-cyan">{item.locality_score.toFixed(3)}</span></span> : null}</div><div className="grid divide-y divide-line sm:grid-cols-2 sm:divide-x sm:divide-y-0">{([ ["clean_model", "Original model"], ["active_model", "Edited model"] ] as const).map(([key, label]) => <div key={key} className="p-4"><div className="flex items-center justify-between gap-2"><div className="label">{label}</div>{key === "active_model" && hasGenerationToggle ? <span className="text-[10px] text-ink/50">{generationLabel}</span> : null}</div><p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6"><HighlightedEvaluationText text={String(generation?.[key] ?? "—")} groundTruth={displayedGroundTruth} newObject={highlightedEditedTarget} /></p></div>)}</div></div>;
}

function HighlightedEvaluationText({ text, groundTruth, newObject }: { text: string; groundTruth: string; newObject: string }) {
  const seen = new Set<string>();
  const targets = [{ value: newObject.trim(), label: "Edited target", className: "bg-cyan/15 text-cyan" }, { value: groundTruth.trim(), label: "Ground truth", className: "bg-rust/15 text-rust" }].filter((target) => target.value && !seen.has(target.value.toLocaleLowerCase()) && Boolean(seen.add(target.value.toLocaleLowerCase()))).sort((left, right) => right.value.length - left.value.length);
  if (!targets.length) return text;
  const matcher = new RegExp(`(${targets.map((target) => escapeRegExp(target.value)).join("|")})`, "gi");
  return <>{text.split(matcher).map((part, index) => { const target = targets.find((item) => item.value.toLocaleLowerCase() === part.toLocaleLowerCase()); return target ? <mark key={index} title={target.label} className={`rounded px-0.5 font-medium ${target.className}`}>{part}</mark> : part; })}</>;
}

function escapeRegExp(value: string) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

function RawEvaluationDataDialog({ result, loading, onClose }: { result: Record<string, unknown> | null; loading: boolean; onClose: () => void }) {
  return <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/30 p-4 sm:items-center" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }} role="presentation"><section aria-labelledby="raw-evaluation-data-title" aria-modal="true" className="my-auto w-full max-w-3xl overflow-hidden rounded-lg border border-line bg-white shadow-xl" role="dialog"><div className="flex items-center justify-between gap-4 border-b border-line p-5"><h2 id="raw-evaluation-data-title" className="text-base font-semibold">Raw evaluation data</h2><button aria-label="Close raw evaluation data" className="ghost-button grid size-8 place-items-center p-0" onClick={onClose} title="Close" type="button"><X size={16} /></button></div>{loading ? <div className="flex min-h-40 items-center justify-center gap-2 p-5 text-sm text-ink/60"><Loader2 size={16} className="animate-spin" />Loading raw data…</div> : <pre className="max-h-[min(680px,calc(100vh-10rem))] overflow-auto bg-panel p-5 text-xs leading-5">{JSON.stringify(result, null, 2)}</pre>}</section></div>;
}
