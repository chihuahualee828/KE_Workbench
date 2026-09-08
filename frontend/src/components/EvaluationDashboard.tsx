"use client";

import { BarChart3, FlaskConical, Loader2, SlidersHorizontal, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { announceExperimentStarted } from "@/lib/experimentEvents";
import type { ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { ModelCacheManager } from "./ModelCacheManager";
import { ModelSelector } from "./ModelSelector";
import { PatchHistory } from "./PatchHistory";
import { useWorkbenchSession } from "./WorkbenchSession";

type EvaluationMode = "reliability" | "generalization" | "locality" | "portability";
const EVALUATION_METRICS: EvaluationMode[] = ["reliability", "generalization", "locality", "portability"];
const RESULTS_PER_PAGE = 10;
const EMPTY_GENERALIZATION_PROMPTS: Record<string, string> = {};
const DEFAULT_SELECTED_METRICS: EvaluationMode[] = ["reliability"];

type ActiveFact = {
  key: string;
  subject: string;
  relationPrompt: string;
  prompt: string;
  newObject: string;
  groundTruth: string;
};

export function EvaluationDashboard({
  models,
  patches,
  runtime = [],
  apiStatus,
  onPatchesChanged
}: {
  models: SupportedModel[];
  patches: Patch[];
  runtime?: ModelRuntimeStatus[];
  apiStatus: "loading" | "ready" | "offline";
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const [modelId, setModelId] = usePersistentState("kew:evaluate:modelId", "gpt2-xl");
  const [selectedMetrics, setSelectedMetrics] = usePersistentState<EvaluationMode[]>("kew:evaluate:selectedMetrics", DEFAULT_SELECTED_METRICS);
  const [generalizationPrompts, setGeneralizationPrompts] = usePersistentState<Record<string, string>>("kew:evaluate:generalizationPrompts", EMPTY_GENERALIZATION_PROMPTS);
  const [localityPrompts, setLocalityPrompts] = usePersistentState("kew:evaluate:localityPrompts", "");
  const [portabilityPrompts, setPortabilityPrompts] = usePersistentState("kew:evaluate:portabilityPrompts", "");
  const [editingRephrase, setEditingRephrase] = useState<{ key: string; value: string } | null>(null);
  const { evaluationResult: result, setEvaluationResult: setResult } = useWorkbenchSession();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [rawDataOpen, setRawDataOpen] = useState(false);
  const [generationMetric, setGenerationMetric] = useState<"reliability" | "generalization">("reliability");
  const [resultPage, setResultPage] = useState(0);
  const [pendingResultFocus, setPendingResultFocus] = useState<string | null>(null);
  const resultListRef = useRef<HTMLDivElement>(null);
  const resultRefs = useRef(new Map<string, HTMLDivElement>());
  const modelLoaded = runtime.some((status) => status.model_id === modelId && status.loaded);
  const activePatchesForModel = useMemo(
    () => patches
      .filter((patch) => patch.active && patch.base_model === modelId)
      .sort((left, right) => (right.active_index ?? -1) - (left.active_index ?? -1)),
    [modelId, patches]
  );
  const activePatchIds = useMemo(
    () => patches
      .filter((patch) => patch.active && patch.base_model === modelId)
      .sort((left, right) => (left.active_index ?? Number.MAX_SAFE_INTEGER) - (right.active_index ?? Number.MAX_SAFE_INTEGER))
      .map((patch) => patch.patch_id),
    [modelId, patches]
  );
  const activePatchSignature = JSON.stringify(activePatchIds);
  const evaluationPatchId = activePatchesForModel[0]?.patch_id ?? "";
  const reliabilityMethod = useMemo(() => {
    const method = activePatchesForModel[0]?.method;
    return method === "rome" || method === "r_rome" || method === "memit" || method === "kn" ? method : null;
  }, [activePatchesForModel]);
  const activeFacts = useMemo<ActiveFact[]>(
    () => !modelLoaded ? [] : deduplicateFacts(activePatchesForModel
      .flatMap((patch) => patch.edits.map((edit, index) => ({
        key: `${patch.patch_id}:${index}`,
        subject: edit.subject,
        relationPrompt: edit.relation_prompt,
        prompt: buildPatchPrompt(edit.subject, edit.relation_prompt),
        newObject: edit.new_object,
        groundTruth: edit.ground_truth ?? ""
      })))),
    [activePatchesForModel, modelLoaded]
  );
  const promptsByMetric = useMemo<Record<EvaluationMode, string[]>>(() => ({
    reliability: activeFacts.map((fact) => fact.prompt),
    generalization: activeFacts.map((fact) => generalizationPrompts[fact.key]?.trim() || fact.prompt),
    locality: localityPrompts.split("\n").map((prompt) => prompt.trim()).filter(Boolean),
    portability: portabilityPrompts.split("\n").map((prompt) => prompt.trim()).filter(Boolean),
  }), [activeFacts, generalizationPrompts, localityPrompts, portabilityPrompts]);
  const hasRunnableMetric = selectedMetrics.some((metric) => metric === "locality" || metric === "portability" || promptsByMetric[metric].length > 0);
  const selectedMetricsHaveInputs = selectedMetrics.every((metric) => metric === "locality" || metric === "portability" || promptsByMetric[metric].length > 0);
  const selectedMetricOrder = EVALUATION_METRICS.filter((metric) => selectedMetrics.includes(metric));

  useLoadedModelSelection(models, runtime, setModelId);

  useEffect(() => {
    if (apiStatus !== "ready" || !result) return;
    const resultModelId = typeof result.model_id === "string" ? result.model_id : null;
    const resultPatchIds = Array.isArray(result.active_patch_ids)
      ? result.active_patch_ids.filter((patchId): patchId is string => typeof patchId === "string")
      : [];
    if (resultModelId !== modelId || JSON.stringify(resultPatchIds) !== activePatchSignature) {
      setResult(null);
    }
  }, [activePatchSignature, apiStatus, modelId, result, setResult]);

  async function run() {
    const needsEasyEditQuality = selectedMetrics.includes("reliability") || selectedMetrics.includes("generalization");
    if (!modelLoaded || busy || !selectedMetrics.length || !hasRunnableMetric || !selectedMetricsHaveInputs || (needsEasyEditQuality && !reliabilityMethod) || (!needsEasyEditQuality && !evaluationPatchId)) return;
    setError("");
    setBusy(true);
    if (needsEasyEditQuality) announceExperimentStarted("evaluation");
    try {
      const metricResults: Partial<Record<EvaluationMode, Record<string, unknown>>> = {};
      let easyEditQualityResult: Record<string, unknown> | null = null;
      for (const metric of selectedMetricOrder) {
        if (metric === "reliability" || metric === "generalization") {
          if (!easyEditQualityResult) {
            if (!reliabilityMethod) throw new Error("The active patch method does not support this evaluation.");
            const response = await api.easyEditReliability({
              model_id: modelId,
              method: reliabilityMethod,
              edits: activeFacts.map((fact) => ({
                subject: fact.subject,
                relation_prompt: fact.relationPrompt,
                new_object: fact.newObject,
                ground_truth: fact.groundTruth || null,
                rephrase_prompts: selectedMetrics.includes("generalization")
                  ? [generalizationPrompts[fact.key]?.trim() || fact.prompt]
                  : [],
              })),
            include_generations: true,
            evaluation_modes: selectedMetricOrder,
            });
            announceExperimentStarted(response.experiment_id);
            if (!response.result) throw new Error("Evaluation completed without a result.");
            easyEditQualityResult = response.result;
          }
          metricResults[metric] = metric === "reliability"
            ? {
                metrics: { rewrite_acc: easyEditQualityResult.mean_reliability },
                cases: easyEditQualityResult.cases,
                sample_generations: easyEditQualityResult.sample_generations,
              }
            : {
                metrics: { rephrase_acc: easyEditQualityResult.mean_generalization },
                cases: easyEditQualityResult.cases,
                sample_generations: easyEditQualityResult.sample_rephrase_generations,
              };
        } else {
          metricResults[metric] = {
            metrics: { [`${metric}_score`]: 0 },
            note: `Temporary placeholder: formal ${metric} evaluation is not implemented yet.`,
          };
        }
      }
      const metrics = Object.fromEntries(selectedMetricOrder.flatMap((metric) => {
        const metricResult = metricResults[metric];
        const rawMetrics = metricResult?.metric_name === "rewrite_acc"
          ? { rewrite_acc: metricResult.mean_reliability }
          : (metricResult?.metrics ?? {}) as Record<string, unknown>;
        return Object.entries(rawMetrics).map(([key, value]) => [`${metric}_${key}`, value]);
      }));
      setResult({
        model_id: modelId,
        active_patch_ids: activePatchIds,
        metrics,
        sample_generations: selectedMetrics.includes("reliability")
          ? easyEditQualityResult?.sample_generations ?? []
          : easyEditQualityResult?.sample_rephrase_generations ?? [],
        generation_sets: {
          reliability: selectedMetrics.includes("reliability")
            ? easyEditQualityResult?.sample_generations ?? []
            : [],
          generalization: selectedMetrics.includes("generalization")
            ? easyEditQualityResult?.sample_rephrase_generations ?? []
            : [],
        },
        cases: easyEditQualityResult?.cases ?? [],
        evaluation_modes: selectedMetricOrder,
        evaluation_fact_keys: easyEditQualityResult ? activeFacts.map((fact) => fact.key) : [],
        metric_results: metricResults,
      });
      setGenerationMetric(selectedMetricOrder.includes("reliability") ? "reliability" : "generalization");
      setResultPage(0);
      setPendingResultFocus(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const metrics = (result?.metrics ?? (result?.metric_name === "rewrite_acc" ? { rewrite_acc: result.mean_reliability } : {})) as Record<string, unknown>;
  const generationSets = result?.generation_sets && typeof result.generation_sets === "object" && !Array.isArray(result.generation_sets)
    ? result.generation_sets as Record<string, unknown>
    : {};
  const availableGenerationMetrics = (["reliability", "generalization"] as const).filter((metric) =>
    Array.isArray(generationSets[metric]) && generationSets[metric].length > 0
  );
  const activeGenerationMetric = availableGenerationMetrics.includes(generationMetric)
    ? generationMetric
    : availableGenerationMetrics[0] ?? "reliability";
  const generations = availableGenerationMetrics.length
    ? generationSets[activeGenerationMetric] as Array<Record<string, unknown>>
    : Array.isArray(result?.sample_generations) ? result.sample_generations as Array<Record<string, unknown>> : [];
  const resultPageCount = Math.max(1, Math.ceil(generations.length / RESULTS_PER_PAGE));
  const currentResultPage = Math.min(resultPage, resultPageCount - 1);
  const visibleResultStart = currentResultPage * RESULTS_PER_PAGE;
  const visibleGenerations = generations.slice(visibleResultStart, visibleResultStart + RESULTS_PER_PAGE);
  const reliabilityCases = Array.isArray(result?.cases) ? result.cases as Array<Record<string, unknown>> : [];
  const resultFactKeys = Array.isArray(result?.evaluation_fact_keys) ? result.evaluation_fact_keys.filter((key): key is string => typeof key === "string") : [];
  const resultIncludesReliability = Array.isArray(result?.evaluation_modes)
    ? result.evaluation_modes.includes("reliability")
    : result?.evaluation_mode === "reliability";
  const resultIncludesGeneralization = Array.isArray(result?.evaluation_modes)
    ? result.evaluation_modes.includes("generalization")
    : false;

  useEffect(() => {
    setResultPage((current) => Math.min(current, resultPageCount - 1));
  }, [resultPageCount]);

  function scrollAndFocusResult(factKey: string) {
    const resultCard = resultRefs.current.get(factKey);
    const resultList = resultListRef.current;
    if (!resultCard || !resultList) return;

    const scrollTop = resultList.scrollTop + resultCard.getBoundingClientRect().top - resultList.getBoundingClientRect().top - 8;
    resultList.scrollTo({ top: Math.max(0, scrollTop), behavior: "smooth" });
    resultCard.focus({ preventScroll: true });
  }

  function focusResultForFact(factKey: string) {
    const resultIndex = resultFactKeys.indexOf(factKey);
    if (resultIndex < 0) {
      scrollAndFocusResult(factKey);
      return;
    }
    setResultPage(Math.floor(resultIndex / RESULTS_PER_PAGE));
    setPendingResultFocus(factKey);
  }

  useEffect(() => {
    resultListRef.current?.scrollTo({ top: 0 });
  }, [currentResultPage]);

  useEffect(() => {
    if (!pendingResultFocus) return;
    const frame = window.requestAnimationFrame(() => {
      scrollAndFocusResult(pendingResultFocus);
      setPendingResultFocus(null);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [activeGenerationMetric, pendingResultFocus, resultPage]);

  function focusGeneralizationResultForFact(factKey: string) {
    setGenerationMetric("generalization");
    focusResultForFact(factKey);
  }

  function saveRephrase(fact: ActiveFact) {
    if (editingRephrase?.key !== fact.key) return;
    const value = editingRephrase.value.trim();
    setGeneralizationPrompts((current) => ({
      ...current,
      [fact.key]: value === fact.prompt ? "" : value,
    }));
    setEditingRephrase(null);
  }

  return (
    <div className="grid gap-5">
      <div className="grid items-start gap-5 xl:grid-cols-[300px_minmax(0,1fr)]">
        <section className="panel">
        <div className="panel-heading"><h2 className="flex items-center gap-2 text-sm font-semibold"><SlidersHorizontal size={16} className="text-cyan" />Evaluation setup</h2></div>
        <fieldset disabled={busy} className="grid min-w-0 gap-4">
        <ModelSelector models={models} value={modelId} onChange={setModelId} />
        <div className="grid gap-3">
          <div className="inline-flex max-w-full flex-wrap gap-1 rounded-lg border border-line bg-white p-1" role="group" aria-label="Evaluation metrics">
            {EVALUATION_METRICS.map((tab) => (
              <button
                key={tab}
                type="button"
                aria-pressed={selectedMetrics.includes(tab)}
                className={`rounded-md px-2.5 py-1.5 text-xs font-medium capitalize transition ${selectedMetrics.includes(tab) ? "bg-cyan/10 text-cyan" : "text-ink/55 hover:bg-panel hover:text-ink"}`}
                onClick={() => setSelectedMetrics((current) => current.includes(tab) ? current.filter((metric) => metric !== tab) : EVALUATION_METRICS.filter((metric) => [...current, tab].includes(metric)))}
              >
                {tab}
              </button>
            ))}
          </div>
          {selectedMetrics.includes("reliability") ? (
            <EvaluationFactList
              facts={activeFacts}
              title="Original edit prompts"
              emptyMessage="No active patches are loaded for this model. Load a patch in Patch History to evaluate its edit facts."
              onFactClick={(resultIncludesReliability || resultIncludesGeneralization) && generations.length ? focusResultForFact : undefined}
            />
          ) : null}
          {selectedMetrics.includes("generalization") ? (
            <div className="grid gap-2">
              <span className="label">Manual rephrases</span>
              {activeFacts.length ? (
                <div className="grid max-h-56 gap-2 overflow-y-auto pr-1">
                  {activeFacts.map((fact) => (
                    <div
                      key={fact.key}
                      className={`rounded-md border border-line bg-panel p-3 transition hover:border-cyan/50 hover:bg-cyan/5 ${resultIncludesGeneralization && generations.length ? "cursor-pointer" : ""}`}
                      onClick={() => {
                        if (resultIncludesGeneralization && generations.length) focusGeneralizationResultForFact(fact.key);
                      }}
                      onDoubleClick={() => {
                        if (!editingRephrase) {
                          setEditingRephrase({ key: fact.key, value: generalizationPrompts[fact.key]?.trim() || fact.prompt });
                        }
                      }}
                    >
                      {editingRephrase?.key === fact.key ? (
                        <input
                          autoFocus
                          className="control"
                          value={editingRephrase.value}
                          aria-label={`Rephrased prompt for ${fact.prompt}`}
                          onClick={(event) => event.stopPropagation()}
                          onBlur={() => saveRephrase(fact)}
                          onChange={(event) => setEditingRephrase({ key: fact.key, value: event.target.value })}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") event.currentTarget.blur();
                            if (event.key === "Escape") setEditingRephrase(null);
                          }}
                        />
                      ) : (
                        <>
                          <div className="break-words font-mono text-xs text-ink">{generalizationPrompts[fact.key]?.trim() || fact.prompt}</div>
                          {!generalizationPrompts[fact.key]?.trim() ? <div className="mt-1 text-[11px] text-ink/45">Double-click the prompt to enter a rephrase.</div> : null}
                        </>
                      )}
                      <div className="mt-1 break-words text-xs text-ink/55">Target: <span className="font-medium text-cyan">{fact.newObject}</span></div>
                    </div>
                  ))}
                </div>
              ) : <div className="rounded-md border border-line bg-panel p-3 text-xs leading-5 text-ink/60">Load a patch to add rephrases for its edit facts.</div>}
              <span className="text-[11px] leading-5 text-ink/55">Review rephrases manually. Automatic suggestions from the edited model can contaminate the generalization test.</span>
            </div>
          ) : null}
          {selectedMetrics.includes("locality") ? (
            <label className="grid gap-2">
              <span className="label">Unrelated fact prompts</span>
              <textarea
                className="control min-h-40"
                value={localityPrompts}
                onChange={(event) => setLocalityPrompts(event.target.value)}
              />
              <span className="text-[11px] leading-5 text-ink/55">One prompt per line, up to 10 per evaluation. This compares generations; it does not yet compute a formal locality or portability metric.</span>
            </label>
          ) : null}
          {selectedMetrics.includes("portability") ? (
            <label className="grid gap-2">
              <span className="label">Derived-fact prompts</span>
              <textarea
                className="control min-h-40"
                value={portabilityPrompts}
                onChange={(event) => setPortabilityPrompts(event.target.value)}
              />
              <span className="text-[11px] leading-5 text-ink/55">Temporary placeholder only: formal portability evaluation is not implemented yet.</span>
            </label>
          ) : null}
        </div>
        {!patches.length ? <p className="text-xs leading-5 text-ink/55">Create a knowledge edit first, then select its patch here.</p> : !modelLoaded ? <p className="text-xs leading-5 text-ink/55">Load the selected model to enable evaluation.</p> : null}
        <button className="tool-button" onClick={run} disabled={busy || !modelLoaded || !selectedMetrics.length || !hasRunnableMetric || !selectedMetricsHaveInputs || ((selectedMetrics.includes("reliability") || selectedMetrics.includes("generalization")) && !reliabilityMethod) || (!(selectedMetrics.includes("reliability") || selectedMetrics.includes("generalization")) && !evaluationPatchId)} title={modelLoaded ? "Run all highlighted metrics" : "Load the selected model before evaluating"}>
          {busy ? <Loader2 size={16} className="animate-spin" /> : <BarChart3 size={16} />}
          {busy ? "Evaluating patch…" : "Run evaluation"}
        </button>
        </fieldset>
        {error ? <div role="alert" className="inline-error mt-4">{error}</div> : null}
        </section>
        <section className="panel grid gap-4" aria-busy={busy}>
        <h2 className="flex items-center gap-2 text-sm font-semibold"><FlaskConical size={17} className="text-cyan" />Evaluation results</h2>
        {busy ? <div role="status" className="flex items-center gap-2 text-xs text-cyan"><Loader2 size={14} className="animate-spin" />Running highlighted metrics…</div> : null}
        {result ? <>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {Object.entries(metrics).map(([key, value]) => {
            const scoreGenerationMetric = key === "reliability_rewrite_acc"
              ? "reliability"
              : key === "generalization_rephrase_acc"
                ? "generalization"
                : null;
            const canSelectGeneration = scoreGenerationMetric !== null && availableGenerationMetrics.includes(scoreGenerationMetric);
            const selected = canSelectGeneration && activeGenerationMetric === scoreGenerationMetric;

            return (
              <button
                key={key}
                type="button"
                disabled={!canSelectGeneration}
                aria-pressed={canSelectGeneration ? selected : undefined}
                title={canSelectGeneration ? `Show ${scoreGenerationMetric} prompt generations below` : undefined}
                onClick={() => scoreGenerationMetric && setGenerationMetric(scoreGenerationMetric)}
                className={`rounded-md border p-3 text-left transition ${canSelectGeneration ? "cursor-pointer hover:border-cyan/60 focus:outline-none focus:ring-2 focus:ring-cyan/40" : "cursor-default"} ${selected ? "border-cyan bg-cyan/5" : "border-line bg-panel"}`}
              >
                <div className="label">{key.replace(/_/g, " ")}</div>
                <div className="mt-2 text-2xl font-semibold tabular-nums">{typeof value === "number" ? value.toFixed(3) : String(value)}</div>
              </button>
            );
          })}
        </div>
        {Object.keys(metrics).some((key) => key.startsWith("locality_") || key.startsWith("portability_")) ? <p className="text-xs leading-5 text-ink/55">Locality and portability scores are temporary placeholders and are not formal metrics yet.</p> : null}
        {generations.length > RESULTS_PER_PAGE ? (
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-ink/60">
            <span>Showing {visibleResultStart + 1}–{Math.min(visibleResultStart + RESULTS_PER_PAGE, generations.length)} of {generations.length} facts</span>
            <div className="flex items-center gap-2">
              <button className="ghost-button min-h-8 px-2 py-1 text-xs" type="button" disabled={currentResultPage === 0} onClick={() => setResultPage((current) => Math.max(0, current - 1))}>Previous</button>
              <label className="flex items-center gap-1">Page
                <input
                  key={currentResultPage}
                  aria-label="Jump to result page"
                  className="control h-8 w-14 px-1 py-1 text-center text-xs"
                  defaultValue={currentResultPage + 1}
                  min={1}
                  max={resultPageCount}
                  type="number"
                  onBlur={(event) => {
                    const requestedPage = Number(event.currentTarget.value);
                    if (Number.isInteger(requestedPage)) setResultPage(Math.min(Math.max(requestedPage - 1, 0), resultPageCount - 1));
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") event.currentTarget.blur();
                  }}
                />
                <span>of {resultPageCount}</span>
              </label>
              <button className="ghost-button min-h-8 px-2 py-1 text-xs" type="button" disabled={currentResultPage === resultPageCount - 1} onClick={() => setResultPage((current) => Math.min(resultPageCount - 1, current + 1))}>Next</button>
            </div>
          </div>
        ) : null}
        <div ref={resultListRef} className="grid h-[520px] content-start gap-3 overflow-y-auto overscroll-contain p-1">
        {visibleGenerations.map((generation, pageIndex) => {
          const index = visibleResultStart + pageIndex;
          const prompt = String(generation.prompt ?? "");
          const fact = activeFacts.find((item) => item.key === resultFactKeys[index]) ?? activeFacts.find((item) => item.prompt === prompt);
          const resultKey = fact?.key ?? `result-${index}`;

          return (
            <div
              key={index}
              ref={(node) => {
                if (node) resultRefs.current.set(resultKey, node);
                else resultRefs.current.delete(resultKey);
              }}
              tabIndex={-1}
              className="h-max rounded-lg border border-line outline-none focus:ring-2 focus:ring-cyan/50"
            >
              <p className="border-b border-line bg-panel px-4 py-3 text-xs font-medium leading-5">{prompt}</p>
              {fact ? (
                <div className="flex flex-wrap gap-x-4 gap-y-1 border-b border-line bg-panel/60 px-4 py-2 text-xs leading-5 text-ink/60">
                  {fact.groundTruth ? <span>Ground truth: <span className="font-medium text-rust">{fact.groundTruth}</span></span> : null}
                  <span>Edited target: <span className="font-medium text-cyan">{fact.newObject}</span></span>
                  {resultIncludesReliability && typeof reliabilityCases[index]?.reliability_score === "number" ? <span>Rewrite accuracy: <span className="font-medium text-cyan">{Number(reliabilityCases[index].reliability_score).toFixed(3)}</span></span> : null}
                  {resultIncludesGeneralization && typeof reliabilityCases[index]?.generalization_score === "number" ? <span>Rephrase accuracy: <span className="font-medium text-cyan">{Number(reliabilityCases[index].generalization_score).toFixed(3)}</span></span> : null}
                </div>
              ) : null}
              <div className="grid divide-y divide-line sm:grid-cols-2 sm:divide-x sm:divide-y-0">
                {([
                  ["clean_model", "Original model"],
                  ["active_model", "Edited model"]
                ] as const).map(([key, label]) => (
                  <div key={key} className="p-4">
                    <div className="flex items-center justify-between gap-2">
                      <div className="label">{label}</div>
                      {key === "active_model" && availableGenerationMetrics.length > 1 ? <span className="text-[10px] text-ink/50">{activeGenerationMetric === "reliability" ? "Reliability prompt" : "Generalization prompt"}</span> : null}
                    </div>
                    <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6">
                      <HighlightedEvaluationText
                        text={String(generation[key] ?? "—")}
                        groundTruth={fact?.groundTruth ?? ""}
                        newObject={fact?.newObject ?? ""}
                      />
                    </p>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
        </div>
        <button className="ghost-button justify-self-start text-xs" onClick={() => setRawDataOpen(true)} type="button">View raw evaluation data</button>
        </> : <div className="empty-state min-h-[360px] rounded-lg border border-dashed border-line bg-panel/50"><span className="icon-tile bg-white shadow-sm"><BarChart3 size={22} strokeWidth={1.5} /></span><h3 className="text-sm font-medium">Put your edit to the test</h3><p className="max-w-sm text-xs leading-6 text-ink/55">Load an edited patch, then run prompts to compare the clean model with the active edited model.</p></div>}
        </section>
      </div>
      <PatchHistory
        patches={patches}
        models={models}
        runtime={runtime}
        onPatchesChanged={onPatchesChanged}
      />
      <ModelCacheManager models={models} apiStatus={apiStatus} onModelsChanged={onPatchesChanged} />
      {rawDataOpen && result ? <RawEvaluationDataDialog result={result} onClose={() => setRawDataOpen(false)} /> : null}
    </div>
  );
}

function EvaluationFactList({
  facts,
  emptyMessage,
  title,
  onFactClick
}: {
  facts: ActiveFact[];
  emptyMessage: string;
  title: string;
  onFactClick?: (factKey: string) => void;
}) {
  return (
    <div className="grid gap-2">
      <span className="label">{title}</span>
      {facts.length ? (
        <div className="grid max-h-56 gap-2 overflow-y-auto pr-1">
          {facts.map((fact) => (
            <button
              key={fact.key}
              type="button"
              className={`grid rounded-md border border-line bg-panel p-3 text-left ${onFactClick ? "transition hover:border-cyan/50 hover:bg-cyan/5 focus:outline-none focus:ring-2 focus:ring-cyan/40" : "cursor-default"}`}
              onClick={() => onFactClick?.(fact.key)}
              title={onFactClick ? "Show this fact's evaluation result" : undefined}
            >
              <div className="break-words font-mono text-xs text-ink">{fact.prompt}</div>
              <div className="mt-1 break-words text-xs text-ink/55">Target: <span className="font-medium text-cyan">{fact.newObject}</span></div>
            </button>
          ))}
        </div>
      ) : <div className="rounded-md border border-line bg-panel p-3 text-xs leading-5 text-ink/60">{emptyMessage}</div>}
    </div>
  );
}

function buildPatchPrompt(subject: string, relationPrompt: string) {
  return relationPrompt.includes("{}") ? relationPrompt.replace("{}", subject) : `${subject} ${relationPrompt}`.trim();
}

function deduplicateFacts(facts: ActiveFact[]) {
  const identities = new Set<string>();
  return facts.filter((fact) => {
    const identity = JSON.stringify([fact.prompt, fact.newObject]);
    if (identities.has(identity)) return false;
    identities.add(identity);
    return true;
  });
}

function HighlightedEvaluationText({
  text,
  groundTruth,
  newObject
}: {
  text: string;
  groundTruth: string;
  newObject: string;
}) {
  const seen = new Set<string>();
  const targets = [
    { value: newObject.trim(), label: "Edited target", className: "bg-cyan/15 text-cyan" },
    { value: groundTruth.trim(), label: "Ground truth", className: "bg-rust/15 text-rust" }
  ]
    .filter((target) => target.value && !seen.has(target.value.toLocaleLowerCase()) && Boolean(seen.add(target.value.toLocaleLowerCase())))
    .sort((left, right) => right.value.length - left.value.length);

  if (!targets.length) return text;

  const matcher = new RegExp(`(${targets.map((target) => escapeRegExp(target.value)).join("|")})`, "gi");
  return <>{text.split(matcher).map((part, index) => {
    const target = targets.find((item) => item.value.toLocaleLowerCase() === part.toLocaleLowerCase());
    return target ? <mark key={index} title={target.label} className={`rounded px-0.5 font-medium ${target.className}`}>{part}</mark> : part;
  })}</>;
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function RawEvaluationDataDialog({ result, onClose }: { result: Record<string, unknown>; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/30 p-4 sm:items-center"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      role="presentation"
    >
      <section aria-labelledby="raw-evaluation-data-title" aria-modal="true" className="my-auto w-full max-w-3xl overflow-hidden rounded-lg border border-line bg-white shadow-xl" role="dialog">
        <div className="flex items-center justify-between gap-4 border-b border-line p-5">
          <h2 id="raw-evaluation-data-title" className="text-base font-semibold">Raw evaluation data</h2>
          <button aria-label="Close raw evaluation data" className="ghost-button grid size-8 place-items-center p-0" onClick={onClose} title="Close" type="button"><X size={16} /></button>
        </div>
        <pre className="max-h-[min(680px,calc(100vh-10rem))] overflow-auto bg-panel p-5 text-xs leading-5">{JSON.stringify(result, null, 2)}</pre>
      </section>
    </div>
  );
}
