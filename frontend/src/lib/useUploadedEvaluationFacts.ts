"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { readAllDatasetValues, replaceDatasetValues } from "./factDataset";
import type { EvaluationFact } from "./types";

type StoredEvaluationFact = EvaluationFact & { rephrase_override?: string };
type UploadedDescriptor = { id: string; fileName: string };
type UploadedSnapshot = { descriptor: UploadedDescriptor; facts: StoredEvaluationFact[] };
type UploadScope = {
  datasetId: string;
  storageKey: string;
  active: boolean;
  snapshot: UploadedSnapshot | null;
  queue: Promise<void>;
};
type UploadState = {
  scope: UploadScope | null;
  snapshot: UploadedSnapshot | null;
  ready: boolean;
  error: string;
};

const EMPTY_FACTS: StoredEvaluationFact[] = [];

/** Retain the normalized evaluation inputs, excluding raw upload metadata and overrides. */
function inputFact(fact: EvaluationFact): EvaluationFact {
  return {
    key: fact.key,
    subject: fact.subject,
    relation_prompt: fact.relation_prompt,
    prompt: fact.prompt,
    new_object: fact.new_object,
    ground_truth: fact.ground_truth,
    rephrase_prompt: fact.rephrase_prompt,
    locality_prompt: fact.locality_prompt,
    locality_ground_truth: fact.locality_ground_truth,
    portability_prompt: fact.portability_prompt,
    portability_ground_truth: fact.portability_ground_truth,
  };
}

function enqueue<T>(scope: UploadScope, operation: () => Promise<T>): Promise<T> {
  const pending = scope.queue.then(operation);
  scope.queue = pending.then(() => undefined, () => undefined);
  return pending;
}

function readDescriptor(storageKey: string): UploadedDescriptor | null {
  const raw = window.sessionStorage.getItem(storageKey);
  if (!raw) return null;
  const value: unknown = JSON.parse(raw);
  if (!value || typeof value !== "object" || !("id" in value) || !("fileName" in value)
    || typeof value.id !== "string" || !value.id || typeof value.fileName !== "string") {
    throw new Error("The saved evaluation upload is invalid. Please upload the file again.");
  }
  return { id: value.id, fileName: value.fileName };
}

/** One replaceable snapshot per tab/model/backend keeps repeated uploads bounded. */
export function useUploadedEvaluationFacts({ modelId, backendUrl, sessionId, method }: {
  modelId: string;
  backendUrl: string;
  sessionId: string;
  method: string;
}) {
  const datasetId = `kew:evaluate:uploaded-facts:${JSON.stringify([sessionId, modelId, backendUrl])}`;
  const scope = useMemo<UploadScope>(() => ({
    datasetId,
    storageKey: `${datasetId}:source`,
    active: false,
    snapshot: null,
    queue: Promise.resolve(),
  }), [datasetId]);
  const [state, setState] = useState<UploadState>({ scope: null, snapshot: null, ready: false, error: "" });

  const publish = useCallback((error = "", ready = true) => {
    if (scope.active) setState({ scope, snapshot: scope.snapshot, ready, error });
  }, [scope]);

  useEffect(() => {
    let canceled = false;
    scope.active = true;
    setState({ scope, snapshot: null, ready: false, error: "" });
    void enqueue(scope, async () => {
      try {
        // sessionStorage is read only after mount to keep hydration deterministic.
        const descriptor = readDescriptor(scope.storageKey);
        if (!descriptor) {
          scope.snapshot = null;
        } else {
          const [snapshot] = await readAllDatasetValues<UploadedSnapshot>(scope.datasetId);
          if (!snapshot || snapshot.descriptor?.id !== descriptor.id || !Array.isArray(snapshot.facts)) {
            throw new Error("The saved evaluation upload is unavailable. Please upload the file again.");
          }
          scope.snapshot = { descriptor, facts: snapshot.facts };
        }
        if (!canceled) publish();
      } catch (reason) {
        if (!canceled) publish(reason instanceof Error ? reason.message : String(reason), false);
      }
    });
    return () => {
      canceled = true;
      scope.active = false;
    };
  }, [publish, scope]);

  const mutate = useCallback((operation: () => Promise<void>) => enqueue(scope, operation).catch((reason) => {
    if (scope.active) setState((current) => current.scope === scope
      ? { ...current, error: reason instanceof Error ? reason.message : String(reason) }
      : current);
    throw reason;
  }), [scope]);

  const commitSource = useCallback(async (next: UploadedSnapshot | null) => {
    const previous = scope.snapshot;
    await replaceDatasetValues<UploadedSnapshot>(scope.datasetId, next ? [next] : []);
    try {
      if (next) window.sessionStorage.setItem(scope.storageKey, JSON.stringify(next.descriptor));
      else window.sessionStorage.removeItem(scope.storageKey);
    } catch (reason) {
      // A failed descriptor write must not replace the previously usable upload.
      await replaceDatasetValues<UploadedSnapshot>(scope.datasetId, previous ? [previous] : []);
      throw reason;
    }
    scope.snapshot = next;
    publish();
  }, [publish, scope]);

  const importFacts = useCallback((facts: EvaluationFact[], fileName: string) => mutate(async () => {
    if (!facts.length) throw new Error("The uploaded file contains no evaluation facts.");
    const id = window.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    await commitSource({ descriptor: { id, fileName }, facts: facts.map(inputFact) });
  }), [commitSource, mutate]);

  const clear = useCallback(() => mutate(() => commitSource(null)), [commitSource, mutate]);

  const saveOverride = useCallback((key: string, value: string) => mutate(async () => {
    const previous = scope.snapshot;
    if (!previous || !previous.facts.some((fact) => fact.key === key)) {
      throw new Error("The uploaded evaluation fact no longer exists.");
    }
    const next: UploadedSnapshot = {
      ...previous,
      facts: previous.facts.map((fact) => fact.key === key ? { ...fact, rephrase_override: value } : fact),
    };
    await replaceDatasetValues(scope.datasetId, [next]);
    scope.snapshot = next;
    publish();
  }), [mutate, publish, scope]);

  const current = state.scope === scope ? state : null;
  const snapshot = current?.snapshot ?? null;
  const facts = snapshot?.facts ?? EMPTY_FACTS;
  const inputFacts = useMemo(() => facts.map(inputFact), [facts]);
  const overrides = useMemo<Record<string, string>>(() => Object.fromEntries(facts
    .filter((fact) => typeof fact.rephrase_override === "string")
    .map((fact) => [fact.key, fact.rephrase_override!])), [facts]);
  const getItem = useCallback((index: number): StoredEvaluationFact | undefined => facts[index], [facts]);
  const ensureRange = useCallback((_owner: string, _start: number, _end: number) => undefined, []);
  const promptCounts = useMemo(() => ({
    locality: facts.filter((fact) => fact.locality_prompt.trim()).length,
    portability: facts.filter((fact) => fact.portability_prompt.trim()).length,
  }), [facts]);
  const findMatches = useCallback((query: string): number[] => {
    const needle = query.trim().toLowerCase();
    if (!needle) return [];
    return facts.reduce<number[]>((matches, fact, index) => {
      if ([fact.subject, fact.ground_truth, fact.new_object].some((value) => value.toLowerCase().includes(needle))) {
        matches.push(index);
      }
      return matches;
    }, []);
  }, [facts]);

  return {
    getItem,
    ensureRange,
    overrides,
    saveOverride,
    error: current?.error ?? "",
    localityPromptCount: promptCounts.locality,
    portabilityPromptCount: promptCounts.portability,
    method: method || "rome",
    ready: current?.ready ?? false,
    sourceKey: `${datasetId}:${snapshot?.descriptor.id ?? "none"}`,
    total: facts.length,
    uploaded: snapshot?.descriptor ?? null,
    inputFacts,
    importFacts,
    clear,
    findMatches,
  };
}
