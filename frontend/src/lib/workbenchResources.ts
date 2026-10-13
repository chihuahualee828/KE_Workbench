"use client";

import { useSyncExternalStore } from "react";
import type { Dispatch, SetStateAction } from "react";
import { api, apiReadVersion, invalidateApiReads } from "./api";
import type { EditJob, Experiment, ModelCacheStatus, ModelDownloadJob, ModelRuntimeStatus, Patch, SupportedModel } from "./types";
import { applyEditPatchState } from "./editPatchState";

function resource<T>(initial: T) {
  let value = initial;
  const listeners = new Set<() => void>();
  const get = () => value;
  const set: Dispatch<SetStateAction<T>> = (update) => {
    const next = typeof update === "function" ? (update as (current: T) => T)(value) : update;
    if (next === value) return;
    value = next;
    listeners.forEach((listener) => listener());
  };
  const subscribe = (listener: () => void) => {
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  };
  const serverSnapshot = () => initial;
  return { get, set, reset: () => set(initial), use: (): [T, typeof set] => [useSyncExternalStore(subscribe, get, serverSnapshot), set] };
}

export const modelResource = resource<SupportedModel[]>([]);
export const cacheResource = resource<ModelCacheStatus[]>([]);
export const runtimeResource = resource<ModelRuntimeStatus[]>([]);
export const downloadResource = resource<ModelDownloadJob[]>([]);
export const experimentResource = resource<Experiment[]>([]);
export const patchResource = resource<Patch[]>([]);
export const historyResource = resource({ experiments: 0, completed: 0, patches: 0 });
let generation = 0;
let experimentOffset = 0;
let patchOffset = 0;
let experimentRevision = 0;
let patchRevision = 0;
const pending = new Map<string, Promise<void>>();
const editPatchVersions = new Map<string, number>();

export function resetWorkbenchResources() {
  generation += 1;
  pending.clear();
  editPatchVersions.clear();
  experimentOffset = patchOffset = 0;
  [modelResource, cacheResource, runtimeResource, downloadResource, experimentResource, patchResource, historyResource].forEach((item) => item.reset());
}

function once(key: string, work: (version: number) => Promise<void>) {
  const existing = pending.get(key);
  if (existing) return existing;
  const version = generation;
  const promise = (async () => {
    let epoch: number;
    do {
      epoch = apiReadVersion();
      await work(version);
    } while (version === generation && epoch !== apiReadVersion());
  })().finally(() => { if (pending.get(key) === promise) pending.delete(key); });
  pending.set(key, promise);
  return promise;
}

export function refreshTransfers() {
  return once("transfers", async (version) => {
    const [cache, jobs] = await Promise.all([api.modelCache(), api.modelDownloadJobs()]);
    if (version !== generation) return;
    cacheResource.set(cache);
    downloadResource.set(jobs);
  });
}

function unique<T>(items: T[], identity: (item: T) => string) {
  return [...new Map(items.map((item) => [identity(item), item])).values()];
}

export function refreshExperimentHistory() {
  return once("experiments", async (version) => {
    experimentRevision += 1;
    const [first, running] = await Promise.all([api.experimentPage(0), api.experimentPage(0, "running")]);
    const retained = [...first.items];
    for (let offset = retained.length; offset < Math.min(experimentOffset, first.total); offset += 50) retained.push(...(await api.experimentPage(offset)).items);
    const active = [...running.items];
    for (let offset = active.length; offset < running.total; offset += 50) active.push(...(await api.experimentPage(offset, "running")).items);
    if (version !== generation) return;
    experimentOffset = retained.length;
    experimentResource.set(unique([...retained, ...active], (item) => item.id));
    historyResource.set((current) => ({ ...current, experiments: first.total, completed: first.status_counts.completed ?? 0 }));
  });
}

export function refreshPatchHistory() {
  return once("patches", async (version) => {
    const epoch = apiReadVersion();
    patchRevision += 1;
    const [page, active] = await Promise.all([api.patchPage(0), api.activePatches()]);
    const retained = [...page.items];
    for (let offset = retained.length; offset < Math.min(patchOffset, page.total); offset += 50) retained.push(...(await api.patchPage(offset)).items);
    if (version !== generation || epoch !== apiReadVersion()) return;
    patchOffset = retained.length;
    const activeById = new Map(active.patches.map((patch) => [patch.patch_id, patch]));
    // The active-stack response is authoritative, including when it is empty.
    // A history page may have been read just before worker shutdown finished.
    patchResource.set(unique([...retained.map((patch) => ({
      ...patch, active: activeById.has(patch.patch_id),
      active_index: activeById.get(patch.patch_id)?.active_index ?? null,
    })), ...active.patches], (item) => item.patch_id));
    historyResource.set((current) => ({ ...current, patches: page.total }));
  });
}

export function updatePatchesFromEditJob(job: EditJob) {
  const state = job.patch_state;
  if (!state || state.version <= (editPatchVersions.get(job.job_id) ?? 0)) return;
  editPatchVersions.set(job.job_id, state.version);
  invalidateApiReads();
  patchResource.set((current) => applyEditPatchState(current, state));
  void refreshPatchHistory().catch(() => undefined);
}

export function reconcilePatchesAfterRuntimeUpdate(previous: ModelRuntimeStatus[], statuses: ModelRuntimeStatus[]) {
  const unloaded = new Set(statuses.filter((status) => !status.loaded).map((status) => status.model_id));
  const needsRefresh = previous.some((status) => status.loaded && unloaded.has(status.model_id))
    || patchResource.get().some((patch) => patch.active && unloaded.has(patch.base_model)
      && !patch.lineage_only && !patch.baked_into_model);
  if (!needsRefresh) return Promise.resolve();
  // Cancellation completes before asynchronous worker termination. Bypass any
  // pre-shutdown cache and make an in-flight refresh retry after this event.
  invalidateApiReads();
  return refreshPatchHistory();
}

export function loadMoreExperiments() {
  return once("more-experiments", async (version) => {
    if (pending.has("experiments")) await pending.get("experiments");
    const revision = experimentRevision;
    const offset = experimentOffset;
    if (offset >= historyResource.get().experiments) return;
    const page = await api.experimentPage(offset);
    if (version !== generation || revision !== experimentRevision || offset !== experimentOffset) return;
    experimentOffset += page.items.length;
    experimentResource.set((current) => unique([...current, ...page.items], (item) => item.id));
  });
}

export function loadMorePatches() {
  return once("more-patches", async (version) => {
    if (pending.has("patches")) await pending.get("patches");
    const revision = patchRevision;
    const offset = patchOffset;
    if (offset >= historyResource.get().patches) return;
    const page = await api.patchPage(offset);
    if (version !== generation || revision !== patchRevision || offset !== patchOffset) return;
    patchOffset += page.items.length;
    patchResource.set((current) => unique([...current, ...page.items], (item) => item.patch_id));
  });
}

export async function refreshJobs() {
  await Promise.all([refreshTransfers(), refreshExperimentHistory()]);
}
