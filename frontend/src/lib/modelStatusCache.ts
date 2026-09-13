import type { ModelCacheStatus, ModelRuntimeStatus } from "./types";

// Retain the latest server state across client-side page navigation. Keep this
// in memory so a browser reload still checks the backend from scratch.
let cacheStatuses: ModelCacheStatus[] = [];
let runtimeStatuses: ModelRuntimeStatus[] = [];
let unloadingModelId: string | null = null;

export function getModelCacheSnapshot() { return cacheStatuses; }
export function getModelRuntimeSnapshot() { return runtimeStatuses; }
export function getUnloadingModelSnapshot() { return unloadingModelId; }
export function setModelCacheSnapshot(statuses: ModelCacheStatus[]) { cacheStatuses = statuses; }
export function setModelRuntimeSnapshot(statuses: ModelRuntimeStatus[]) { runtimeStatuses = statuses; }
export function setUnloadingModelSnapshot(modelId: string | null) { unloadingModelId = modelId; }
