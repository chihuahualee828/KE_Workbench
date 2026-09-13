import { getModelCacheSnapshot, setModelCacheSnapshot } from "./modelStatusCache";
import type { ModelCacheStatus, ModelDownloadJob } from "./types";

const eventName = "kew:model-download-started";
const cacheStatusEventName = "kew:model-cache-status-changed";

export function announceModelDownload(job: ModelDownloadJob) {
  setModelCacheSnapshot(applyDownloadJobToCache(getModelCacheSnapshot(), job));
  window.dispatchEvent(new CustomEvent<ModelDownloadJob>(eventName, { detail: job }));
}

export function listenForModelDownloads(callback: (job: ModelDownloadJob) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<ModelDownloadJob>).detail);
  }
  window.addEventListener(eventName, handle);
  return () => window.removeEventListener(eventName, handle);
}

export function announceModelCacheStatus(status: ModelCacheStatus) {
  setModelCacheSnapshot(applyCacheStatusToCache(getModelCacheSnapshot(), status));
  window.dispatchEvent(new CustomEvent<ModelCacheStatus>(cacheStatusEventName, { detail: status }));
}

export function listenForModelCacheStatuses(callback: (status: ModelCacheStatus) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<ModelCacheStatus>).detail);
  }
  window.addEventListener(cacheStatusEventName, handle);
  return () => window.removeEventListener(cacheStatusEventName, handle);
}

export function upsertModelDownloadJob(
  jobs: ModelDownloadJob[],
  job: ModelDownloadJob
): ModelDownloadJob[] {
  return [job, ...jobs.filter((item) => item.job_id !== job.job_id)];
}

export function applyDownloadJobToCache(
  cache: ModelCacheStatus[],
  job: ModelDownloadJob
): ModelCacheStatus[] {
  // A local import has no registered model/cache row until the user confirms
  // the inferred settings by clicking Register.
  if (job.job_type === "local_import") return cache;
  if (!cache.some((status) => status.model_id === job.model_id)) {
    return [
      ...cache,
      {
        model_id: job.model_id,
        hf_id: job.hf_id,
        cached: false,
        cache_state: job.status as ModelCacheStatus["cache_state"],
        cache_path: "",
        size_bytes: 0,
        snapshot_count: 0,
        has_model_weights: false,
        has_config: false,
        incomplete_file_count: 0,
        active_job: job
      }
    ];
  }
  return cache.map((status) =>
    status.model_id === job.model_id
      ? {
          ...status,
          active_job: job,
          cached: false,
          cache_state: job.status as ModelCacheStatus["cache_state"]
        }
      : status
  );
}

export function applyCacheStatusToCache(
  cache: ModelCacheStatus[],
  status: ModelCacheStatus
): ModelCacheStatus[] {
  if (!cache.some((item) => item.model_id === status.model_id)) {
    return [...cache, status];
  }
  return cache.map((item) => item.model_id === status.model_id ? status : item);
}
