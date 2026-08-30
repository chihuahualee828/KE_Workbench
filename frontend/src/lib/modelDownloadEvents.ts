import type { ModelCacheStatus, ModelDownloadJob } from "./types";

const eventName = "kew:model-download-started";

export function announceModelDownload(job: ModelDownloadJob) {
  window.dispatchEvent(new CustomEvent<ModelDownloadJob>(eventName, { detail: job }));
}

export function listenForModelDownloads(callback: (job: ModelDownloadJob) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<ModelDownloadJob>).detail);
  }
  window.addEventListener(eventName, handle);
  return () => window.removeEventListener(eventName, handle);
}

export function applyDownloadJobToCache(
  cache: ModelCacheStatus[],
  job: ModelDownloadJob
): ModelCacheStatus[] {
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
