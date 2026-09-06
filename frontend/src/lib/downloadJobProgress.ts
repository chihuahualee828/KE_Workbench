import type { ModelCacheStatus, ModelDownloadJob } from "./types";

const activeStates = new Set(["queued", "downloading", "canceling"]);

// Only the exact active job may read live cache data. Finished jobs are records
// of a past transfer, even when the same model is downloaded or deleted again.
export function downloadJobCache(job: ModelDownloadJob, cache: ModelCacheStatus[]) {
  const live = cache.find((item) => item.model_id === job.model_id);
  return activeStates.has(job.status) && live?.active_job?.job_id === job.job_id
    ? live
    : job.cache;
}

export function transferProgress(job: ModelDownloadJob, status?: ModelCacheStatus | null) {
  if (job.job_type === "local_import") {
    return { downloadedBytes: job.progress_current_bytes ?? 0, totalBytes: job.progress_total_bytes, known: true };
  }
  const active = activeStates.has(job.status);
  const snapshot = active ? status ?? job.cache : job.cache;
  const downloadedBytes = active
    ? snapshot?.size_bytes ?? job.progress_current_bytes
    : job.progress_current_bytes ?? snapshot?.size_bytes;
  const totalBytes = job.progress_total_bytes ?? snapshot?.hub_size_bytes
    ?? (job.status === "completed" ? downloadedBytes : undefined);
  return { downloadedBytes: downloadedBytes ?? 0, totalBytes, known: downloadedBytes != null };
}

export function formatTransferSize(job: ModelDownloadJob, status?: ModelCacheStatus | null) {
  const { downloadedBytes, totalBytes, known } = transferProgress(job, status);
  if (!known) return "Progress unavailable";
  if (totalBytes && totalBytes > 0) {
    const percentage = Math.min(100, Math.round((downloadedBytes / totalBytes) * 100));
    return `${formatBytes(downloadedBytes)} / ${formatBytes(totalBytes)} (${percentage}%)`;
  }
  return formatBytes(downloadedBytes);
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
