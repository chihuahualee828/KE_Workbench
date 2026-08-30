import type { ModelRuntimeJob, ModelRuntimeStatus } from "./types";
import { api } from "./api";

const eventName = "kew:model-runtime-load-started";
const statusEventName = "kew:model-runtime-status-changed";
const activeStatuses = ["queued", "loading"];
const waiters = new Map<string, Promise<ModelRuntimeJob>>();

export function announceModelRuntimeLoad(job: ModelRuntimeJob) {
  window.dispatchEvent(new CustomEvent<ModelRuntimeJob>(eventName, { detail: job }));
}

export function listenForModelRuntimeLoads(callback: (job: ModelRuntimeJob) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<ModelRuntimeJob>).detail);
  }
  window.addEventListener(eventName, handle);
  return () => window.removeEventListener(eventName, handle);
}

export function announceModelRuntimeStatus(status: ModelRuntimeStatus) {
  window.dispatchEvent(new CustomEvent<ModelRuntimeStatus>(statusEventName, { detail: status }));
}

export function announceModelRuntimeStatuses(statuses: ModelRuntimeStatus[]) {
  statuses.forEach(announceModelRuntimeStatus);
}

export function listenForModelRuntimeStatuses(callback: (status: ModelRuntimeStatus) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<ModelRuntimeStatus>).detail);
  }
  window.addEventListener(statusEventName, handle);
  return () => window.removeEventListener(statusEventName, handle);
}

export function applyRuntimeStatusToRuntime(
  runtime: ModelRuntimeStatus[],
  status: ModelRuntimeStatus
): ModelRuntimeStatus[] {
  if (!runtime.some((item) => item.model_id === status.model_id)) {
    return [...runtime, status];
  }
  return runtime.map((item) => (item.model_id === status.model_id ? status : item));
}

export function applyRuntimeJobToRuntime(
  runtime: ModelRuntimeStatus[],
  job: ModelRuntimeJob
): ModelRuntimeStatus[] {
  return runtime.map((status) =>
    status.model_id === job.model_id
      ? {
          ...status,
          active_job: activeStatuses.includes(job.status) ? job : null,
          loaded: job.status === "completed" ? true : job.status === "failed" ? false : status.loaded,
          tokenizer_loaded: job.status === "completed" ? true : status.tokenizer_loaded,
          device: job.device ?? status.device,
          runtime_state: job.status === "completed" ? "loaded" : (job.status as ModelRuntimeStatus["runtime_state"])
        }
      : status
  );
}

export function watchModelRuntimeJob(job: ModelRuntimeJob) {
  if (!activeStatuses.includes(job.status)) {
    announceModelRuntimeLoad(job);
    return Promise.resolve(job);
  }
  const existing = waiters.get(job.job_id);
  if (existing) return existing;

  const waiter = waitUntilRuntimeJobSettles(job)
    .then((updatedJob) => {
      announceModelRuntimeLoad(updatedJob);
      return updatedJob;
    })
    .finally(() => {
      waiters.delete(job.job_id);
    });
  waiters.set(job.job_id, waiter);
  return waiter;
}

async function waitUntilRuntimeJobSettles(job: ModelRuntimeJob) {
  let current = job;
  while (activeStatuses.includes(current.status)) {
    current = await api.waitModelRuntimeJob(current.job_id, 120);
  }
  return current;
}
