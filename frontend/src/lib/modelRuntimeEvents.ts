import type { ModelRuntimeJob, ModelRuntimeStatus } from "./types";
import { api } from "./api";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const eventName = "kew:model-runtime-load-started";
const statusEventName = "kew:model-runtime-status-changed";
const connectionEventName = "kew:model-runtime-connection-changed";
const activeStatuses = ["queued", "loading"];
const waiters = new Map<string, Promise<ModelRuntimeJob>>();
let runtimeConnected = true;

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

export function announceModelRuntimeDisconnected() {
  runtimeConnected = false;
  window.dispatchEvent(new CustomEvent<boolean>(connectionEventName, { detail: false }));
}

export function announceModelRuntimeConnected() {
  runtimeConnected = true;
  window.dispatchEvent(new CustomEvent<boolean>(connectionEventName, { detail: true }));
}

export function isModelRuntimeConnected() {
  return runtimeConnected;
}

export function listenForModelRuntimeConnection(callback: (connected: boolean) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<boolean>).detail);
  }
  window.addEventListener(connectionEventName, handle);
  return () => window.removeEventListener(connectionEventName, handle);
}

export function listenForModelRuntimeStream(
  callback: (statuses: ModelRuntimeStatus[]) => void,
  onDisconnected?: () => void
) {
  const source = new EventSource(`${API_BASE}/api/models/runtime/events`);
  let closed = false;
  const handleRuntime = (event: Event) => {
    try {
      const statuses = JSON.parse((event as MessageEvent<string>).data) as ModelRuntimeStatus[];
      callback(statuses);
    } catch {
      // Ignore a malformed transient event; EventSource will keep the stream alive.
    }
  };
  source.addEventListener("runtime", handleRuntime);
  source.onerror = () => {
    if (!closed) onDisconnected?.();
  };
  return () => {
    closed = true;
    source.close();
  };
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
