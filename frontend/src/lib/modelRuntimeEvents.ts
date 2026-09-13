import { getModelRuntimeSnapshot, setModelRuntimeSnapshot, setUnloadingModelSnapshot } from "./modelStatusCache";
import type { ModelRuntimeJob, ModelRuntimeStatus } from "./types";
import { api } from "./api";

const eventName = "kew:model-runtime-load-started";
const statusEventName = "kew:model-runtime-status-changed";
const connectionEventName = "kew:model-runtime-connection-changed";
const activeModelEventName = "kew:active-runtime-model-changed";
const unloadEventName = "kew:model-runtime-unload-changed";
const activeStatuses = ["queued", "loading"];
const modelSelectionKeys = [
  "kew:generation:modelId",
  "kew:lens:modelId",
  "kew:trace:modelId",
  "kew:evaluate:modelId",
  "kew:edit:rome:modelId",
  "kew:edit:memit:modelId",
  "kew:settings:mom2:modelId"
];
const waiters = new Map<string, Promise<ModelRuntimeJob>>();
let runtimeConnected = true;

export function announceModelRuntimeLoad(job: ModelRuntimeJob) {
  setModelRuntimeSnapshot(applyRuntimeJobToRuntime(getModelRuntimeSnapshot(), job));
  // Select the requested model as soon as its load job starts. The worker may
  // still be transferring weights, but every tool should already target the
  // same model the user chose to load.
  if (["queued", "loading", "completed"].includes(job.status)) {
    announceActiveRuntimeModel(job.model_id);
  }
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
  setModelRuntimeSnapshot(applyRuntimeStatusToRuntime(getModelRuntimeSnapshot(), status));
  if (status.loaded) announceActiveRuntimeModel(status.model_id);
  window.dispatchEvent(new CustomEvent<ModelRuntimeStatus>(statusEventName, { detail: status }));
}

export function listenForActiveRuntimeModel(callback: (modelId: string) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<string>).detail);
  }
  window.addEventListener(activeModelEventName, handle);
  return () => window.removeEventListener(activeModelEventName, handle);
}

function announceActiveRuntimeModel(modelId: string) {
  modelSelectionKeys.forEach((key) => window.localStorage.setItem(key, JSON.stringify(modelId)));
  window.dispatchEvent(new CustomEvent<string>(activeModelEventName, { detail: modelId }));
}

export function announceModelRuntimeStatuses(statuses: ModelRuntimeStatus[]) {
  setModelRuntimeSnapshot(statuses);
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

export function announceModelRuntimeUnloading(modelId: string, unloading: boolean) {
  setUnloadingModelSnapshot(unloading ? modelId : null);
  window.dispatchEvent(
    new CustomEvent<{ modelId: string; unloading: boolean }>(unloadEventName, {
      detail: { modelId, unloading }
    })
  );
}

export function listenForModelRuntimeUnloading(
  callback: (event: { modelId: string; unloading: boolean }) => void
) {
  function handle(event: Event) {
    callback((event as CustomEvent<{ modelId: string; unloading: boolean }>).detail);
  }
  window.addEventListener(unloadEventName, handle);
  return () => window.removeEventListener(unloadEventName, handle);
}

export function listenForModelRuntimeStream(
  callback: (statuses: ModelRuntimeStatus[]) => void,
  onDisconnected?: () => void,
  onConnected?: () => void
) {
  let closed = false;
  const stopStream = api.subscribeToEvents("/api/models/runtime/events", {
    onOpen: () => { if (!closed) onConnected?.(); },
    onEvent: (event, data) => {
      if (event !== "runtime") return;
      try {
        callback(JSON.parse(data) as ModelRuntimeStatus[]);
      } catch {
        // Ignore a malformed transient event; the subscription will remain open.
      }
    },
    onError: () => { if (!closed) onDisconnected?.(); }
  });
  return () => {
    closed = true;
    stopStream();
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
