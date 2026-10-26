import type { EditJobOptions, EditMethod, BatchWeightMethod, TrainingMethod, Mom2Method } from "./editingMethods";
import type { LoraModuleCatalog } from "./loraModules";
import type { TrainingHyperparameters } from "./trainingHyperparameters";
import type { ActiveEvaluationFacts, EvaluationFact, EvaluationMetric, EvaluationProtocols, EvaluationResultOverview, EvaluationResultPage, AppSettings, EditJob, Experiment, ExperimentDetails, HfTokenStatus, LogitLensLayer, ModelCacheLocation, ModelCacheStatus, ModelDownloadJob, ModelRuntimeJob, ModelRuntimeStatus, Mom2Cache, Mom2Dtype, Mom2StatisticsRequest, Mom2StatisticsStatus, Patch, PatchEditPage, ReliabilityTestResultPage, SupportedModel, TraceCell } from "./types";
import { backendAuthorizationHeaders, backendUrl } from "./backendConnection";

import { setModelCacheSnapshot, setModelRuntimeSnapshot } from "./modelStatusCache";

type EventStreamHandlers = {
  onOpen?: () => void;
  onEvent: (event: string, data: string) => void;
  onError?: (error: Error) => void;
};

function requestHeaders(init?: RequestInit) {
  const headers = new Headers(init?.headers);
  Object.entries(backendAuthorizationHeaders()).forEach(([name, value]) => headers.set(name, value));
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return headers;
}

// Coalesce shared GETs across mounts/events; mutations invalidate snapshots.
const reads = new Map<string, { promise: Promise<unknown>; expires: number }>();
let readEpoch = 0;
export function apiReadVersion() { return readEpoch; }
export function invalidateApiReads() { readEpoch += 1; reads.clear(); }
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const mutation = init?.method && init.method !== "GET";
  const cacheable = !mutation && !init?.signal && /^(\/api\/models(?:$|\/(?:cache$|runtime$|cache\/jobs$))|\/api\/(?:experiments|patches)\/page|\/api\/patches\/active)/.test(path);
  if (mutation) invalidateApiReads();
  if (!cacheable) {
    try { return await rawRequest<T>(path, init); }
    finally { if (mutation) invalidateApiReads(); }
  }
  for (const [key, entry] of reads) if (entry.expires <= Date.now()) reads.delete(key);
  const key = JSON.stringify([backendUrl(path), backendAuthorizationHeaders()]);
  const current = reads.get(key);
  if (current && current.expires > Date.now()) return current.promise as Promise<T>;
  const epoch = readEpoch;
  const entry = { promise: Promise.resolve(undefined) as Promise<unknown>, expires: Infinity };
  entry.promise = rawRequest<T>(path, init).then((value) => {
    entry.expires = Date.now() + 1000;
    setTimeout(() => { if (reads.get(key) === entry) reads.delete(key); }, 1000);
    if (epoch !== readEpoch && reads.get(key) === entry) reads.delete(key);
    return value;
  }, (error) => { if (reads.get(key) === entry) reads.delete(key); throw error; });
  reads.set(key, entry);
  return entry.promise as Promise<T>;
}

async function rawRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const url = backendUrl(path);
  const auth = JSON.stringify(backendAuthorizationHeaders());
  const response = await fetch(url, {
    ...init,
    headers: requestHeaders(init),
    cache: "no-store"
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      detail = response.statusText;
    }
    throw new Error(`${response.status} ${detail}`);
  }
  const payload = await response.json() as T;
  if (url !== backendUrl(path) || auth !== JSON.stringify(backendAuthorizationHeaders())) throw new Error("Backend connection changed during request.");
  return payload;
}

async function download(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(backendUrl(path), {
    ...init,
    headers: requestHeaders(init),
    cache: "no-store"
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      detail = response.statusText;
    }
    throw new Error(`${response.status} ${detail}`);
  }
  return response;
}

async function upload<T>(path: string, body: FormData): Promise<T> {
  invalidateApiReads();
  try {
    return await rawUpload<T>(path, body);
  } finally { invalidateApiReads(); }
}

async function rawUpload<T>(path: string, body: FormData): Promise<T> {
  const response = await fetch(backendUrl(path), {
    method: "POST",
    body,
    headers: requestHeaders(),
    cache: "no-store"
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      detail = response.statusText;
    }
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json() as Promise<T>;
}

function parseEventBlock(block: string, handlers: EventStreamHandlers) {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    const value = (separator === -1 ? "" : line.slice(separator + 1)).replace(/^ /, "");
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  }
  if (data.length) handlers.onEvent(event, data.join("\n"));
}

function openEventStream(path: string, handlers: EventStreamHandlers) {
  const url = backendUrl(path);
  const headers = requestHeaders({ headers: { Accept: "text/event-stream" } });
  let closed = false;
  let controller: AbortController | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let resumeRetry: (() => void) | null = null;

  const connect = async () => {
    while (!closed) {
      controller = new AbortController();
      try {
        const response = await fetch(url, {
          headers,
          cache: "no-store",
          signal: controller.signal
        });
        if (!response.ok || !response.body) throw new Error(`Unable to open event stream: ${response.status} ${response.statusText}`);
        handlers.onOpen?.();

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (!closed) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split(/\r?\n\r?\n/);
          buffer = blocks.pop() ?? "";
          blocks.forEach((block) => parseEventBlock(block, handlers));
        }
      } catch (error) {
        if (!closed) handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
      }
      if (closed) break;
      await new Promise<void>((resolve) => {
        resumeRetry = resolve;
        retryTimer = setTimeout(resolve, 1000);
      });
      retryTimer = null;
      resumeRetry = null;
    }
  };

  void connect();
  return () => {
    closed = true;
    controller?.abort();
    if (retryTimer) clearTimeout(retryTimer);
    resumeRetry?.();
  };
}

const streams = new Map<string, { listeners: Set<EventStreamHandlers>; stop: () => void; last?: [string, string]; opened: boolean }>();
function subscribeToEvents(path: string, handlers: EventStreamHandlers) {
  const key = JSON.stringify([backendUrl(path), backendAuthorizationHeaders()]);
  let stream = streams.get(key);
  if (!stream) {
    stream = { listeners: new Set(), stop: () => undefined, opened: false };
    streams.set(key, stream);
    const current = stream;
    current.stop = openEventStream(path, {
      onOpen: () => { current.opened = true; current.listeners.forEach((listener) => listener.onOpen?.()); },
      onEvent: (event, data) => {
        // Keep a bounded log snapshot for consumers joining an existing job stream.
        // Current listeners still receive the original delta.
        if (event === "job") {
          try {
            const update = JSON.parse(data) as EditJob;
            const previous = current.last?.[0] === "job" ? JSON.parse(current.last[1]) as EditJob : null;
            if (Array.isArray(update.logs)) {
              const logs = (previous && update.log_start_count === previous.log_count
                ? [...previous.logs, ...update.logs] : update.logs).slice(-500);
              current.last = [event, JSON.stringify({ ...update, logs, log_start_count: (update.log_count ?? logs.length) - logs.length })];
            } else current.last = [event, data];
          } catch { current.last = [event, data]; }
        } else if (event !== "complete" || current.last?.[0] !== "job") current.last = [event, data];
        current.listeners.forEach((listener) => listener.onEvent(event, data));
      },
      onError: (error) => { current.opened = false; current.listeners.forEach((listener) => listener.onError?.(error)); }
    });
  }
  const current = stream;
  current.listeners.add(handlers);
  queueMicrotask(() => {
    if (!current.listeners.has(handlers)) return;
    if (current.opened) handlers.onOpen?.();
    if (current.last) handlers.onEvent(...current.last);
  });
  return () => {
    current.listeners.delete(handlers);
    if (!current.listeners.size) { current.stop(); if (streams.get(key) === current) streams.delete(key); }
  };
}

export const api = {
  url: backendUrl,
  subscribeToEvents,
  health: () => request<{ status: string }>("/api/health"),
  settings: () => request<AppSettings>("/api/settings"),
  setModelCacheLocation: (body: { download_dir: string }) =>
    request<ModelCacheLocation>("/api/settings/model-cache", {
      method: "PUT",
      body: JSON.stringify(body)
    }),
  models: () => request<SupportedModel[]>("/api/models"),
  hfTokenStatus: () => request<HfTokenStatus>("/api/models/hf-token"),
  setHfToken: (body: { token: string }) =>
    request<HfTokenStatus>("/api/models/hf-token", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  clearHfToken: () =>
    request<HfTokenStatus>("/api/models/hf-token", {
      method: "DELETE"
    }),
  addLocalModel: (body: {
    id?: string;
    name: string;
    local_path: string;
    model_type: string;
    n_layers: number;
    supports_generation: boolean;
    supports_logit_lens?: boolean;
    supports_causal_trace?: boolean;
    supports_rome?: boolean;
    supports_r_rome?: boolean;
    supports_memit?: boolean;
    supports_ft?: boolean;
    supports_lora?: boolean;
    supports_emmet?: boolean;
    supports_memit_fe?: boolean;
    supports_pmet?: boolean;
    supports_dpo?: boolean;
    supports_eamet?: boolean;
    supports_alphaedit?: boolean;
    supports_sphere?: boolean;
    supports_kn?: boolean;
    rome_hparams?: string | null;
    r_rome_hparams?: string | null;
    memit_hparams?: string | null;
    ft_hparams?: string | null;
    lora_hparams?: string | null;
    emmet_hparams?: string | null;
    memit_fe_hparams?: string | null;
    pmet_hparams?: string | null;
    dpo_hparams?: string | null;
    eamet_hparams?: string | null;
    alphaedit_hparams?: string | null;
    sphere_hparams?: string | null;
    kn_hparams?: string | null;
    import_job_id?: string | null;
  }) =>
    request<SupportedModel>("/api/models/local", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  createLocalFolderImport: (body: { source_name: string; total_bytes: number }) =>
    request<ModelDownloadJob>("/api/models/local/import/jobs", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  uploadLocalFolderChunk: (jobId: string, relativePath: string, offset: number, chunk: Blob) => {
    const form = new FormData();
    form.append("relative_path", relativePath);
    form.append("offset", String(offset));
    form.append("chunk", chunk, "model-file-chunk");
    return upload<ModelDownloadJob>(`/api/models/local/import/jobs/${jobId}/chunks`, form);
  },
  completeLocalFolderImport: (jobId: string) =>
    request<ModelDownloadJob>(`/api/models/local/import/jobs/${jobId}/complete`, {
      method: "POST"
    }),
  deleteLocalModel: (body: { model_id: string }) =>
    request<{ status: string; model_id: string }>("/api/models/local", {
      method: "DELETE",
      body: JSON.stringify(body)
    }),
  modelCache: async () => {
    const statuses = await request<ModelCacheStatus[]>("/api/models/cache");
    setModelCacheSnapshot(statuses);
    return statuses;
  },
  downloadModel: (body: { model_id: string }) =>
    request<ModelDownloadJob>("/api/models/cache/download", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  modelDownloadJobs: () => request<ModelDownloadJob[]>("/api/models/cache/jobs"),
  clearModelDownloadJobs: () => request<{ deleted: number }>("/api/models/cache/jobs", {
    method: "DELETE"
  }),
  modelDownloadJob: (jobId: string) => request<ModelDownloadJob>(`/api/models/cache/jobs/${jobId}`),
  cancelModelDownloadJob: (jobId: string) =>
    request<ModelDownloadJob>(`/api/models/cache/jobs/${jobId}/cancel`, {
      method: "POST"
    }),
  deleteModelCache: (body: { model_id: string }) =>
    request<ModelCacheStatus>("/api/models/cache", {
      method: "DELETE",
      body: JSON.stringify(body)
    }),
  modelRuntime: async () => {
    const statuses = await request<ModelRuntimeStatus[]>("/api/models/runtime");
    setModelRuntimeSnapshot(statuses);
    return statuses;
  },
  loadModelRuntime: (body: { model_id: string }) =>
    request<ModelRuntimeJob>("/api/models/runtime/load", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  modelRuntimeJob: (jobId: string) => request<ModelRuntimeJob>(`/api/models/runtime/jobs/${jobId}`),
  waitModelRuntimeJob: (jobId: string, timeoutSeconds = 60) =>
    request<ModelRuntimeJob>(`/api/models/runtime/jobs/${jobId}/wait?timeout=${timeoutSeconds}`),
  unloadModelRuntime: (body: { model_id: string }) =>
    request<ModelRuntimeStatus>("/api/models/runtime", {
      method: "DELETE",
      body: JSON.stringify(body)
    }),
  exportModel: (body: { model_id: string; output_path: string }) =>
    request<{ model_id: string; output_path: string; baked_patches: number }>("/api/models/export", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  downloadModelArchive: (body: { model_id: string }) =>
    download("/api/models/export/archive", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  exportPatch: (patchId: string, body: { output_path: string }) =>
    request<{ patch_id: string; output_path: string; manifest_path: string; delta_files: string[] }>(`/api/patches/${patchId}/export`, {
      method: "POST",
      body: JSON.stringify(body)
    }),
  experimentPage: (offset = 0, status?: string) => request<{ items: Experiment[]; total: number; status_counts: Record<string, number> }>(`/api/experiments/page?offset=${offset}&limit=50${status ? `&status=${encodeURIComponent(status)}` : ""}`),
  patchPage: (offset = 0) => request<{ items: Patch[]; total: number }>(`/api/patches/page?offset=${offset}&limit=50`),
  activePatches: () => request<{ patches: Patch[] }>("/api/patches/active"),
  experiments: () => request<Experiment[]>("/api/experiments"),
  experimentStatus: (experimentId: string) => request<ExperimentDetails>(`/api/experiments/${experimentId}?include_result=false`),
  experiment: (experimentId: string) => request<ExperimentDetails>(`/api/experiments/${experimentId}`),
  deleteExperiment: (experimentId: string, jobId?: string) =>
    request<{ experiment_id: string }>(`/api/experiments/${encodeURIComponent(experimentId)}${jobId ? `?job_id=${encodeURIComponent(jobId)}` : ""}`, {
      method: "DELETE"
    }),
  clearExperiments: () =>
    request<{ deleted_experiment_count: number }>("/api/experiments", {
      method: "DELETE"
    }),
  patches: () => request<{ patches: Patch[] }>("/api/patches"),
  patchEdits: (patchId: string, offset: number, limit: number) =>
    request<PatchEditPage>(`/api/patches/${encodeURIComponent(patchId)}/edits?offset=${offset}&limit=${limit}`),
  loadPatch: (patchId: string) =>
    request<{ model_id: string; active_patches: string[] }>(`/api/patches/${patchId}/load`, {
      method: "POST"
    }),
  discardPatch: (patchId: string) =>
    request<{ model_id: string; active_patches: string[] }>(`/api/patches/${patchId}/discard`, {
      method: "POST"
    }),
  deletePatch: (patchId: string) =>
    request<{ patch_id: string; deleted_patch_ids: string[]; deleted_patch_count: number }>(`/api/patches/${patchId}`, {
      method: "DELETE"
    }),
  revertToPatch: (patchId: string) =>
    request<{ model_id: string; active_patches: string[] }>(`/api/patches/${patchId}/revert-to`, {
      method: "POST"
    }),
  deleteAllPatches: () =>
    request<{ deleted_patch_count: number }>("/api/patches", {
      method: "DELETE"
    }),
  generate: (body: {
    model_id: string;
    prompt: string;
    max_new_tokens: number;
    decoding_strategy: "greedy" | "sampling" | "beam";
    temperature?: number;
    top_p?: number;
    top_k?: number;
    num_beams?: number;
    patch_id?: string | null;
  }) =>
    request<{ experiment_id: string; text: string }>("/api/generate", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  logitLens: (body: { model_id: string; prompt: string; target_token?: string | null; top_k: number }) =>
    request<{ experiment_id: string; layers: LogitLensLayer[] }>("/api/logit-lens", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  causalTrace: (body: {
    model_id: string;
    prompt: string;
    subject: string;
    target: string;
    component: string;
    corruption: { type: string; noise_level: number };
    samples?: number;
    seed?: number;
  }) =>
    request<{ job_id: string; experiment_id: string; status: string; result: { heatmap: TraceCell[]; tokens: string[]; clean_probability: number; corrupted_probability: number; subject_token_indices?: number[]; recommended_layer: number } }>("/api/causal-trace", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  editDefaultLayers: (body: { model_id: string; method: Exclude<EditMethod, "kn"> }) =>
    request<{ layers: number[]; hparams_path: string }>("/api/edit/default-layers", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  mom2StatisticsStatus: (body: Mom2StatisticsRequest) =>
    request<Mom2StatisticsStatus>("/api/edit/mom2/status", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  setMom2StatisticsSource: (body: { model_id: string; stats_source: "base" | "model" }) =>
    request<{ model_id: string; stats_source: "base" | "model"; available_stats_sources: Array<"base" | "model">; base_model_id?: string | null }>("/api/edit/mom2/source", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  setMom2StatisticsPrecision: (body: { model_id: string; method: Mom2Method; mom2_dtype: Mom2Dtype | null }) =>
    request<Mom2StatisticsStatus>("/api/edit/mom2/precision", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  prepareMom2Statistics: (body: Mom2StatisticsRequest) =>
    request<EditJob>("/api/edit/mom2/prepare", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  importMom2Statistics: (body: Mom2StatisticsRequest & { dataset: string; layers: number[]; files: File[] }) => {
    const form = new FormData();
    form.set("model_id", body.model_id);
    form.set("method", body.method);
    form.set("dataset", body.dataset);
    form.set("layers", JSON.stringify(body.layers));
    body.files.forEach((file) => form.append("files", file));
    if (body.stats_source) form.set("stats_source", body.stats_source);
    if (body.mom2_dtype) form.set("mom2_dtype", body.mom2_dtype);
    return upload<Mom2StatisticsStatus & { imported_layers: number[] }>("/api/edit/mom2/import", form);
  },
  mom2Cache: (body: Mom2StatisticsRequest) =>
    request<Mom2Cache>("/api/edit/mom2/cache", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  deleteMom2Statistics: (body: Mom2StatisticsRequest & { layers: number[] }) =>
    request<Mom2Cache & { deleted_layers: number[] }>("/api/edit/mom2/cache", {
      method: "DELETE",
      body: JSON.stringify(body)
    }),
  romeEdit: (body: EditJobOptions & {
    model_id: string;
    subject: string;
    relation_prompt: string;
    new_object: string;
    ground_truth?: string | null;
    target_layer?: number | null;
    evaluation_prompts: string[];
    rephrase_prompts?: string[];
    locality_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    portability_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    sequential_edit?: boolean;
    mom2_adjustment?: boolean;
  }) =>
    request<EditJob>("/api/edit/rome", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  rRomeEdit: (body: EditJobOptions & {
    model_id: string;
    subject: string;
    relation_prompt: string;
    new_object: string;
    ground_truth?: string | null;
    target_layer?: number | null;
    evaluation_prompts: string[];
    rephrase_prompts?: string[];
    locality_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    portability_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    sequential_edit?: boolean;
    mom2_adjustment?: boolean;
  }) =>
    request<EditJob>("/api/edit/r-rome", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  romeSequentialEdits: (body: EditJobOptions & {
    model_id: string;
    edits: Array<{ subject: string; relation_prompt: string; new_object: string; ground_truth?: string | null; target_neg?: string; rephrase_prompts?: string[]; source_metadata?: Record<string, unknown> }>;
    target_layer: number;
    evaluation_prompts: string[];
    sequential_edit?: boolean;
    mom2_adjustment?: boolean;
  }) =>
    request<EditJob>("/api/edit/rome/sequential", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  rRomeSequentialEdits: (body: EditJobOptions & {
    model_id: string;
    edits: Array<{ subject: string; relation_prompt: string; new_object: string; ground_truth?: string | null; target_neg?: string; rephrase_prompts?: string[]; source_metadata?: Record<string, unknown> }>;
    target_layer: number;
    evaluation_prompts: string[];
    sequential_edit?: boolean;
    mom2_adjustment?: boolean;
  }) =>
    request<EditJob>("/api/edit/r-rome/sequential", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  memitEdit: (body: BatchWeightEditRequest) =>
    request<EditJob>("/api/edit/memit", { method: "POST", body: JSON.stringify(body) }),
  batchWeightEdit: (method: BatchWeightMethod, body: BatchWeightEditRequest) =>
    request<EditJob>(`/api/edit/${method.replaceAll("_", "-")}`, { method: "POST", body: JSON.stringify(body) }),
  loraModules: (body: { model_id: string }) =>
    request<LoraModuleCatalog>("/api/edit/lora/modules", { method: "POST", body: JSON.stringify(body) }),
  trainingDefaults: (body: { model_id: string; method: TrainingMethod }) =>
    request<{ layers: number[]; hparams: TrainingHyperparameters; warning?: string | null }>("/api/edit/training-defaults", { method: "POST", body: JSON.stringify(body) }),
  trainingEdit: (method: TrainingMethod, body: EditJobOptions & {
    model_id: string;
    batch_size: number;
    hparams?: TrainingHyperparameters;
    edits: Array<{ subject: string; relation_prompt: string; new_object: string; ground_truth?: string | null; target_neg?: string; rephrase_prompts?: string[]; source_metadata?: Record<string, unknown> }>;
    layers?: number[] | null;
    evaluation_prompts: string[];
    locality_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    portability_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    sequential_edit?: boolean;
  }) => request<EditJob>(`/api/edit/${method}`, { method: "POST", body: JSON.stringify(body) }),
  knEdit: (body: EditJobOptions & {
    model_id: string;
    subject: string;
    relation_prompt: string;
    new_object: string;
    ground_truth: string;
    evaluation_prompts: string[];
    sequential_edit?: boolean;
  }) =>
    request<EditJob>("/api/edit/kn", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  editJob: (jobId: string) => request<EditJob>(`/api/edit/jobs/${jobId}`),
  waitEditJob: (jobId: string, afterLogCount = 0, timeoutSeconds = 30) =>
    request<EditJob>(`/api/edit/jobs/${jobId}/wait?after_log_count=${afterLogCount}&timeout=${timeoutSeconds}`),
  cancelEditJob: (jobId: string) =>
    request<EditJob>(`/api/edit/jobs/${jobId}/cancel`, {
      method: "POST"
    }),
  evaluate: (body: { model_id: string; patch_id: string; prompts: string[] }) =>
    request<{ job_id: string; experiment_id: string; status: string; result: Record<string, unknown> }>("/api/evaluate", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  easyEditReliability: (body: {
    model_id: string;
    method: EditMethod;
    edits: Array<{
      subject: string;
      relation_prompt: string;
      new_object: string;
      ground_truth?: string | null;
      rephrase_prompts?: string[];
    }>;
    include_generations?: boolean;
    evaluation_modes?: Array<"reliability" | "generalization" | "locality" | "portability">;
    evaluation_protocols?: EvaluationProtocols;
  }) =>
    request<EditJob>("/api/evaluate/reliability", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  easyEditReliabilityJob: (jobId: string) =>
    request<EditJob>(`/api/evaluate/reliability/jobs/${encodeURIComponent(jobId)}`),
  easyEditReliabilityPage: (jobId: string, offset: number) =>
    request<ReliabilityTestResultPage>(`/api/evaluate/reliability/jobs/${encodeURIComponent(jobId)}/page?offset=${Math.max(0, Math.floor(offset))}`),
  cancelEvaluationJob: (jobId: string) =>
    request<EditJob>(`/api/evaluate/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST"
    }),
  activeEvaluationFacts: (modelId: string, offset = 0, limit = 50, query = "") =>
    request<ActiveEvaluationFacts>(`/api/evaluate/active-facts/page?model_id=${encodeURIComponent(modelId)}&offset=${offset}&limit=${limit}&query=${encodeURIComponent(query)}`),
  evaluateActiveModel: (body: {
    model_id: string;
    facts?: EvaluationFact[];
    fact_source_id?: string;
    evaluation_modes: EvaluationMetric[];
    evaluation_protocols?: EvaluationProtocols;
    fluency_modes?: EvaluationMetric[];
    rephrase_overrides: Record<string, string>;
    include_generations: boolean;
  }) =>
    request<EditJob>("/api/evaluate/active-model", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  evaluationResultOverview: (experimentId: string) =>
    request<EvaluationResultOverview>(`/api/evaluate/results/${encodeURIComponent(experimentId)}`),
  evaluationResultPage: (experimentId: string, offset: number) =>
    request<EvaluationResultPage>(`/api/evaluate/results/${encodeURIComponent(experimentId)}/page?offset=${offset}`),
  evaluationResultRaw: (experimentId: string) =>
    request<Record<string, unknown>>(`/api/evaluate/results/${encodeURIComponent(experimentId)}/raw`)
};

export type BatchWeightEditRequest = EditJobOptions & {
    model_id: string;
    batch_size?: number;
    edits: Array<{ subject: string; relation_prompt: string; new_object: string; ground_truth?: string | null; target_neg?: string; rephrase_prompts?: string[]; source_metadata?: Record<string, unknown> }>;
    layers?: number[] | null;
    evaluation_prompts: string[];
    locality_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    portability_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    sequential_edit?: boolean;
    mom2_adjustment?: boolean;
  };
