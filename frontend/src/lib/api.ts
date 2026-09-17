import type { ActiveEvaluationFacts, EvaluationResultOverview, EvaluationResultPage, AppSettings, EditJob, Experiment, ExperimentDetails, HfTokenStatus, LogitLensLayer, ModelCacheStatus, ModelDownloadJob, ModelRuntimeJob, ModelRuntimeStatus, Patch, PatchEditPage, ReliabilityTestResultPage, SupportedModel, TraceCell } from "./types";
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
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
  return response.json() as Promise<T>;
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

function subscribeToEvents(path: string, handlers: EventStreamHandlers) {
  let closed = false;
  let controller: AbortController | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;

  const connect = async () => {
    while (!closed) {
      controller = new AbortController();
      try {
        const response = await fetch(backendUrl(path), {
          headers: requestHeaders({ headers: { Accept: "text/event-stream" } }),
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
        retryTimer = setTimeout(resolve, 1000);
      });
      retryTimer = null;
    }
  };

  void connect();
  return () => {
    closed = true;
    controller?.abort();
    if (retryTimer) clearTimeout(retryTimer);
  };
}

export const api = {
  url: backendUrl,
  subscribeToEvents,
  health: () => request<{ status: string }>("/api/health"),
  settings: () => request<AppSettings>("/api/settings"),
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
    supports_kn?: boolean;
    rome_hparams?: string | null;
    r_rome_hparams?: string | null;
    memit_hparams?: string | null;
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
  experiments: () => request<Experiment[]>("/api/experiments"),
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
  editDefaultLayers: (body: { model_id: string; method: "rome" | "r_rome" | "memit" }) =>
    request<{ layers: number[]; hparams_path: string }>("/api/edit/default-layers", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  mom2StatisticsStatus: (body: { model_id: string; method: "rome" | "r_rome" | "memit"; layers?: number[] | null; stats_source?: "base" | "model" | null }) =>
    request<{ model_id: string; method: string; mom2_adjustment: boolean; dataset: string; layers: number[]; ready: boolean; missing_layers: number[]; stats_dir?: string; stats_source: "base" | "model"; available_stats_sources: Array<"base" | "model">; base_model_id?: string | null; cache_model_id: string; statistics: Array<{ layer: number; path: string; ready: boolean }> }>("/api/edit/mom2/status", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  setMom2StatisticsSource: (body: { model_id: string; stats_source: "base" | "model" }) =>
    request<{ model_id: string; stats_source: "base" | "model"; available_stats_sources: Array<"base" | "model">; base_model_id?: string | null }>("/api/edit/mom2/source", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  prepareMom2Statistics: (body: { model_id: string; method: "rome" | "r_rome" | "memit"; layers?: number[] | null; stats_source?: "base" | "model" | null }) =>
    request<EditJob>("/api/edit/mom2/prepare", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  importMom2Statistics: (body: { model_id: string; method: "rome" | "r_rome" | "memit"; dataset: string; layers: number[]; files: File[]; stats_source?: "base" | "model" | null }) => {
    const form = new FormData();
    form.set("model_id", body.model_id);
    form.set("method", body.method);
    form.set("dataset", body.dataset);
    form.set("layers", JSON.stringify(body.layers));
    body.files.forEach((file) => form.append("files", file));
    if (body.stats_source) form.set("stats_source", body.stats_source);
    return upload<{ ready: boolean; missing_layers: number[]; dataset: string; stats_source: "base" | "model"; available_stats_sources: Array<"base" | "model">; base_model_id?: string | null; cache_model_id: string; imported_layers: number[] }>("/api/edit/mom2/import", form);
  },
  mom2Cache: (body: { model_id: string; method: "rome" | "r_rome" | "memit"; stats_source?: "base" | "model" | null }) =>
    request<{ model_id: string; method: string; dataset: string; stats_dir: string; stats_source: "base" | "model"; available_stats_sources: Array<"base" | "model">; base_model_id?: string | null; cache_model_id: string; statistics: Array<{ layer: number; path: string; filename: string; size_bytes: number; modified_at: number }> }>("/api/edit/mom2/cache", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  deleteMom2Statistics: (body: { model_id: string; method: "rome" | "r_rome" | "memit"; layers: number[]; stats_source?: "base" | "model" | null }) =>
    request<{ model_id: string; method: string; dataset: string; stats_dir: string; stats_source: "base" | "model"; available_stats_sources: Array<"base" | "model">; base_model_id?: string | null; cache_model_id: string; deleted_layers: number[]; statistics: Array<{ layer: number; path: string; filename: string; size_bytes: number; modified_at: number }> }>("/api/edit/mom2/cache", {
      method: "DELETE",
      body: JSON.stringify(body)
    }),
  romeEdit: (body: {
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
  rRomeEdit: (body: {
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
  romeSequentialEdits: (body: {
    model_id: string;
    edits: Array<{ subject: string; relation_prompt: string; new_object: string; ground_truth?: string | null; rephrase_prompts?: string[]; source_metadata?: Record<string, unknown> }>;
    target_layer: number;
    evaluation_prompts: string[];
    sequential_edit?: boolean;
    mom2_adjustment?: boolean;
  }) =>
    request<EditJob>("/api/edit/rome/sequential", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  rRomeSequentialEdits: (body: {
    model_id: string;
    edits: Array<{ subject: string; relation_prompt: string; new_object: string; ground_truth?: string | null; rephrase_prompts?: string[]; source_metadata?: Record<string, unknown> }>;
    target_layer: number;
    evaluation_prompts: string[];
    sequential_edit?: boolean;
    mom2_adjustment?: boolean;
  }) =>
    request<EditJob>("/api/edit/r-rome/sequential", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  memitEdit: (body: {
    model_id: string;
    edits: Array<{ subject: string; relation_prompt: string; new_object: string; ground_truth?: string | null; rephrase_prompts?: string[]; source_metadata?: Record<string, unknown> }>;
    layers?: number[] | null;
    evaluation_prompts: string[];
    locality_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    portability_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    sequential_edit?: boolean;
    mom2_adjustment?: boolean;
  }) =>
    request<EditJob>("/api/edit/memit", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  knEdit: (body: {
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
    method: "rome" | "r_rome" | "memit" | "kn";
    edits: Array<{
      subject: string;
      relation_prompt: string;
      new_object: string;
      ground_truth?: string | null;
      rephrase_prompts?: string[];
    }>;
    include_generations?: boolean;
    evaluation_modes?: Array<"reliability" | "generalization" | "locality" | "portability">;
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
  activeEvaluationFacts: (modelId: string) =>
    request<ActiveEvaluationFacts>(`/api/evaluate/active-facts?model_id=${encodeURIComponent(modelId)}`),
  evaluateActiveModel: (body: {
    model_id: string;
    evaluation_modes: Array<"reliability" | "generalization" | "locality" | "portability">;
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
