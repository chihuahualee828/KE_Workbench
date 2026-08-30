import type { AppSettings, EditJob, Experiment, HfTokenStatus, LogitLensLayer, ModelCacheStatus, ModelDownloadJob, ModelRuntimeJob, ModelRuntimeStatus, Patch, SupportedModel, TraceCell } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
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

export const api = {
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
    id: string;
    name: string;
    local_path: string;
    model_type: string;
    n_layers: number;
    supports_generation: boolean;
    supports_logit_lens: boolean;
    supports_causal_trace: boolean;
    supports_rome: boolean;
    supports_memit: boolean;
    rome_hparams?: string | null;
    memit_hparams?: string | null;
  }) =>
    request<SupportedModel>("/api/models/local", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  deleteLocalModel: (body: { model_id: string }) =>
    request<{ status: string; model_id: string }>("/api/models/local", {
      method: "DELETE",
      body: JSON.stringify(body)
    }),
  modelCache: () => request<ModelCacheStatus[]>("/api/models/cache"),
  downloadModel: (body: { model_id: string }) =>
    request<ModelDownloadJob>("/api/models/cache/download", {
      method: "POST",
      body: JSON.stringify(body)
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
  modelRuntime: () => request<ModelRuntimeStatus[]>("/api/models/runtime"),
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
    request<{ model_id: string; output_path: string; lineage_path: string; active_patches: string[] }>("/api/models/export", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  exportPatch: (patchId: string, body: { output_path: string }) =>
    request<{ patch_id: string; output_path: string; manifest_path: string; delta_files: string[] }>(`/api/patches/${patchId}/export`, {
      method: "POST",
      body: JSON.stringify(body)
    }),
  experiments: () => request<Experiment[]>("/api/experiments"),
  patches: () => request<{ patches: Patch[] }>("/api/patches"),
  discardPatch: (patchId: string) =>
    request<{ model_id: string; active_patches: string[] }>(`/api/patches/${patchId}/discard`, {
      method: "POST"
    }),
  revertToPatch: (patchId: string) =>
    request<{ model_id: string; active_patches: string[] }>(`/api/patches/${patchId}/revert-to`, {
      method: "POST"
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
  }) =>
    request<{ job_id: string; experiment_id: string; status: string; result: { heatmap: TraceCell[]; tokens: string[]; clean_probability: number; corrupted_probability: number } }>("/api/causal-trace", {
      method: "POST",
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
  }) =>
    request<EditJob>("/api/edit/rome", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  memitEdit: (body: {
    model_id: string;
    edits: Array<{ subject: string; relation_prompt: string; new_object: string; ground_truth?: string | null; rephrase_prompts?: string[] }>;
    layers?: number[] | null;
    locality_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    portability_inputs?: Record<string, { prompt: string[]; ground_truth: string[] }> | null;
    sequential_edit?: boolean;
  }) =>
    request<EditJob>("/api/edit/memit", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  editJob: (jobId: string) => request<EditJob>(`/api/edit/jobs/${jobId}`),
  waitEditJob: (jobId: string, afterLogCount = 0, timeoutSeconds = 30) =>
    request<EditJob>(`/api/edit/jobs/${jobId}/wait?after_log_count=${afterLogCount}&timeout=${timeoutSeconds}`),
  evaluate: (body: { model_id: string; patch_id: string; prompts: string[] }) =>
    request<{ job_id: string; experiment_id: string; status: string; result: Record<string, unknown> }>("/api/evaluate", {
      method: "POST",
      body: JSON.stringify(body)
    })
};
