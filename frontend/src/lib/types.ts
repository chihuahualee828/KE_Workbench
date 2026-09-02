export type SupportedModel = {
  id: string;
  name: string;
  hf_id: string;
  local_path?: string;
  source?: "hub" | "local";
  model_type?: string;
  n_layers: number;
  supports_generation: boolean;
  supports_logit_lens: boolean;
  supports_causal_trace: boolean;
  supports_rome: boolean;
  supports_memit: boolean;
  supports_kn: boolean;
  easyedit?: {
    rome_hparams?: string;
    memit_hparams?: string;
    kn_hparams?: string;
    estimated_vram_gb?: Record<string, number>;
  };
  lineage_imported?: number;
};

export type Experiment = {
  id: string;
  type: string;
  model_id: string;
  status: "running" | "completed" | "failed";
  created_at: string;
  updated_at?: string;
  input_json?: Record<string, unknown>;
  result_path?: string | null;
  error_message?: string | null;
};

export type ExperimentJob = {
  id: string;
  experiment_id: string;
  status: "running" | "completed" | "failed";
  progress: number;
  logs: string[];
  created_at: string;
  updated_at: string;
};

export type ExperimentDetails = Experiment & {
  job?: ExperimentJob | null;
  result?: Record<string, unknown> | null;
};

export type Patch = {
  patch_id: string;
  base_model: string;
  method: string;
  edits: Array<{
    subject: string;
    relation_prompt: string;
    new_object: string;
    ground_truth?: string;
    rephrase_prompts?: string[];
  }>;
  modified_layers: number[];
  created_at: string;
  metrics?: Record<string, unknown>;
  delta_paths?: string[];
  runtime?: string;
  sequential_edit?: boolean;
  mom2_adjustment?: boolean;
  parent_patch_ids?: string[];
  layer_source?: string;
  active?: boolean;
  active_index?: number | null;
  lineage_only?: boolean;
  baked_into_model?: boolean;
};

export type ModelCacheStatus = {
  model_id: string;
  hf_id: string;
  cached: boolean;
  cache_state: "not_cached" | "partial" | "queued" | "downloading" | "canceling" | "canceled" | "cached" | "failed";
  cache_path: string;
  size_bytes: number;
  hub_size_bytes?: number | null;
  snapshot_count: number;
  has_model_weights: boolean;
  has_config: boolean;
  incomplete_file_count: number;
  active_job?: ModelDownloadJob | null;
  is_local?: boolean;
};

export type ModelDownloadJob = {
  job_id: string;
  model_id: string;
  hf_id: string;
  status: "queued" | "downloading" | "canceling" | "canceled" | "completed" | "failed";
  message: string;
  created_at: string;
  updated_at: string;
  cache?: ModelCacheStatus;
};

export type ModelRuntimeStatus = {
  model_id: string;
  hf_id: string;
  loaded: boolean;
  tokenizer_loaded: boolean;
  device?: string;
  model_vram_bytes?: number;
  gpu?: {
    cuda_available: boolean;
    device?: string;
    device_name?: string;
    memory_source?: string;
    used_bytes?: number;
    allocated_bytes?: number;
    reserved_bytes?: number;
    free_bytes?: number;
    total_bytes?: number;
  };
  runtime_state: "not_loaded" | "queued" | "loading" | "loaded" | "completed" | "failed";
  active_job?: ModelRuntimeJob | null;
};

export type ModelRuntimeJob = {
  job_id: string;
  model_id: string;
  hf_id: string;
  status: "queued" | "loading" | "completed" | "failed";
  message: string;
  device?: string;
  created_at: string;
  updated_at: string;
};

export type EditJob = {
  job_id: string;
  experiment_id: string;
  status: "running" | "completed" | "failed";
  progress: number;
  logs: string[];
  result?: Record<string, unknown> | null;
};

export type HfTokenStatus = {
  configured: boolean;
  source: "runtime" | "env" | "none";
};

export type AppSettings = {
  app_env: string;
  editing_runtime: string;
  easyedit_root: string;
  easyedit_available: boolean;
  data_dir: string;
  install_easyedit: string;
  hf_token: HfTokenStatus;
};

export type LogitLensLayer = {
  layer: number;
  target_rank: number | null;
  target_logit: number | null;
  target_probability: number | null;
  top_tokens: Array<[string, number]>;
};

export type TraceCell = {
  layer: number;
  token_index: number;
  token: string;
  restored_probability: number;
  indirect_effect: number;
};
