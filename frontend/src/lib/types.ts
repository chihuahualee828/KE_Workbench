import type { Mom2Method } from "./editingMethods";
export type SupportedModel = {
  id: string;
  name: string;
  hf_id: string;
  local_path?: string;
  source?: "hub" | "local";
  managed_local?: boolean;
  base_model_id?: string;
  architecture_model_type?: string;
  import_metadata?: {
    import_id?: string;
    uploaded_at?: string;
    source_folder?: string;
  };
  model_type?: string;
  n_layers: number;
  supports_generation: boolean;
  supports_logit_lens: boolean;
  supports_causal_trace: boolean;
  supports_rome: boolean;
  supports_r_rome: boolean;
  supports_memit: boolean;
  supports_ft: boolean;
  supports_lora: boolean;
  supports_emmet?: boolean;
  supports_memit_fe?: boolean;
  supports_pmet?: boolean;
  supports_dpo?: boolean;
  supports_eamet?: boolean;
  supports_alphaedit?: boolean;
  supports_sphere?: boolean;
  supports_kn: boolean;
  easyedit?: {
    rome_hparams?: string;
    r_rome_hparams?: string;
    memit_hparams?: string;
    ft_hparams?: string;
    lora_hparams?: string;
    emmet_hparams?: string;
    memit_fe_hparams?: string;
    pmet_hparams?: string;
    dpo_hparams?: string;
    eamet_hparams?: string;
    alphaedit_hparams?: string;
    sphere_hparams?: string;
    kn_hparams?: string;
    estimated_vram_gb?: Record<string, number>;
  };
  lineage_imported?: number;
};

export type Mom2Dtype = "float16" | "float32" | "float64";

export type Mom2StatisticsRequest = {
  model_id: string;
  method: Mom2Method;
  layers?: number[] | null;
  stats_source?: "base" | "model" | null;
  mom2_dtype?: Mom2Dtype | null;
};

export type Mom2Profile = {
  model_id: string;
  method: string;
  dataset: string;
  stats_dir: string;
  stats_source: "base" | "model";
  available_stats_sources: Array<"base" | "model">;
  base_model_id?: string | null;
  cache_model_id: string;
  mom2_dtype: Mom2Dtype;
  default_mom2_dtype: Mom2Dtype;
  available_mom2_dtypes: Mom2Dtype[];
  device: string;
};

export type Mom2StatisticsStatus = Mom2Profile & {
  mom2_adjustment: boolean;
  layers: number[];
  ready: boolean;
  missing_layers: number[];
  statistics: Array<{ layer: number; path: string; ready: boolean }>;
};

export type Mom2Cache = Mom2Profile & {
  statistics: Array<{ layer: number; path: string; filename: string; size_bytes: number; modified_at: number; mom2_dtype: Mom2Dtype }>;
};

export type Experiment = {
  id: string;
  type: string;
  model_id: string;
  status: "running" | "completed" | "failed" | "canceled";
  created_at: string;
  updated_at?: string;
  input_json?: Record<string, unknown>;
  result_path?: string | null;
  error_message?: string | null;
  job?: ExperimentJobSummary;
};

export type ExperimentJobSummary = {
  progress_phase?: string | null;
  id: string;
  experiment_id: string;
  status: "running" | "completed" | "failed" | "canceled";
  cancel_requested?: boolean;
  progress: number;
  progress_current?: number | null;
  progress_total?: number | null;
  created_at: string;
  updated_at: string;
};

export type ExperimentJob = {
  progress_phase?: string | null;
  id: string;
  experiment_id: string;
  status: "running" | "completed" | "failed" | "canceled";
  cancel_requested?: boolean;
  progress: number;
  progress_current?: number | null;
  progress_total?: number | null;
  logs: string[];
  created_at: string;
  updated_at: string;
};

export type ExperimentDetails = Omit<Experiment, "job"> & {
  job?: ExperimentJob | null;
  result?: Record<string, unknown> | null;
};

export type PatchEdit = {
  subject: string;
  relation_prompt: string;
  new_object: string;
  ground_truth?: string;
  rephrase_prompts?: string[];
};

export type PatchEditPage = {
  patch_id: string;
  offset: number;
  limit: number;
  total: number;
  items: PatchEdit[];
};

export type Patch = {
  lineage_child_count?: number;
  seed?: number;
  patch_id: string;
  base_model: string;
  method: string;
  edits: PatchEdit[];
  edit_count?: number;
  modified_layers: number[];
  created_at: string;
  metrics?: Record<string, unknown>;
  delta_paths?: string[];
  tensor_size_bytes?: number | null;
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

export type PatchLibraryQuery = {
  query?: string;
  method?: string;
  model_id?: string;
  sort?: "newest" | "oldest";
};

export type PatchLibraryPage = {
  items: Patch[];
  total: number;
  methods?: string[];
  model_ids?: string[];
};

export type PatchDeletionPreview = {
  selected_patch_ids: string[];
  deleted_patch_ids: string[];
  dependent_patch_ids: string[];
  blocked_patch_ids: string[];
  preview_token: string;
  patches: Array<{
    patch_id: string;
    base_model: string;
    active: boolean;
    baked_into_model: boolean;
    included_by: string[];
  }>;
};

export type PatchBatchDeletion = {
  deleted_patch_ids: string[];
  deleted_patch_count: number;
  cleanup_pending: boolean;
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
  managed_local?: boolean;
};

export type ModelDownloadJob = {
  job_id: string;
  model_id: string;
  hf_id: string;
  job_type?: "download" | "local_import";
  source_name?: string;
  source_kind?: "folder";
  status: "queued" | "downloading" | "uploading" | "validating" | "registering" | "canceling" | "canceled" | "completed" | "failed";
  message: string;
  created_at: string;
  updated_at: string;
  cache?: ModelCacheStatus;
  progress_current_bytes?: number;
  progress_total_bytes?: number;
  preview?: LocalModelImportPreview;
  registered_model_id?: string;
};

export type LocalModelImportPreview = {
  id: string;
  name: string;
  display_base_name?: string;
  local_path: string;
  base_model_id?: string;
  architecture_model_type?: string;
  model_type: string;
  n_layers: number;
  supports_generation: boolean;
  supports_logit_lens: boolean;
  supports_causal_trace: boolean;
  supports_rome: boolean;
  supports_r_rome: boolean;
  supports_memit: boolean;
  supports_ft: boolean;
  supports_lora: boolean;
  supports_emmet?: boolean;
  supports_memit_fe?: boolean;
  supports_pmet?: boolean;
  supports_dpo?: boolean;
  supports_eamet?: boolean;
  supports_alphaedit?: boolean;
  supports_sphere?: boolean;
  supports_kn: boolean;
  trust_remote_code?: boolean;
  generation_use_cache?: boolean;
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
    mps_available?: boolean;
    driver_allocated_bytes?: number;
    recommended_max_bytes?: number;
    sampled_at?: number;
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
  patch_state?: { version: number; model_id: string; active_patch_ids: string[] } | null;
  progress_phase?: string | null;
  job_id: string;
  experiment_id: string;
  status: "running" | "completed" | "failed" | "canceled";
  cancel_requested?: boolean;
  progress: number;
  progress_current?: number | null;
  progress_total?: number | null;
  logs: string[];
  log_count: number;
  log_start_count: number;
  result_item_count?: number;
  result_page_size?: number | null;
  result?: Record<string, unknown> | null;
};

export type ReliabilityTestResultItem = {
  index: number;
  generation?: {
    prompt: string;
    active_model: string;
  } | null;
  case?: {
    prompt: string;
    target_new: string;
    reliability_score?: number | null;
  } | null;
};

export type ReliabilityTestResultPage = {
  offset: number;
  limit: number;
  total: number;
  items: ReliabilityTestResultItem[];
};

export type EvaluationMetric = "reliability" | "generalization" | "locality" | "portability";
export type EvaluationProtocol = "traditional" | "LLM-judge" | "generate-text";
export type EvaluationProtocols = Partial<Record<EvaluationMetric, EvaluationProtocol>>;

export type EvaluationFact = {
  key: string;
  subject: string;
  relation_prompt: string;
  prompt: string;
  new_object: string;
  ground_truth: string;
  rephrase_prompt: string;
  locality_prompt: string;
  locality_ground_truth: string;
  portability_prompt: string;
  portability_ground_truth: string;
};

export type ActiveEvaluationFacts = {
  model_id: string;
  active_patch_ids: string[];
  method: string;
  total: number;
  facts: EvaluationFact[];
  locality_count: number;
  portability_count: number;
  matches: number[];
  match_total: number;
};

export type EvaluationResultOverview = {
  model_id: string;
  method: string;
  active_patch_ids: string[];
  metrics: Record<string, unknown>;
  evaluation_modes: EvaluationMetric[];
  evaluation_protocols?: EvaluationProtocols;
  total: number;
  available_generation_metrics: EvaluationMetric[];
};

export type EvaluationGeneration = {
  prompt: string;
  clean_model: string;
  active_model: string;
};

export type EvaluationResultItem = {
  portability_generation?: EvaluationGeneration | null;
  portability_score?: number | null;
  evaluation_protocols?: EvaluationProtocols;
  fluency?: Partial<Record<EvaluationMetric, { ngram_entropy?: number; error?: string }>>;
  index: number;
  fact: EvaluationFact;
  reliability_generation: EvaluationGeneration | null;
  generalization_generation: EvaluationGeneration | null;
  locality_generation: EvaluationGeneration | null;
  reliability_score: number | null;
  generalization_score: number | null;
  locality_score: number | null;
};

export type EvaluationResultPage = {
  offset: number;
  limit: number;
  total: number;
  items: EvaluationResultItem[];
};

export type HfTokenStatus = {
  configured: boolean;
  source: "runtime" | "env" | "none";
};

export type ModelCacheLocation = {
  download_dir: string;
  environment_dir: string;
  search_dirs: string[];
  source: "settings" | "environment";
};

export type AppSettings = {
  app_env: string;
  editing_runtime: string;
  easyedit_root: string;
  easyedit_available: boolean;
  data_dir: string;
  install_easyedit: string;
  hf_token: HfTokenStatus;
  model_cache: ModelCacheLocation;
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
