"use client";

import { Download, FolderUp, Loader2, PlusCircle, RefreshCw, Trash2, Upload, XCircle } from "lucide-react";
import { Fragment, useEffect, useRef, useState } from "react";

import { cacheResource, downloadResource, runtimeResource } from "@/lib/workbenchResources";
import { api } from "@/lib/api";
import { announceModelCacheStatus, announceModelDownload, applyDownloadJobToCache, listenForModelDownloads } from "@/lib/modelDownloadEvents";
import { announceModelRuntimeLoad, announceModelRuntimeStatuses, announceModelRuntimeUnloading, isModelRuntimeConnected, listenForModelRuntimeConnection, listenForModelRuntimeLoads, listenForModelRuntimeStatuses, listenForModelRuntimeUnloading, watchModelRuntimeJob } from "@/lib/modelRuntimeEvents";
import type { ModelCacheStatus, ModelDownloadJob, ModelRuntimeJob, SupportedModel } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";
import { GpuVramMonitor } from "./GpuVramMonitor";

const activeDownloadStatuses = ["queued", "downloading", "uploading", "validating", "registering", "canceling"];
const folderImportChunkSize = 8 * 1024 * 1024;
const supportedConfigModelTypes: Record<string, string> = {
  gpt2: "gpt2",
  gptj: "gptj",
  baichuan: "baichuan",
  chatglm: "chatglm",
  internlm: "internlm",
  llama: "llama",
  mistral: "mistral",
  qwen: "qwen",
  qwen2: "qwen",
  qwen2_moe: "qwen",
  qwen3: "qwen",
  qwen3_moe: "qwen"
};

export function ModelCacheManager({
  models,
  apiStatus,
  onModelsChanged
}: {
  models: SupportedModel[];
  apiStatus: "loading" | "ready" | "offline";
  onModelsChanged?: () => void | Promise<void>;
}) {
  const [cache, setCache] = cacheResource.use();
  const [runtime, setRuntime] = runtimeResource.use();
  const [jobs, setJobs] = useState<Record<string, ModelDownloadJob>>({});
  const [runtimeJobs, setRuntimeJobs] = useState<Record<string, ModelRuntimeJob>>({});
  const [unloadingModelId, setUnloadingModelId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [runtimeConnected, setRuntimeConnected] = useState(isModelRuntimeConnected);
  const [importing, setImporting] = useState(false);
  const [selectedFolderFiles, setSelectedFolderFiles] = useState<File[]>([]);
  const [selectedFolderError, setSelectedFolderError] = useState("");
  const [checkingFolder, setCheckingFolder] = useState(false);
  const [selectedLocalModel, setSelectedLocalModel] = useState<SupportedModel | null>(null);
  const localImportInputRef = useRef<HTMLInputElement>(null);
  const [localName, setLocalName] = usePersistentState("kew:modelManager:localName", "");
  const [localPath, setLocalPath] = usePersistentState("kew:modelManager:localPath", "");
  const [localModelType, setLocalModelType] = usePersistentState("kew:modelManager:localModelType", "gpt2");
  const [localLayers, setLocalLayers] = usePersistentState("kew:modelManager:localLayers", "0");

  useEffect(() => {
    configureLocalFolderInput(localImportInputRef.current);
  }, []);

  function openLocalFolderPicker() {
    const input = localImportInputRef.current;
    configureLocalFolderInput(input);
    input?.click();
  }

  async function refresh() {
    if (!runtimeConnected) return;
    setError("");
    try {
      const [result, runtimeResult] = await Promise.all([
        api.modelCache(),
        api.modelRuntime()
      ]);
      setCache(result);
      setRuntime(runtimeResult);
      const activeRuntimeJobs = runtimeResult
        .map((status) => status.active_job)
        .filter((job): job is ModelRuntimeJob => Boolean(job));
      activeRuntimeJobs.forEach((job) => {
        watchModelRuntimeJob(job).catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
        });
      });
      setJobs((current) => {
        const next = { ...current };
        result.forEach((status) => {
          if (status.active_job) {
            next[status.active_job.job_id] = status.active_job;
          }
        });
        return next;
      });
      setRuntimeJobs((current) => {
        const next = { ...current };
        activeRuntimeJobs.forEach((job) => {
          next[job.job_id] = job;
        });
        return next;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function download(modelId: string) {
    if (!runtimeConnected) return;
    setError("");
    try {
      const job = await api.downloadModel({ model_id: modelId });
      setJobs((current) => ({ ...current, [job.job_id]: job }));
      if (activeDownloadStatuses.includes(job.status)) {
        setCache((current) => applyDownloadJobToCache(current, job));
        announceModelDownload(job);
      }
      void refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function deleteCache(modelId: string) {
    if (!runtimeConnected) return;
    setError("");
    try {
      const status = await api.deleteModelCache({ model_id: modelId });
      announceModelCacheStatus(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function loadRuntime(modelId: string) {
    if (!runtimeConnected) return;
    setError("");
    try {
      const job = await api.loadModelRuntime({ model_id: modelId });
      setRuntimeJobs((current) => ({ ...current, [job.job_id]: job }));
      announceModelRuntimeLoad(job);
      watchModelRuntimeJob(job).catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function unloadRuntime(modelId: string) {
    if (!runtimeConnected || unloadingModelId) return;
    setError("");
    setUnloadingModelId(modelId);
    announceModelRuntimeUnloading(modelId, true);
    try {
      await api.unloadModelRuntime({ model_id: modelId });
      const runtimeResult = await api.modelRuntime();
      setRuntime(runtimeResult);
      announceModelRuntimeStatuses(runtimeResult);
      await onModelsChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUnloadingModelId(null);
      announceModelRuntimeUnloading(modelId, false);
    }
  }

  function updateImportJob(job: ModelDownloadJob) {
    setJobs((current) => ({ ...current, [job.job_id]: job }));
    announceModelDownload(job);
  }

  async function selectLocalModelFolder(files: FileList | null) {
    const selectedFiles = files ? Array.from(files) : [];
    if (!selectedFiles.length || importing) return;
    setError("");
    setSelectedFolderFiles(selectedFiles);
    setSelectedFolderError("");
    setCheckingFolder(true);
    try {
      const sourcePath = selectedFolderSourcePath(selectedFiles);
      const configFile = selectedFolderConfigFile(selectedFiles);
      if (!configFile) {
        throw new Error("The selected folder must contain config.json directly under its root.");
      }
      const config = JSON.parse(await configFile.text()) as Record<string, unknown>;
      const rawModelType = typeof config.model_type === "string" ? config.model_type.toLowerCase() : "";
      const modelType = supportedConfigModelTypes[rawModelType];
      if (!modelType || config.is_encoder_decoder === true) {
        throw new Error("This folder is not a currently supported decoder-only causal language model.");
      }
      const layerCount = configuredLayerCount(config);
      if (!layerCount) {
        throw new Error("config.json must include a positive transformer layer count.");
      }
      setLocalName(displayNameFromFolderConfig(config, modelType, layerCount, sourcePath));
      setLocalPath(sourcePath);
      setLocalModelType(modelType);
      setLocalLayers(String(layerCount));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setSelectedFolderError(message);
      setError(message);
    } finally {
      setCheckingFolder(false);
    }
  }

  async function addLocalModel() {
    if (!runtimeConnected) return;
    setError("");
    if (selectedFolderFiles.length > 0) {
      if (selectedFolderError) return;
      setImporting(true);
      try {
        const totalBytes = selectedFolderFiles.reduce((total, file) => total + file.size, 0);
        let job = await api.createLocalFolderImport({
          source_name: localPath || selectedFolderSourcePath(selectedFolderFiles),
          total_bytes: totalBytes
        });
        updateImportJob(job);
        for (const file of selectedFolderFiles) {
          const relativePath = file.webkitRelativePath || file.name;
          if (file.size === 0) {
            job = await api.uploadLocalFolderChunk(job.job_id, relativePath, 0, file.slice(0, 0));
            updateImportJob(job);
            continue;
          }
          for (let offset = 0; offset < file.size; offset += folderImportChunkSize) {
            const chunk = file.slice(offset, Math.min(offset + folderImportChunkSize, file.size));
            job = await api.uploadLocalFolderChunk(job.job_id, relativePath, offset, chunk);
            updateImportJob(job);
          }
        }
        job = await api.completeLocalFolderImport(job.job_id);
        updateImportJob(job);
        if (job.status !== "completed" || !job.preview) {
          throw new Error(job.message || "Local model folder could not be validated.");
        }
        const preview = job.preview;
        const registered = await api.addLocalModel({
          id: preview.id,
          name: localName.trim() || preview.display_base_name || preview.name,
          local_path: preview.local_path,
          model_type: preview.model_type,
          n_layers: preview.n_layers,
          supports_generation: true,
          import_job_id: job.job_id
        });
        updateImportJob(await api.modelDownloadJob(job.job_id));
        setLocalName(registered.name);
        setLocalPath("");
        setSelectedFolderFiles([]);
        await onModelsChanged?.();
        await refresh();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setImporting(false);
      }
      return;
    }
    try {
      await api.addLocalModel({
        name: localName.trim(),
        local_path: localPath,
        model_type: localModelType || "auto",
        n_layers: Number(localLayers) || 0,
        supports_generation: true
      });
      await onModelsChanged?.();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function removeLocalModel(modelId: string) {
    if (!runtimeConnected) return;
    setError("");
    try {
      await api.deleteLocalModel({ model_id: modelId });
      await onModelsChanged?.();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    if (apiStatus === "ready") {
      void refresh();
    }
  }, [apiStatus]);

  useEffect(() => {
    return listenForModelDownloads((job) => {
      setJobs((current) => ({ ...current, [job.job_id]: job }));
      setCache((current) => applyDownloadJobToCache(current, job));
    });
  }, []);

  useEffect(() => {
    return listenForModelRuntimeLoads((job) => {
      setRuntimeJobs((current) => ({ ...current, [job.job_id]: job }));
      if (!["queued", "loading"].includes(job.status)) {
        void refresh();
      }
    });
  }, []);

  useEffect(() => {
    return listenForModelRuntimeStatuses((status) => {
    });
  }, []);

  useEffect(() => {
    return listenForModelRuntimeUnloading(({ modelId, unloading }) => {
      setUnloadingModelId(unloading ? modelId : null);
    });
  }, []);

  useEffect(() => {
    return listenForModelRuntimeConnection((connected) => {
      setRuntimeConnected(connected);
      if (connected) {
        setError((current) => current === "Failed to fetch" ? "" : current);
        setRuntimeJobs((current) => {
          const next: Record<string, ModelRuntimeJob> = {};
          Object.entries(current).forEach(([jobId, job]) => {
            if (job.status !== "failed") next[jobId] = job;
          });
          return next;
        });
        return;
      }
      setError("Failed to fetch");
      setRuntimeJobs((current) => {
        const next: Record<string, ModelRuntimeJob> = {};
        Object.entries(current).forEach(([jobId, job]) => {
          next[jobId] = ["queued", "loading"].includes(job.status)
            ? { ...job, status: "failed", message: "Failed to fetch", updated_at: new Date().toISOString() }
            : job;
        });
        return next;
      });
      setRuntime((current) => current.map((status) =>
        ["queued", "loading"].includes(status.runtime_state)
          ? { ...status, loaded: false, runtime_state: "failed", active_job: null }
          : status
      ));
    });
  }, []);

  const [sharedDownloads] = downloadResource.use();
  useEffect(() => {
    setJobs(Object.fromEntries(sharedDownloads.map((job) => [job.job_id, job])));
  }, [sharedDownloads]);

  const localModels = models.filter((model) => model.source === "local");
  const firstLocalModelId = localModels[0]?.id;

  return (
    <section className="panel">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">Model Cache</h2>
        <button className="ghost-button min-h-9 px-2 py-1" onClick={refresh} disabled={apiStatus !== "ready" || !runtimeConnected} title={apiStatus !== "ready" ? "Wait for the backend connection" : runtimeConnected ? "Refresh model cache" : "API is offline"}>
          <RefreshCw size={15} />
          Refresh
        </button>
      </div>
      {apiStatus !== "ready" ? (
        <div className="rounded-md border border-line bg-panel p-4 text-sm text-ink/60">
          {apiStatus === "loading" ? "Connecting to backend..." : "Backend is offline. Model cache is unavailable."}
        </div>
      ) : (
        <>
          <GpuVramMonitor runtime={runtime} />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1050px] table-fixed border-collapse text-sm">
              <colgroup>
                <col className="w-[13rem]" />
                <col className="w-[6.25rem]" />
                <col className="w-[7rem]" />
                <col className="w-[4.125rem]" />
                <col />
                <col className="w-[15.25rem]" />
              </colgroup>
              <thead>
                <tr className="border-b border-line text-left">
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2 pr-4">Cache</th>
                  <th className="py-2 pr-4">Runtime</th>
                  <th className="py-2 pr-4">Size</th>
                  <th className="py-2 pr-4">Path</th>
                  <th className="whitespace-nowrap py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => {
                  const status = cache.find((item) => item.model_id === model.id);
                  const runtimeStatus = runtime.find((item) => item.model_id === model.id);
                  const job = Object.values(jobs).reverse().find((item) => item.model_id === model.id);
                  const activeDownloadJob = job && activeDownloadStatuses.includes(job.status) ? job : status?.active_job;
                  const runtimeJob = Object.values(runtimeJobs).reverse().find((item) => item.model_id === model.id);
                  const active = Boolean(activeDownloadJob) || activeDownloadStatuses.includes(status?.cache_state ?? "not_cached");
                  const loading = Boolean(runtimeJob && ["queued", "loading"].includes(runtimeJob.status)) || ["queued", "loading"].includes(runtimeStatus?.runtime_state ?? "not_loaded");
                  const unloading = unloadingModelId === model.id;
                  const runtimeError = runtimeJob?.status === "failed" && runtimeJob.message !== "Failed to fetch" ? runtimeJob.message : "";
                  const downloadError = job?.status === "failed" ? job.message : "";
                  const cached = Boolean(status?.cached);
                  const loaded = Boolean(runtimeStatus?.loaded);
                  const local = model.source === "local" || status?.is_local;
                  const managedLocal = Boolean(model.managed_local || status?.managed_local);
                  const modelReference = local
                    ? model.base_model_id || model.architecture_model_type || model.model_type || "Local model"
                    : model.hf_id;
                  return (
                    <Fragment key={model.id}>
                    {model.id === firstLocalModelId ? (
                      <tr>
                        <td colSpan={6} className="border-y-2 border-line bg-panel/60 px-2 py-2 text-xs font-semibold uppercase tracking-wide text-ink/60">
                          Local models ({localModels.length})
                        </td>
                      </tr>
                    ) : null}
                    <tr className="border-b border-line/70 align-top">
                  <td className="py-2 pr-4">
                    {local ? (
                      <button className="text-left transition hover:text-cyan" onClick={() => setSelectedLocalModel(model)} title="Show local model details" type="button">
                        <div className="font-medium">{model.name}</div>
                        <div className="font-mono text-xs text-ink/50">{modelReference}</div>
                      </button>
                    ) : (
                      <>
                        <div className="font-medium">{model.name}</div>
                        <div className="font-mono text-xs text-ink/50">{modelReference}</div>
                      </>
                    )}
                  </td>
                  <td className="py-2 pr-4">{formatCacheState(status)}</td>
                  <td className="py-2 pr-4">
                    {loaded ? (
                      <div>
                        <div>{`Loaded (${runtimeStatus?.device ?? "cpu"})`}</div>
                        <div className="mt-1 text-xs text-ink/60">{formatBytes(runtimeStatus?.model_vram_bytes ?? 0)} model VRAM</div>
                      </div>
                    ) : unloading ? "Unloading" : loading ? "Loading" : runtimeError || "Not loaded"}
                  </td>
                  <td className="py-2 pr-4">{formatCacheSize(status)}</td>
                  <td className="truncate py-2 pr-4 font-mono text-xs" title={status?.cache_path}>{status?.cache_path ?? "n/a"}</td>
                  <td className="whitespace-nowrap py-2">
                    <div className="flex justify-start gap-2">
                      <button aria-label={activeDownloadJob?.status === "canceling" ? "Canceling model download" : active ? "Model download in progress" : "Download model"} className="ghost-button min-h-9 px-2 py-1" onClick={() => download(model.id)} disabled={!runtimeConnected || Boolean(active) || cached || local} title={!runtimeConnected ? "API is offline" : active ? "Open Jobs to monitor or cancel this download" : local ? "Local models do not need downloading" : cached ? "Model is already cached" : "Download model"}>
                        {active ? <Loader2 className="animate-spin" size={15} /> : <Download size={15} />}
                      </button>
                      <button className="ghost-button min-h-9 px-2 py-1" onClick={() => loadRuntime(model.id)} disabled={!runtimeConnected || !cached || loading || loaded || Boolean(unloadingModelId)} title={!runtimeConnected ? "API is offline" : unloadingModelId ? "Wait for the current model unload to finish" : loaded ? "Model is already loaded" : "Load model weights into backend memory"}>
                        {loading ? <Loader2 className="animate-spin" size={15} /> : <Upload size={15} />}
                      </button>
                      <button className="ghost-button min-h-9 px-2 py-1" onClick={() => unloadRuntime(model.id)} disabled={!runtimeConnected || !loaded || loading || Boolean(unloadingModelId)} title={!runtimeConnected ? "API is offline" : unloading ? "Waiting for the model worker to exit" : unloadingModelId ? "Another model unload is in progress" : "Unload model from backend memory"}>
                        {unloading ? <Loader2 className="animate-spin" size={15} /> : <XCircle size={15} />}
                        {unloading ? "Unloading" : "Unload"}
                      </button>
                      <button className="ghost-button min-h-9 px-2 py-1" onClick={() => local ? removeLocalModel(model.id) : deleteCache(model.id)} disabled={!runtimeConnected || active || loaded || loading || Boolean(unloadingModelId) || (!local && !(status?.cached || status?.cache_state === "partial"))} title={!runtimeConnected ? "API is offline" : unloadingModelId ? "Wait for the current model unload to finish" : local ? managedLocal ? "Delete imported local model and its cached files" : "Remove local model registration" : loaded || loading ? "Unload the model before deleting its cache" : "Delete cached model files"}>
                        <Trash2 size={15} />
                      </button>
                    </div>
                    {downloadError ? <div className="mt-2 max-w-80 text-xs text-rust" role="alert">{downloadError}</div> : null}
                  </td>
                    </tr>
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-5 grid gap-3 rounded-md border border-line bg-panel p-4">
            <div className="flex items-center gap-2">
              <PlusCircle size={16} />
              <h3 className="text-sm font-semibold">Register Local Model</h3>
            </div>
            <p className="max-w-3xl text-sm leading-6 text-ink/70">
              Register an existing local path below, or choose a model folder. Choosing a folder reads its root <code>config.json</code> locally and fills the source path; the folder files are not sent to the backend until you click Register.
            </p>
            <p className="text-xs text-ink/60">
              Compatible folder imports are stored after registration in the backend data directory under <code>data/models/imported_models</code>.
            </p>
            <input
              ref={localImportInputRef}
              className="sr-only"
              type="file"
              multiple
              onChange={(event) => {
                void selectLocalModelFolder(event.target.files);
                event.target.value = "";
              }}
            />
            <div className="flex flex-wrap gap-2">
              <button className="ghost-button w-fit" onClick={openLocalFolderPicker} disabled={!runtimeConnected || importing || checkingFolder} title={runtimeConnected ? "Choose a local model folder; files stay local until Register" : "API is offline"} type="button">
                {checkingFolder ? <Loader2 className="animate-spin" size={16} /> : <FolderUp size={16} />}
                Choose Model Folder
              </button>
            </div>
            {selectedFolderFiles.length > 0 && !selectedFolderError ? <div className="rounded-md border border-cyan/40 bg-cyan/10 p-3 text-sm text-cyan">Folder selected. Clicking Register starts an upload job; its live progress, validation, and registration appear in Jobs.</div> : null}
            {selectedFolderFiles.length > 0 ? <p className="text-xs text-ink/60">For privacy, browsers expose the selected folder’s relative source name rather than its absolute filesystem path.</p> : null}
            <div className="grid gap-3 md:grid-cols-2">
              <TextInput label="Display Name" value={localName} onChange={setLocalName} placeholder="GPT-2 XL" />
              <TextInput label="Local Path" value={localPath} onChange={(value) => {
                setLocalPath(value);
                setSelectedFolderFiles([]);
                setSelectedFolderError("");
              }} placeholder="/home/nick/models/my-model" />
              <label className="grid gap-2">
                <span className="label">Model Type</span>
                <select className="control" value={localModelType} onChange={(event) => setLocalModelType(event.target.value)}>
                  <option value="gpt2">GPT-2 architecture</option>
                  <option value="auto">AutoModel causal LM</option>
                  <option value="pythia">Pythia / GPT-NeoX</option>
                  <option value="llama">Llama-style causal LM</option>
                  <option value="gptj">GPT-J architecture</option>
                  <option value="baichuan">Baichuan causal LM</option>
                  <option value="chatglm">ChatGLM causal LM</option>
                  <option value="internlm">InternLM causal LM</option>
                  <option value="mistral">Mistral-style causal LM</option>
                  <option value="qwen">Qwen causal LM</option>
                </select>
              </label>
              <TextInput label="Layers" value={localLayers} onChange={setLocalLayers} placeholder="48" />
            </div>
            <button className="tool-button w-fit" onClick={addLocalModel} disabled={!runtimeConnected || importing || checkingFolder || Boolean(selectedFolderError) || (selectedFolderFiles.length === 0 && (!localName.trim() || !localPath.trim()))} title={runtimeConnected ? selectedFolderFiles.length ? "Upload and register the selected model folder" : "Register local model" : "API is offline"} type="button">
              <PlusCircle size={16} />
              Register
            </button>
          </div>
          {!runtimeConnected || error ? <div className="mt-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{runtimeConnected ? error : "Failed to fetch"}</div> : null}
          {selectedLocalModel ? <LocalModelDetailsDialog model={selectedLocalModel} onClose={() => setSelectedLocalModel(null)} /> : null}
        </>
      )}
    </section>
  );
}

function TextInput({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (value: string) => void; placeholder?: string }) {
  return (
    <label className="grid gap-2">
      <span className="label">{label}</span>
      <input className="control" value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function LocalModelDetailsDialog({ model, onClose }: { model: SupportedModel; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4" role="dialog" aria-modal="true" aria-label="Local model details">
      <div className="max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-lg border border-line bg-white p-5 shadow-xl">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold">{model.name}</h3>
            <p className="mt-1 text-sm text-ink/60">Upload details are kept outside the cache table.</p>
          </div>
          <button className="ghost-button min-h-9 px-2 py-1" onClick={onClose} title="Close local model details" type="button">
            <XCircle size={16} />
            Close
          </button>
        </div>
        <dl className="mt-5 grid gap-x-4 gap-y-3 rounded-md border border-line bg-panel p-4 text-sm sm:grid-cols-[9rem_minmax(0,1fr)]">
          <dt className="text-ink/60">Model ID</dt>
          <dd className="break-all font-mono text-xs">{model.id}</dd>
          <dt className="text-ink/60">Upload ID</dt>
          <dd className="break-all font-mono text-xs">{model.import_metadata?.import_id ?? "Not recorded"}</dd>
          <dt className="text-ink/60">Uploaded</dt>
          <dd>{formatLocalModelDate(model.import_metadata?.uploaded_at)}</dd>
          <dt className="text-ink/60">Source folder</dt>
          <dd className="break-all font-mono text-xs">{model.import_metadata?.source_folder ?? "Not recorded"}</dd>
          <dt className="text-ink/60">Stored at</dt>
          <dd className="break-all font-mono text-xs">{model.local_path ?? "Not recorded"}</dd>
        </dl>
      </div>
    </div>
  );
}

function selectedFolderSourcePath(files: File[]) {
  const relativePath = files[0]?.webkitRelativePath;
  if (relativePath) return relativePath.split("/")[0] || files[0].name;
  return files[0]?.name ?? "Selected model folder";
}

function selectedFolderConfigFile(files: File[]) {
  return files.find((file) => {
    const path = file.webkitRelativePath || file.name;
    const parts = path.split("/").filter(Boolean);
    return parts.length === 2 && parts[1] === "config.json";
  });
}

function displayNameFromFolderConfig(
  config: Record<string, unknown>,
  modelType: string,
  layerCount: number,
  fallback: string
) {
  const sourceName = typeof config._name_or_path === "string" ? config._name_or_path.toLowerCase() : "";
  if (sourceName.includes("gpt2-xl") || (modelType === "gpt2" && layerCount === 48)) return "GPT-2 XL";
  if (sourceName.includes("gpt-j") || modelType === "gptj") return "GPT-J 6B";
  if (modelType === "llama") return "Llama";
  if (modelType === "mistral") return "Mistral";
  if (modelType === "qwen") return "Qwen";
  if (modelType === "baichuan") return "Baichuan";
  if (modelType === "chatglm") return "ChatGLM";
  if (modelType === "internlm") return "InternLM";
  if (modelType === "gpt2") return "GPT-2";
  return fallback;
}

function configuredLayerCount(config: Record<string, unknown>) {
  for (const key of ["n_layer", "num_hidden_layers", "num_layers", "n_layers"]) {
    const value = config[key];
    if (typeof value === "number" && Number.isInteger(value) && value > 0) return value;
  }
  return null;
}

function formatLocalModelDate(value?: string) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function configureLocalFolderInput(input: HTMLInputElement | null) {
  if (!input) return;
  // Chromium uses webkitdirectory; setting the property immediately before
  // opening the picker avoids falling back to a regular file chooser.
  const directoryInput = input as HTMLInputElement & {
    webkitdirectory?: boolean;
    directory?: boolean;
  };
  input.setAttribute("webkitdirectory", "");
  input.setAttribute("directory", "");
  directoryInput.webkitdirectory = true;
  directoryInput.directory = true;
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatCacheState(status?: ModelCacheStatus) {
  if (!status) return "Unknown";
  if (status.cache_state === "not_cached") return "Not cached";
  if (status.cache_state === "partial") return "Partial";
  return status.cache_state.charAt(0).toUpperCase() + status.cache_state.slice(1);
}

function formatCacheSize(status?: ModelCacheStatus) {
  if (!status) return "Unknown";
  if (!status.cached && status.size_bytes === 0 && status.hub_size_bytes) {
    return formatBytes(status.hub_size_bytes);
  }
  return formatBytes(status.size_bytes);
}
