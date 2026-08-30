"use client";

import { Download, Loader2, PlusCircle, RefreshCw, Trash2, Upload, XCircle } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { announceModelDownload, applyDownloadJobToCache, listenForModelDownloads } from "@/lib/modelDownloadEvents";
import { announceModelRuntimeLoad, announceModelRuntimeStatuses, applyRuntimeJobToRuntime, applyRuntimeStatusToRuntime, isModelRuntimeConnected, listenForModelRuntimeConnection, listenForModelRuntimeLoads, listenForModelRuntimeStatuses, watchModelRuntimeJob } from "@/lib/modelRuntimeEvents";
import type { ModelCacheStatus, ModelDownloadJob, ModelRuntimeJob, ModelRuntimeStatus, SupportedModel } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";

const activeDownloadStatuses = ["queued", "downloading", "canceling"];

export function ModelCacheManager({ models, onModelsChanged }: { models: SupportedModel[]; onModelsChanged?: () => void | Promise<void> }) {
  const [cache, setCache] = useState<ModelCacheStatus[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeStatus[]>([]);
  const [jobs, setJobs] = useState<Record<string, ModelDownloadJob>>({});
  const [runtimeJobs, setRuntimeJobs] = useState<Record<string, ModelRuntimeJob>>({});
  const [error, setError] = useState("");
  const [runtimeConnected, setRuntimeConnected] = useState(isModelRuntimeConnected);
  const [localId, setLocalId] = usePersistentState("kew:modelManager:localId", "");
  const [localName, setLocalName] = usePersistentState("kew:modelManager:localName", "");
  const [localPath, setLocalPath] = usePersistentState("kew:modelManager:localPath", "");
  const [localModelType, setLocalModelType] = usePersistentState("kew:modelManager:localModelType", "gpt2");
  const [localLayers, setLocalLayers] = usePersistentState("kew:modelManager:localLayers", "0");
  const [localRome, setLocalRome] = usePersistentState("kew:modelManager:localRome", false);
  const [localMemit, setLocalMemit] = usePersistentState("kew:modelManager:localMemit", false);
  const [localTrace, setLocalTrace] = usePersistentState("kew:modelManager:localTrace", false);
  const [localLens, setLocalLens] = usePersistentState("kew:modelManager:localLens", false);
  const [romeHparams, setRomeHparams] = usePersistentState("kew:modelManager:romeHparams", "");
  const [memitHparams, setMemitHparams] = usePersistentState("kew:modelManager:memitHparams", "");

  useEffect(() => {
    if (localModelType === "gpt2") return;
    setLocalTrace(false);
  }, [localModelType, setLocalTrace]);

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
      await api.deleteModelCache({ model_id: modelId });
      await refresh();
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
      setRuntime((current) => applyRuntimeJobToRuntime(current, job));
      announceModelRuntimeLoad(job);
      watchModelRuntimeJob(job).catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function unloadRuntime(modelId: string) {
    if (!runtimeConnected) return;
    setError("");
    try {
      await api.unloadModelRuntime({ model_id: modelId });
      const runtimeResult = await api.modelRuntime();
      setRuntime(runtimeResult);
      announceModelRuntimeStatuses(runtimeResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function addLocalModel() {
    if (!runtimeConnected) return;
    setError("");
    try {
      await api.addLocalModel({
        id: localId,
        name: localName || localId,
        local_path: localPath,
        model_type: localModelType || "auto",
        n_layers: Number(localLayers) || 0,
        supports_generation: true,
        supports_logit_lens: localLens,
        supports_causal_trace: localTrace,
        supports_rome: localRome,
        supports_memit: localMemit,
        rome_hparams: romeHparams || null,
        memit_hparams: memitHparams || null
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
    refresh();
  }, []);

  useEffect(() => {
    return listenForModelDownloads((job) => {
      setJobs((current) => ({ ...current, [job.job_id]: job }));
      setCache((current) => applyDownloadJobToCache(current, job));
    });
  }, []);

  useEffect(() => {
    return listenForModelRuntimeLoads((job) => {
      setRuntimeJobs((current) => ({ ...current, [job.job_id]: job }));
      setRuntime((current) => applyRuntimeJobToRuntime(current, job));
      if (!["queued", "loading"].includes(job.status)) {
        void refresh();
      }
    });
  }, []);

  useEffect(() => {
    return listenForModelRuntimeStatuses((status) => {
      setRuntime((current) => applyRuntimeStatusToRuntime(current, status));
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

  useEffect(() => {
    const activeJobs = Object.values(jobs).filter((job) => activeDownloadStatuses.includes(job.status));
    const activeCache = cache.some((status) => activeDownloadStatuses.includes(status.cache_state));
    if (activeJobs.length === 0 && !activeCache) return;
    const timer = window.setInterval(async () => {
      if (activeJobs.length > 0) {
        const updates = await Promise.all(activeJobs.map((job) => api.modelDownloadJob(job.job_id)));
        setJobs((current) => {
          const next = { ...current };
          updates.forEach((job) => {
            next[job.job_id] = job;
          });
          return next;
        });
      }
      refresh();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobs, cache]);

  return (
    <section className="panel">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">Model Cache</h2>
        <button className="ghost-button min-h-9 px-2 py-1" onClick={refresh} disabled={!runtimeConnected} title={runtimeConnected ? "Refresh model cache" : "API is offline"}>
          <RefreshCw size={15} />
          Refresh
        </button>
      </div>
      <GpuRuntimeSummary runtime={runtime} />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4">Cache</th>
              <th className="py-2 pr-4">Runtime</th>
              <th className="py-2 pr-4">Size</th>
              <th className="py-2 pr-4">Path</th>
              <th className="py-2 pr-4">Job</th>
              <th className="py-2 pr-4">Actions</th>
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
              const runtimeError = runtimeJob?.status === "failed" && runtimeJob.message !== "Failed to fetch" ? runtimeJob.message : "";
              const cached = Boolean(status?.cached);
              const loaded = Boolean(runtimeStatus?.loaded);
              const local = model.source === "local" || status?.is_local;
              return (
                <tr key={model.id} className="border-b border-line/70 align-top">
                  <td className="py-2 pr-4">
                    <div className="font-medium">{model.name}</div>
                    <div className="font-mono text-xs text-ink/50">{model.hf_id}</div>
                    {local ? <div className="mt-1 text-xs text-cyan">Local registration</div> : null}
                  </td>
                  <td className="py-2 pr-4">{formatCacheState(status)}</td>
                  <td className="py-2 pr-4">
                    {loaded ? (
                      <div>
                        <div>{`Loaded (${runtimeStatus?.device ?? "cpu"})`}</div>
                        <div className="mt-1 text-xs text-ink/60">{formatBytes(runtimeStatus?.model_vram_bytes ?? 0)} model VRAM</div>
                      </div>
                    ) : loading ? "Loading" : runtimeError || "Not loaded"}
                  </td>
                  <td className="py-2 pr-4">{formatBytes(status?.size_bytes ?? 0)}</td>
                  <td className="max-w-80 truncate py-2 pr-4 font-mono text-xs" title={status?.cache_path}>{status?.cache_path ?? "n/a"}</td>
                  <td className="py-2 pr-4 text-xs text-ink/70">{job ? `${job.status}: ${job.message}` : status?.active_job ? `${status.active_job.status}: ${status.active_job.message}` : "n/a"}</td>
                  <td className="py-2 pr-4">
                    <div className="flex gap-2">
                      <button className="ghost-button min-h-9 px-2 py-1" onClick={() => download(model.id)} disabled={!runtimeConnected || Boolean(active) || cached || local} title={!runtimeConnected ? "API is offline" : active ? "Open Jobs to monitor or cancel this download" : local ? "Local models do not need downloading" : cached ? "Model is already cached" : "Download model"}>
                        {active ? <Loader2 className="animate-spin" size={15} /> : <Download size={15} />}
                        {activeDownloadJob?.status === "canceling" ? "Canceling" : active ? "Downloading" : "Download"}
                      </button>
                      <button className="ghost-button min-h-9 px-2 py-1" onClick={() => loadRuntime(model.id)} disabled={!runtimeConnected || !cached || loading || loaded} title={!runtimeConnected ? "API is offline" : loaded ? "Model is already loaded" : "Load model weights into backend memory"}>
                        {loading ? <Loader2 className="animate-spin" size={15} /> : <Upload size={15} />}
                      </button>
                      <button className="ghost-button min-h-9 px-2 py-1" onClick={() => unloadRuntime(model.id)} disabled={!runtimeConnected || !loaded || loading} title={!runtimeConnected ? "API is offline" : "Unload model from backend memory"}>
                        <XCircle size={15} />
                        Unload
                      </button>
                      <button className="ghost-button min-h-9 px-2 py-1" onClick={() => local ? removeLocalModel(model.id) : deleteCache(model.id)} disabled={!runtimeConnected || active || loaded || loading || (!local && !(status?.cached || status?.cache_state === "partial"))} title={!runtimeConnected ? "API is offline" : local ? "Remove local model registration" : loaded || loading ? "Unload the model before deleting its cache" : "Delete cached model files"}>
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </td>
                </tr>
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
        <div className="grid gap-3 md:grid-cols-2">
          <TextInput label="Model ID" value={localId} onChange={setLocalId} placeholder="local-gpt2-xl" />
          <TextInput label="Display Name" value={localName} onChange={setLocalName} placeholder="Local GPT-2 XL" />
          <TextInput label="Local Path" value={localPath} onChange={setLocalPath} placeholder="/home/nick/models/my-model" />
          <label className="grid gap-2">
            <span className="label">Model Type</span>
            <select className="control" value={localModelType} onChange={(event) => setLocalModelType(event.target.value)}>
              <option value="gpt2">GPT-2 architecture</option>
              <option value="auto">AutoModel causal LM</option>
              <option value="pythia">Pythia / GPT-NeoX</option>
              <option value="llama">Llama-style causal LM</option>
            </select>
          </label>
          <TextInput label="Layers" value={localLayers} onChange={setLocalLayers} placeholder="48" />
          <TextInput label="ROME HParams" value={romeHparams} onChange={setRomeHparams} placeholder="hparams/ROME/gpt2-xl.yaml" />
          <TextInput label="MEMIT HParams" value={memitHparams} onChange={setMemitHparams} placeholder="hparams/MEMIT/gpt2-xl.yaml" />
        </div>
        <div className="flex flex-wrap gap-4 text-sm">
          <Check label="Logit Lens" checked={localLens} onChange={setLocalLens} />
          <Check label="Causal Trace" checked={localTrace} onChange={setLocalTrace} disabled={localModelType !== "gpt2"} />
          <Check label="ROME" checked={localRome} onChange={setLocalRome} />
          <Check label="MEMIT" checked={localMemit} onChange={setLocalMemit} />
        </div>
        <button className="tool-button w-fit" onClick={addLocalModel} disabled={!runtimeConnected || !localId.trim() || !localPath.trim()} title={runtimeConnected ? "Register local model path" : "API is offline"}>
          <PlusCircle size={16} />
          Register
        </button>
      </div>
      {!runtimeConnected || error ? <div className="mt-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{runtimeConnected ? error : "Failed to fetch"}</div> : null}
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

function Check({ label, checked, onChange, disabled = false }: { label: string; checked: boolean; onChange: (value: boolean) => void; disabled?: boolean }) {
  return (
    <label className={`inline-flex items-center gap-2 ${disabled ? "text-ink/40" : ""}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

function GpuRuntimeSummary({ runtime }: { runtime: ModelRuntimeStatus[] }) {
  const loadedRuntime = runtime.find((status) => status.loaded);
  const gpu = loadedRuntime?.gpu ?? runtime.find((status) => status.gpu)?.gpu;
  if (!gpu?.cuda_available) {
    return (
      <div className="mb-4 rounded-md border border-line bg-panel p-3 text-sm text-ink/70">
        GPU: CUDA not available to the backend.
      </div>
    );
  }
  const totalBytes = gpu.total_bytes ?? 0;
  const usedBytes = gpu.used_bytes ?? (totalBytes && gpu.free_bytes !== undefined
    ? Math.max(0, totalBytes - gpu.free_bytes)
    : gpu.allocated_bytes ?? 0);
  const usedPercent = totalBytes ? Math.min(100, Math.max(0, (usedBytes / totalBytes) * 100)) : 0;
  return (
    <div className="mb-4 grid gap-3 rounded-md border border-line bg-panel p-3 text-sm md:grid-cols-[minmax(0,0.9fr)_minmax(0,1.4fr)_minmax(0,1fr)]">
      <div>
        <div className="label mb-1">GPU</div>
        <div className="font-medium">{gpu.device_name ?? `CUDA ${gpu.device ?? "0"}`}</div>
      </div>
      <div>
        <div className="mb-1 flex items-center justify-between gap-3">
          <div className="label">GPU VRAM</div>
          <div className="font-mono text-xs text-ink/70">
            {formatBytes(usedBytes)} / {formatBytes(totalBytes)}
          </div>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-line">
          <div className="h-full rounded-full bg-cyan transition-[width]" style={{ width: `${usedPercent}%` }} />
        </div>
        <div className="mt-1 font-mono text-xs text-ink/50">{usedPercent.toFixed(1)}% used · {gpu.memory_source ?? "torch"}</div>
      </div>
      <div>
        <div className="label mb-1">Loaded Model</div>
        <div className="font-mono text-xs text-ink/70">
          {loadedRuntime ? `${formatBytes(loadedRuntime.model_vram_bytes ?? 0)} (${loadedRuntime.model_id})` : "No model loaded"}
        </div>
        <div className="mt-1 font-mono text-xs text-ink/50">{formatBytes(gpu.allocated_bytes ?? 0)} allocated / {formatBytes(gpu.reserved_bytes ?? 0)} reserved</div>
      </div>
    </div>
  );
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
