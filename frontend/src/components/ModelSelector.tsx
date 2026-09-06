"use client";

import { Download, HardDrive, Loader2, Upload, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import { announceModelDownload, applyDownloadJobToCache, listenForModelDownloads } from "@/lib/modelDownloadEvents";
import { announceModelRuntimeLoad, announceModelRuntimeStatuses, announceModelRuntimeUnloading, applyRuntimeJobToRuntime, applyRuntimeStatusToRuntime, isModelRuntimeConnected, listenForModelRuntimeConnection, listenForModelRuntimeLoads, listenForModelRuntimeStatuses, listenForModelRuntimeUnloading, watchModelRuntimeJob } from "@/lib/modelRuntimeEvents";
import type { ModelCacheStatus, ModelDownloadJob, ModelRuntimeJob, ModelRuntimeStatus, SupportedModel } from "@/lib/types";

const activeDownloadStatuses = ["queued", "downloading", "canceling"];

export function ModelSelector({
  models,
  value,
  onChange,
  onRuntimeUnloaded,
  showCacheControls = true,
  refreshKey = 0
}: {
  models: SupportedModel[];
  value: string;
  onChange: (modelId: string) => void;
  onRuntimeUnloaded?: () => void | Promise<void>;
  showCacheControls?: boolean;
  refreshKey?: number;
}) {
  const [cache, setCache] = useState<ModelCacheStatus[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeStatus[]>([]);
  const [job, setJob] = useState<ModelDownloadJob | null>(null);
  const [runtimeJob, setRuntimeJob] = useState<ModelRuntimeJob | null>(null);
  const [unloadingModelId, setUnloadingModelId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [runtimeConnected, setRuntimeConnected] = useState(isModelRuntimeConnected);

  const selectedCache = useMemo(
    () => cache.find((item) => item.model_id === value),
    [cache, value]
  );
  const selectedModel = useMemo(
    () => models.find((model) => model.id === value),
    [models, value]
  );
  const selectedLocal = selectedModel?.source === "local" || selectedCache?.is_local;
  const selectedJobActive = job?.model_id === value && activeDownloadStatuses.includes(job.status);
  const selectedRuntime = useMemo(
    () => runtime.find((item) => item.model_id === value),
    [runtime, value]
  );
  const selectedRuntimeActive = runtimeJob?.model_id === value && ["queued", "loading"].includes(runtimeJob.status);
  const selectedRuntimeUnloading = unloadingModelId === value;

  useEffect(() => {
    if (models.length && !models.some((model) => model.id === value)) {
      onChange(models[0].id);
    }
  }, [models, onChange, value]);

  async function refreshCache() {
    if (!runtimeConnected) return;
    try {
      const result = await api.modelCache();
      const runtimeResult = await api.modelRuntime();
      setCache(result);
      setRuntime(runtimeResult);
      const activeSelectedJob = result.find((item) => item.model_id === value)?.active_job;
      if (activeSelectedJob) {
        setJob(activeSelectedJob);
      }
      const activeRuntimeJob = runtimeResult.find((item) => item.model_id === value)?.active_job;
      if (activeRuntimeJob) {
        setRuntimeJob(activeRuntimeJob);
        watchModelRuntimeJob(activeRuntimeJob).catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function startDownload() {
    if (!runtimeConnected) return;
    setError("");
    try {
      const result = await api.downloadModel({ model_id: value });
      setJob(result);
      if (activeDownloadStatuses.includes(result.status)) {
        setCache((current) => applyDownloadJobToCache(current, result));
        announceModelDownload(result);
      }
      void refreshCache();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function loadRuntime() {
    if (!runtimeConnected) return;
    setError("");
    try {
      const result = await api.loadModelRuntime({ model_id: value });
      setRuntimeJob(result);
      setRuntime((current) => applyRuntimeJobToRuntime(current, result));
      announceModelRuntimeLoad(result);
      watchModelRuntimeJob(result).catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function unloadRuntime() {
    if (!runtimeConnected || unloadingModelId) return;
    setError("");
    setUnloadingModelId(value);
    announceModelRuntimeUnloading(value, true);
    try {
      await api.unloadModelRuntime({ model_id: value });
      setRuntimeJob(null);
      const runtimeResult = await api.modelRuntime();
      setRuntime(runtimeResult);
      announceModelRuntimeStatuses(runtimeResult);
      await onRuntimeUnloaded?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUnloadingModelId(null);
      announceModelRuntimeUnloading(value, false);
    }
  }

  useEffect(() => {
    if (!showCacheControls) return;
    refreshCache();
  }, [showCacheControls, value, refreshKey]);

  useEffect(() => {
    return listenForModelDownloads((downloadJob) => {
      if (downloadJob.model_id === value) {
        setJob(downloadJob);
      }
      setCache((current) => applyDownloadJobToCache(current, downloadJob));
    });
  }, [value]);

  useEffect(() => {
    return listenForModelRuntimeLoads((loadJob) => {
      if (loadJob.model_id === value) {
        setRuntimeJob(loadJob);
      }
      setRuntime((current) => applyRuntimeJobToRuntime(current, loadJob));
      if (!["queued", "loading"].includes(loadJob.status)) {
        void refreshCache();
      }
    });
  }, [value]);

  useEffect(() => {
    return listenForModelRuntimeStatuses((status) => {
      if (status.model_id === value && !status.active_job) {
        setRuntimeJob(null);
      }
      setRuntime((current) => applyRuntimeStatusToRuntime(current, status));
    });
  }, [value]);

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
        setRuntimeJob((current) => current?.status === "failed" ? null : current);
        return;
      }
      setError("Failed to fetch");
      setRuntimeJob((current) => current && ["queued", "loading"].includes(current.status)
        ? { ...current, status: "failed" as const, message: "Failed to fetch", updated_at: new Date().toISOString() }
        : current);
      setRuntime((current) => current.map((status) =>
        status.model_id === value && ["queued", "loading"].includes(status.runtime_state)
          ? { ...status, loaded: false, runtime_state: "failed", active_job: null }
          : status
      ));
    });
  }, [value]);

  useEffect(() => {
    const activeCache = activeDownloadStatuses.includes(selectedCache?.cache_state ?? "not_cached");
    if ((!job || !activeDownloadStatuses.includes(job.status)) && !activeCache) return;
    const timer = window.setInterval(async () => {
      try {
        if (job && activeDownloadStatuses.includes(job.status)) {
          const result = await api.modelDownloadJob(job.job_id);
          setJob(result);
        }
        refreshCache();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        window.clearInterval(timer);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job, selectedCache?.cache_state]);

  return (
    <div className="grid gap-2">
      <label className="grid gap-2">
        <span className="label">Model</span>
        <select className="control" value={value} disabled={!models.length} onChange={(event) => onChange(event.target.value)}>
          {!models.length ? <option value={value}>No models available</option> : null}
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </label>
      {selectedModel ? <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-ink/50"><span className="min-w-0 break-all font-mono">{selectedModel.hf_id}</span><span aria-hidden="true">·</span><span>{selectedModel.n_layers} layers</span></div> : null}
      {showCacheControls ? (
        <div className="grid gap-2">
          <div className="flex min-h-10 items-center justify-between gap-2 rounded-md border border-line bg-panel px-3 py-2 text-sm">
            <span className="inline-flex items-center gap-2">
              <HardDrive size={15} />
              {selectedCache?.cached ? `Cached (${formatBytes(selectedCache.size_bytes)})` : `${formatCacheState(selectedCache)} (${formatExpectedSize(selectedCache)})`}
            </span>
            {!selectedCache?.cached && !selectedLocal ? (
              <button className="ghost-button min-h-8 px-2 py-1" onClick={startDownload} disabled={!runtimeConnected || !selectedModel || selectedJobActive} title={!runtimeConnected ? "API is offline" : selectedJobActive ? "Open Jobs to monitor or cancel this download" : "Download model into Hugging Face cache"}>
                {selectedJobActive ? <Loader2 className="animate-spin" size={15} /> : <Download size={15} />}
                {job?.status === "canceling" ? "Canceling" : selectedJobActive ? "Downloading" : "Download"}
              </button>
            ) : null}
          </div>
          <div className="flex min-h-10 items-center justify-between gap-2 rounded-md border border-line bg-panel px-3 py-2 text-sm">
            <span>
              {selectedRuntime?.loaded ? (
                <span>
                  <span>{`Loaded · ${selectedRuntime.device ?? "cpu"}`}</span>
                  <span className="mt-1 block font-mono text-[10px] text-ink/50">{formatBytes(selectedRuntime.model_vram_bytes ?? 0)} VRAM</span>
                </span>
              ) : selectedRuntimeUnloading ? "Unloading model" : selectedRuntimeActive ? "Loading weights" : runtimeJob?.model_id === value && runtimeJob.status === "failed" && runtimeJob.message !== "Failed to fetch" ? runtimeJob.message : "Not loaded in memory"}
            </span>
            {selectedRuntime?.loaded || selectedRuntimeUnloading ? (
              <button className="ghost-button min-h-8 px-2 py-1" onClick={unloadRuntime} disabled={!runtimeConnected || selectedRuntimeActive || Boolean(unloadingModelId)} title={!runtimeConnected ? "API is offline" : selectedRuntimeUnloading ? "Waiting for the model worker to exit" : "Unload model from backend memory"}>
                {selectedRuntimeUnloading ? <Loader2 className="animate-spin" size={15} /> : <XCircle size={15} />}
                {selectedRuntimeUnloading ? "Unloading" : "Unload"}
              </button>
            ) : (
              <button className="ghost-button min-h-8 px-2 py-1" onClick={loadRuntime} disabled={!runtimeConnected || !selectedCache?.cached || selectedRuntimeActive || Boolean(unloadingModelId)} title={!runtimeConnected ? "API is offline" : unloadingModelId ? "Wait for the current model unload to finish" : selectedCache?.cached ? "Load model weights into backend memory" : "Download model before loading"}>
                {selectedRuntimeActive ? <Loader2 className="animate-spin" size={15} /> : <Upload size={15} />}
                {selectedRuntimeActive ? "Loading" : "Load"}
              </button>
            )}
          </div>
          {job?.model_id === value ? (
            <div className="rounded-md border border-line bg-white px-3 py-2 text-xs text-ink/70">
              {job.status}: {job.message}
            </div>
          ) : null}
          {runtimeJob?.model_id === value ? (
            <div className="rounded-md border border-line bg-white px-3 py-2 text-xs text-ink/70">
              {runtimeJob.status}: {runtimeJob.message}
            </div>
          ) : null}
          {!runtimeConnected || error ? <div role="alert" className="inline-error text-xs">{!runtimeConnected || error === "Failed to fetch" ? "Backend unavailable. Check that the backend service is running." : error}</div> : null}
        </div>
      ) : null}
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

function formatExpectedSize(status?: ModelCacheStatus) {
  if (!status) return "Unknown";
  if (!status.cached && status.size_bytes === 0 && status.hub_size_bytes) {
    return formatBytes(status.hub_size_bytes);
  }
  return formatBytes(status.size_bytes);
}
