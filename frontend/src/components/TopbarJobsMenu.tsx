"use client";

import { Activity, CheckCircle2, CircleAlert, Clock3, Loader2, Trash2, X, XCircle } from "lucide-react";
import { createPortal } from "react-dom";
import { useEffect, useMemo, useRef, useState } from "react";

import { ExperimentHistory } from "@/components/ExperimentHistory";
import { api } from "@/lib/api";
import { applyDownloadJobToCache, listenForModelDownloads, upsertModelDownloadJob } from "@/lib/modelDownloadEvents";
import type { Experiment, ModelCacheStatus, ModelDownloadJob, ModelRuntimeStatus } from "@/lib/types";

const activeDownloadStates = ["queued", "downloading", "canceling"];
const activeRuntimeStates = ["queued", "loading"];

export function TopbarJobsMenu() {
  const menuRef = useRef<HTMLDivElement | null>(null);
  const newDownloadTimer = useRef<number | null>(null);
  const [open, setOpen] = useState(false);
  const [cache, setCache] = useState<ModelCacheStatus[]>([]);
  const [downloadJobs, setDownloadJobs] = useState<ModelDownloadJob[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeStatus[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [busyJob, setBusyJob] = useState("");
  const [error, setError] = useState("");
  const [hasNewDownload, setHasNewDownload] = useState(false);
  const [downloadDetails, setDownloadDetails] = useState<ModelDownloadJob | null>(null);
  const [confirmingClearDownloads, setConfirmingClearDownloads] = useState(false);
  const [clearingDownloads, setClearingDownloads] = useState(false);

  const activeDownloads = useMemo(
    () => downloadJobs.filter((job) => activeDownloadStates.includes(job.status)),
    [downloadJobs]
  );
  const downloadHistory = useMemo(
    () => downloadJobs.filter((job) => !activeDownloadStates.includes(job.status)),
    [downloadJobs]
  );
  const activeRuntime = useMemo(
    () => runtime.filter((item) => item.active_job || activeRuntimeStates.includes(item.runtime_state)),
    [runtime]
  );
  const hasFinishedDownloads = downloadHistory.length > 0;
  const runningExperiments = useMemo(
    () => experiments.filter((item) => item.status === "running"),
    [experiments]
  );
  const activeCount = activeDownloads.length + activeRuntime.length + runningExperiments.length;
  const hasActiveWork = activeCount > 0;

  async function refresh() {
    setError("");
    try {
      const [cacheResult, downloadJobResult, runtimeResult, experimentResult] = await Promise.all([
        api.modelCache(),
        api.modelDownloadJobs(),
        api.modelRuntime(),
        api.experiments()
      ]);
      setCache(cacheResult);
      setDownloadJobs(downloadJobResult);
      setRuntime(runtimeResult);
      setExperiments(experimentResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function cancelDownload(jobId: string) {
    setError("");
    setBusyJob(jobId);
    try {
      await api.cancelModelDownloadJob(jobId);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyJob("");
    }
  }

  async function clearDownloadHistory() {
    setClearingDownloads(true);
    setError("");
    try {
      await api.clearModelDownloadJobs();
      setConfirmingClearDownloads(false);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setClearingDownloads(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    const stopListening = listenForModelDownloads((job) => {
      setCache((current) => applyDownloadJobToCache(current, job));
      setDownloadJobs((current) => upsertModelDownloadJob(current, job));
      setHasNewDownload(true);
      if (newDownloadTimer.current !== null) {
        window.clearTimeout(newDownloadTimer.current);
      }
      newDownloadTimer.current = window.setTimeout(() => {
        setHasNewDownload(false);
        newDownloadTimer.current = null;
      }, 4000);
      void refresh();
    });
    return () => {
      stopListening();
      if (newDownloadTimer.current !== null) {
        window.clearTimeout(newDownloadTimer.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    refresh();
    if (!hasActiveWork) return;
    const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, [open, hasActiveWork]);

  useEffect(() => {
    if (!open) return;
    function handlePointerDown(event: MouseEvent) {
      const target = event.target;
      const withinDialog = target instanceof Element && Boolean(target.closest("[data-experiment-details-dialog], [data-download-details-dialog]"));
      if (menuRef.current && !menuRef.current.contains(target as Node) && !withinDialog) {
        setOpen(false);
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !document.querySelector("[data-experiment-details-dialog], [data-download-details-dialog]")) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  return (
    <div className="relative" ref={menuRef}>
      <button
        className="relative rounded-md px-3 py-2 text-sm text-ink/70 transition hover:bg-white hover:text-ink"
        onClick={() => {
          if (!open) setHasNewDownload(false);
          setOpen((value) => !value);
        }}
        type="button"
      >
        Jobs
        {activeCount > 0 ? (
          <span className={`ml-2 inline-flex min-w-5 items-center justify-center rounded-full bg-cyan px-1.5 py-0.5 text-xs font-semibold text-white ${hasNewDownload ? "animate-pulse ring-2 ring-cyan/30" : ""}`}>
            {activeCount}
          </span>
        ) : null}
      </button>
      {open ? (
        <div className="absolute right-0 top-full z-30 mt-2 w-[min(92vw,520px)] rounded-md border border-line bg-white p-3 shadow-lg">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Activity size={16} />
              <h2 className="text-sm font-semibold">Jobs</h2>
            </div>
            <button className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={refresh} type="button">
              Refresh
            </button>
          </div>

          <div className="grid max-h-[70vh] gap-3 overflow-y-auto pr-1">
            <section className="grid gap-2">
              <div className="flex items-center justify-between gap-3">
                <h3 className="label">Downloads</h3>
                <button className="ghost-button min-h-8 px-2 py-1 text-xs" disabled={!hasFinishedDownloads || clearingDownloads} onClick={() => setConfirmingClearDownloads(true)} title="Clear completed, failed, and canceled download history" type="button">
                  <Trash2 size={14} />
                  Clear history
                </button>
              </div>
              {confirmingClearDownloads ? (
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-rust/40 bg-rust/10 p-2 text-xs">
                  <span>Delete finished download records?</span>
                  <div className="flex gap-2">
                    <button className="ghost-button min-h-8 px-2 py-1 text-xs" disabled={clearingDownloads} onClick={() => setConfirmingClearDownloads(false)} type="button">Cancel</button>
                    <button className="ghost-button min-h-8 px-2 py-1 text-xs text-rust" disabled={clearingDownloads} onClick={clearDownloadHistory} type="button">
                      {clearingDownloads ? <Loader2 className="animate-spin" size={14} /> : <Trash2 size={14} />}
                      {clearingDownloads ? "Clearing" : "Clear"}
                    </button>
                  </div>
                </div>
              ) : null}
              {activeDownloads.map((job) => {
                const status = cache.find((item) => item.model_id === job.model_id) ?? job.cache;
                const canceling = job.status === "canceling";
                return (
                  <div key={job.job_id} className="rounded-md border border-line bg-panel p-3">
                    <div className="flex items-start justify-between gap-3">
                      <button className="min-w-0 flex-1 text-left" onClick={() => setDownloadDetails(job)} type="button">
                        <div className="flex min-w-0 items-center gap-2">
                          <DownloadStatusIcon status={job.status} />
                          <span className="truncate text-sm font-medium">{job.model_id}</span>
                        </div>
                        <div className="mt-1 text-xs text-ink/60">
                          {formatState(job.status)} · {formatDownloadSize(status)}
                        </div>
                      </button>
                      <button className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={() => cancelDownload(job.job_id)} disabled={canceling || busyJob === job.job_id} title="Cancel download and delete partial cache" type="button">
                        {canceling || busyJob === job.job_id ? <Loader2 className="animate-spin" size={14} /> : <XCircle size={14} />}
                        {canceling || busyJob === job.job_id ? "Canceling" : "Cancel"}
                      </button>
                    </div>
                    <Progress downloadedBytes={status?.size_bytes ?? 0} totalBytes={status?.hub_size_bytes} active complete={false} />
                  </div>
                );
              })}
              {downloadHistory.length > 0 ? (
                <div className="grid max-h-64 gap-2 overflow-y-auto pr-1">
                  {downloadHistory.map((job) => {
                    const status = cache.find((item) => item.model_id === job.model_id) ?? job.cache;
                    return (
                      <button key={job.job_id} className="block w-full rounded-md border border-line bg-white px-3 py-2 text-left text-sm transition hover:border-cyan/50 hover:bg-panel" onClick={() => setDownloadDetails(job)} type="button">
                        <div className="flex min-w-0 items-center gap-2">
                          <DownloadStatusIcon status={job.status} />
                          <span className="min-w-0 flex-1 truncate font-medium">{job.model_id}</span>
                          <span className="shrink-0 font-mono text-xs text-ink/60">{formatState(job.status)}</span>
                        </div>
                        <div className="mt-1 pl-6 font-mono text-xs text-ink/60">{formatDownloadSize(status)}</div>
                      </button>
                    );
                  })}
                </div>
              ) : null}
              {downloadJobs.length === 0 ? <div className="text-sm text-ink/60">No model download history.</div> : null}
            </section>

            <section className="grid gap-2">
              <h3 className="label">Runtime Loads</h3>
              {activeRuntime.map((status) => (
                <div key={status.model_id} className="rounded-md border border-line bg-panel p-3 text-sm">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium">{status.model_id}</span>
                    <span className="font-mono text-xs text-ink/60">{formatState(status.active_job?.status ?? status.runtime_state)}</span>
                  </div>
                  <div className="mt-1 text-xs text-ink/60">{status.device ?? "not_loaded"}</div>
                </div>
              ))}
              {activeRuntime.length === 0 ? <div className="text-sm text-ink/60">No active model loads.</div> : null}
            </section>

            <ExperimentHistory compact experiments={experiments} onChanged={refresh} />
          </div>

          {error ? <div className="mt-3 rounded-md border border-rust/40 bg-rust/10 p-2 text-xs text-rust">{error}</div> : null}
        </div>
      ) : null}
      {downloadDetails ? <DownloadDetailsDialog job={downloadDetails} status={cache.find((item) => item.model_id === downloadDetails.model_id) ?? downloadDetails.cache} onClose={() => setDownloadDetails(null)} /> : null}
    </div>
  );
}

function Progress({ downloadedBytes, totalBytes, active, complete }: { downloadedBytes: number; totalBytes?: number | null; active: boolean; complete: boolean }) {
  const total = totalBytes ?? 0;
  const hasTotal = total > 0;
  const percentage = hasTotal ? Math.min(100, Math.round((downloadedBytes / total) * 100)) : 0;
  return (
    <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-line" role="progressbar" aria-label="Model download progress" aria-valuemin={0} aria-valuemax={hasTotal ? 100 : undefined} aria-valuenow={hasTotal ? percentage : undefined}>
      <div className={`h-full bg-cyan transition-[width] ${!hasTotal && active ? "w-1/3 animate-pulse" : ""}`} style={hasTotal ? { width: `${complete ? 100 : percentage}%` } : undefined} />
    </div>
  );
}

function formatDownloadSize(status?: ModelCacheStatus | null) {
  if (!status) return "n/a";
  if (status.hub_size_bytes && status.hub_size_bytes > 0) {
    const percentage = Math.min(100, Math.round((status.size_bytes / status.hub_size_bytes) * 100));
    return `${formatBytes(status.size_bytes)} / ${formatBytes(status.hub_size_bytes)} (${percentage}%)`;
  }
  return formatBytes(status.size_bytes);
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatState(state: string) {
  return state.replace("_", " ").replace(/^\w/, (char) => char.toUpperCase());
}

function DownloadDetailsDialog({ job, status, onClose }: { job: ModelDownloadJob; status?: ModelCacheStatus | null; onClose: () => void }) {
  if (typeof document === "undefined") return null;
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/30 p-4 sm:items-center" data-download-details-dialog onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }} role="presentation">
      <section aria-modal="true" className="my-auto w-full max-w-xl rounded-md border border-line bg-white p-5 shadow-xl" role="dialog">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-base font-semibold">
              <DownloadStatusIcon status={job.status} />
              <span className="truncate">{job.model_id}</span>
            </div>
            <div className="mt-1 font-mono text-xs text-ink/60">{job.job_id}</div>
          </div>
          <button aria-label="Close download details" className="ghost-button grid size-8 place-items-center p-0" onClick={onClose} title="Close" type="button"><X size={16} /></button>
        </div>
        <dl className="mt-5 grid gap-3 text-sm sm:grid-cols-2">
          <DownloadDetail label="Status" value={formatState(job.status)} />
          <DownloadDetail label="Downloaded" value={formatDownloadSize(status)} />
          <DownloadDetail label="Repository" value={job.hf_id} />
          <DownloadDetail label="Cache state" value={status ? formatState(status.cache_state) : "Unknown"} />
          <DownloadDetail label="Started" value={formatTimestamp(job.created_at)} />
          <DownloadDetail label="Updated" value={formatTimestamp(job.updated_at)} />
        </dl>
        <section className="mt-5">
          <h3 className="label">Message</h3>
          <p className="mt-2 break-words rounded-md border border-line bg-panel p-3 text-sm text-ink/80">{job.message}</p>
        </section>
      </section>
    </div>,
    document.body
  );
}

function DownloadStatusIcon({ status }: { status: ModelDownloadJob["status"] }) {
  if (status === "completed") return <CheckCircle2 className="text-cyan" size={15} />;
  if (status === "failed" || status === "canceled") return <CircleAlert className="text-rust" size={15} />;
  return <Clock3 className="text-ink/60" size={15} />;
}

function DownloadDetail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="mt-1 break-words text-sm text-ink/80">{value}</dd>
    </div>
  );
}

function formatTimestamp(value: string) {
  return value.replace("T", " ").replace(/\.\d+\+00:00$/, " UTC").replace("+00:00", " UTC");
}
