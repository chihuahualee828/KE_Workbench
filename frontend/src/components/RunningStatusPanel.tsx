"use client";

import { Activity, CheckCircle2, CircleAlert, Clock3, Download, Loader2, RefreshCw, Server, Trash2, X, XCircle } from "lucide-react";
import { createPortal } from "react-dom";
import { useEffect, useMemo, useRef, useState } from "react";

import { ExperimentHistory } from "@/components/ExperimentHistory";
import { api } from "@/lib/api";
import { downloadJobCache, formatTransferSize, transferProgress } from "@/lib/downloadJobProgress";
import { listenForExperimentCanceled } from "@/lib/experimentEvents";
import { listenForModelDownloads, upsertModelDownloadJob } from "@/lib/modelDownloadEvents";
import type { Experiment, ModelCacheStatus, ModelDownloadJob, ModelRuntimeStatus } from "@/lib/types";

const activeDownloadStates = ["queued", "downloading", "uploading", "validating", "registering", "canceling"];
const activeRuntimeStates = ["queued", "loading"];

export function RunningStatusPanel() {
  const [cache, setCache] = useState<ModelCacheStatus[]>([]);
  const [downloadJobs, setDownloadJobs] = useState<ModelDownloadJob[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeStatus[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [busyJob, setBusyJob] = useState("");
  const [error, setError] = useState("");
  const [details, setDetails] = useState<ModelDownloadJob | null>(null);
  const [confirmingClearDownloads, setConfirmingClearDownloads] = useState(false);
  const [clearingDownloads, setClearingDownloads] = useState(false);
  const refreshing = useRef(false);

  const activeDownloads = useMemo(
    () => downloadJobs.filter((job) => activeDownloadStates.includes(job.status)),
    [downloadJobs]
  );
  const requiresTransferPolling = activeDownloads.some(
    (job) => job.job_type !== "local_import" || job.status !== "uploading"
  );
  const activeRuntime = useMemo(
    () => runtime.filter((item) => item.active_job || activeRuntimeStates.includes(item.runtime_state)),
    [runtime]
  );
  const runningExperiments = useMemo(
    () => experiments.filter((item) => item.status === "running"),
    [experiments]
  );
  const hasActiveWork = requiresTransferPolling || activeRuntime.length > 0 || runningExperiments.length > 0;
  const hasFinishedDownloads = downloadJobs.some((job) => !activeDownloadStates.includes(job.status));

  async function refresh() {
    if (refreshing.current) return;
    refreshing.current = true;
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
    } finally {
      refreshing.current = false;
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

  useEffect(() => listenForExperimentCanceled(({ experimentId, jobId }) => {
    setExperiments((current) => current.map((experiment) => {
      if (experiment.id !== experimentId && experiment.job?.id !== jobId) return experiment;
      return {
        ...experiment,
        status: "canceled",
        error_message: "",
        job: experiment.job ? { ...experiment.job, status: "canceled", progress: 1 } : experiment.job
      };
    }));
  }), []);

  useEffect(() => {
    return listenForModelDownloads((job) => {
      setDownloadJobs((current) => upsertModelDownloadJob(current, job));
      if (job.job_type !== "local_import" || job.status !== "uploading") {
        void refresh();
      }
    });
  }, []);

  useEffect(() => {
    if (!hasActiveWork) return;
    const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, [hasActiveWork]);

  return (
    <section className="panel">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Activity size={17} />
          <h2 className="text-base font-semibold">Jobs</h2>
        </div>
        <button className="ghost-button min-h-9 px-2 py-1" onClick={refresh} title="Refresh jobs">
          <RefreshCw size={15} />
          Refresh
        </button>
      </div>

      <div className="grid gap-4">
        <section className="grid gap-2">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Download size={15} />
              <h3 className="text-sm font-semibold">Model Transfers</h3>
            </div>
            <button className="ghost-button min-h-8 px-2 py-1 text-xs" disabled={!hasFinishedDownloads || clearingDownloads} onClick={() => setConfirmingClearDownloads(true)} title="Clear completed, failed, and canceled transfer history" type="button">
              <Trash2 size={14} />
              Clear history
            </button>
          </div>
          {confirmingClearDownloads ? (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm">
              <span>Delete completed, failed, and canceled transfer records?</span>
              <div className="flex gap-2">
                <button className="ghost-button min-h-8 px-2 py-1 text-xs" disabled={clearingDownloads} onClick={() => setConfirmingClearDownloads(false)} type="button">Cancel</button>
                <button className="ghost-button min-h-8 px-2 py-1 text-xs text-rust" disabled={clearingDownloads} onClick={clearDownloadHistory} type="button">
                  {clearingDownloads ? <Loader2 className="animate-spin" size={14} /> : <Trash2 size={14} />}
                  {clearingDownloads ? "Clearing" : "Clear"}
                </button>
              </div>
            </div>
          ) : null}
          <div className="max-h-[540px] divide-y divide-line overflow-y-auto rounded-md border border-line">
            {downloadJobs.map((job) => {
              const status = downloadJobCache(job, cache);
              const canceling = job.status === "canceling";
              const active = activeDownloadStates.includes(job.status);
              return (
                <div key={job.job_id} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 py-3">
                  <button className="min-w-0 text-left" onClick={() => setDetails(job)} type="button">
                    <div className="flex items-center gap-2">
                      <DownloadStatusIcon status={job.status} />
                      <span className="truncate text-sm font-medium">{jobTitle(job)}</span>
                    </div>
                    <div className="mt-1 truncate font-mono text-xs text-ink/60">{formatTransferSize(job, status)}</div>
                    {active ? <Progress {...transferProgress(job, status)} active complete={false} /> : null}
                  </button>
                  <div className="flex items-center gap-2">
                    <DownloadStatusBadge status={job.status} />
                    {active ? (
                      <button className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={() => cancelDownload(job.job_id)} disabled={canceling || busyJob === job.job_id} title={job.job_type === "local_import" ? "Cancel local model import" : "Cancel download and delete partial cache"} type="button">
                        {canceling || busyJob === job.job_id ? <Loader2 className="animate-spin" size={14} /> : <XCircle size={14} />}
                        {canceling || busyJob === job.job_id ? "Canceling" : "Cancel"}
                      </button>
                    ) : null}
                  </div>
                </div>
              );
            })}
            {downloadJobs.length === 0 ? <div className="px-3 py-3 text-sm text-ink/60">No model transfer history.</div> : null}
          </div>
        </section>

        <section className="grid gap-2">
          <div className="flex items-center gap-2">
            <Server size={15} />
            <h3 className="text-sm font-semibold">Runtime Loads</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[620px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-line text-left">
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2 pr-4">Status</th>
                  <th className="py-2 pr-4">Device</th>
                  <th className="py-2 pr-4">Job</th>
                </tr>
              </thead>
              <tbody>
                {activeRuntime.map((status) => (
                  <tr key={status.model_id} className="border-b border-line/70">
                    <td className="py-2 pr-4">{status.model_id}</td>
                    <td className="py-2 pr-4">{formatState(status.active_job?.status ?? status.runtime_state)}</td>
                    <td className="py-2 pr-4">{status.device ?? "not_loaded"}</td>
                    <td className="py-2 pr-4 text-xs text-ink/70">{status.active_job ? `${status.active_job.job_id}: ${status.active_job.message}` : "n/a"}</td>
                  </tr>
                ))}
                {activeRuntime.length === 0 ? (
                  <tr>
                    <td className="py-3 text-sm text-ink/60" colSpan={4}>No active model load jobs.</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </section>

        <ExperimentHistory experiments={experiments} onChanged={refresh} />
      </div>
      {error ? <div className="mt-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      {details ? <DownloadDetailsDialog job={downloadJobs.find((job) => job.job_id === details.job_id) ?? details} status={downloadJobCache(downloadJobs.find((job) => job.job_id === details.job_id) ?? details, cache)} onClose={() => setDetails(null)} /> : null}
    </section>
  );
}

function Progress({ downloadedBytes, totalBytes, active, complete }: { downloadedBytes: number; totalBytes?: number | null; active: boolean; complete: boolean }) {
  const total = totalBytes ?? 0;
  const hasTotal = total > 0;
  const percentage = hasTotal ? Math.min(100, Math.round((downloadedBytes / total) * 100)) : 0;
  return (
    <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-line" role="progressbar" aria-label="Model download progress" aria-valuemin={0} aria-valuemax={hasTotal ? 100 : undefined} aria-valuenow={hasTotal ? percentage : undefined}>
      <div className={`h-full bg-cyan transition-[width] ${!hasTotal && active ? "w-1/3 animate-pulse" : ""}`} style={hasTotal ? { width: `${complete ? 100 : percentage}%` } : undefined} />
    </div>
  );
}

function jobTitle(job: ModelDownloadJob) {
  return job.source_name ?? job.model_id;
}

function formatState(state: string) {
  return state.replace("_", " ").replace(/^\w/, (char) => char.toUpperCase());
}

function DownloadStatusIcon({ status }: { status: ModelDownloadJob["status"] }) {
  if (status === "completed") return <CheckCircle2 className="text-cyan" size={15} />;
  if (status === "failed" || status === "canceled") return <CircleAlert className="text-rust" size={15} />;
  return <Clock3 className="text-ink/60" size={15} />;
}

function DownloadStatusBadge({ status }: { status: ModelDownloadJob["status"] }) {
  const className = status === "completed"
    ? "border-cyan/40 bg-cyan/10 text-cyan"
    : status === "failed" || status === "canceled"
      ? "border-rust/40 bg-rust/10 text-rust"
      : "border-line bg-panel text-ink/70";
  return <span className={`rounded-md border px-2 py-1 font-mono text-xs ${className}`}>{formatState(status)}</span>;
}

function DownloadDetailsDialog({ job, status, onClose }: { job: ModelDownloadJob; status?: ModelCacheStatus | null; onClose: () => void }) {
  if (typeof document === "undefined") return null;
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/30 p-4 sm:items-center" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }} role="presentation">
      <section aria-modal="true" className="my-auto w-full max-w-xl rounded-md border border-line bg-white p-5 shadow-xl" role="dialog">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-base font-semibold">
              <DownloadStatusIcon status={job.status} />
              <span className="truncate">{jobTitle(job)}</span>
            </div>
            <div className="mt-1 font-mono text-xs text-ink/60">{job.job_id}</div>
          </div>
          <button aria-label="Close download details" className="ghost-button grid size-8 place-items-center p-0" onClick={onClose} title="Close" type="button"><X size={16} /></button>
        </div>
        <dl className="mt-5 grid gap-3 text-sm sm:grid-cols-2">
          <DownloadDetail label="Status" value={formatState(job.status)} />
          <DownloadDetail label={job.job_type === "local_import" ? "Progress" : "Downloaded"} value={formatTransferSize(job, status)} />
          <DownloadDetail label={job.job_type === "local_import" ? "Transfer" : "Repository"} value={job.job_type === "local_import" ? "Local model import" : job.hf_id} />
          <DownloadDetail label="Recorded cache state" value={job.job_type === "local_import" ? "Stored in backend data after registration" : status ? formatState(status.cache_state) : "Unknown"} />
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
