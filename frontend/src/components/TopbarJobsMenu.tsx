"use client";

import { Activity, Loader2, XCircle } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { ExperimentHistory } from "@/components/ExperimentHistory";
import { api } from "@/lib/api";
import type { Experiment, ModelCacheStatus, ModelRuntimeStatus } from "@/lib/types";

const activeDownloadStates = ["queued", "downloading", "canceling"];
const activeRuntimeStates = ["queued", "loading"];

export function TopbarJobsMenu() {
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [open, setOpen] = useState(false);
  const [cache, setCache] = useState<ModelCacheStatus[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeStatus[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [busyJob, setBusyJob] = useState("");
  const [error, setError] = useState("");

  const activeDownloads = useMemo(
    () => cache.filter((item) => item.active_job || activeDownloadStates.includes(item.cache_state)),
    [cache]
  );
  const activeRuntime = useMemo(
    () => runtime.filter((item) => item.active_job || activeRuntimeStates.includes(item.runtime_state)),
    [runtime]
  );
  const runningExperiments = useMemo(
    () => experiments.filter((item) => item.status === "running"),
    [experiments]
  );
  const activeCount = activeDownloads.length + activeRuntime.length + runningExperiments.length;
  const hasActiveWork = activeCount > 0;

  async function refresh() {
    setError("");
    try {
      const [cacheResult, runtimeResult, experimentResult] = await Promise.all([
        api.modelCache(),
        api.modelRuntime(),
        api.experiments()
      ]);
      setCache(cacheResult);
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

  useEffect(() => {
    refresh();
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
      const withinExperimentDialog = target instanceof Element && Boolean(target.closest("[data-experiment-details-dialog]"));
      if (menuRef.current && !menuRef.current.contains(target as Node) && !withinExperimentDialog) {
        setOpen(false);
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !document.querySelector("[data-experiment-details-dialog]")) {
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
        onClick={() => setOpen((value) => !value)}
        type="button"
      >
        Jobs
        {activeCount > 0 ? (
          <span className="ml-2 inline-flex min-w-5 items-center justify-center rounded-full bg-cyan px-1.5 py-0.5 text-xs font-semibold text-white">
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
              <h3 className="label">Downloads</h3>
              {activeDownloads.map((status) => {
                const job = status.active_job;
                const canceling = job?.status === "canceling" || status.cache_state === "canceling";
                return (
                  <div key={status.model_id} className="rounded-md border border-line bg-panel p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium">{status.model_id}</div>
                        <div className="mt-1 text-xs text-ink/60">
                          {formatState(job?.status ?? status.cache_state)} · {formatBytes(status.size_bytes)}
                        </div>
                      </div>
                      {job && activeDownloadStates.includes(job.status) ? (
                        <button className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={() => cancelDownload(job.job_id)} disabled={canceling || busyJob === job.job_id} title="Cancel download and delete partial cache" type="button">
                          {canceling || busyJob === job.job_id ? <Loader2 className="animate-spin" size={14} /> : <XCircle size={14} />}
                          {canceling || busyJob === job.job_id ? "Canceling" : "Cancel"}
                        </button>
                      ) : null}
                    </div>
                    <Progress active={!status.cached && activeDownloadStates.includes(status.cache_state)} complete={status.cached} />
                    <div className="mt-2 truncate font-mono text-xs text-ink/50">{job?.message ?? "n/a"}</div>
                  </div>
                );
              })}
              {activeDownloads.length === 0 ? <div className="text-sm text-ink/60">No active downloads.</div> : null}
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
    </div>
  );
}

function Progress({ active, complete }: { active: boolean; complete: boolean }) {
  return (
    <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-line">
      <div className={`h-full ${complete ? "w-full bg-cyan" : active ? "w-2/3 animate-pulse bg-cyan" : "w-0 bg-cyan"}`} />
    </div>
  );
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
