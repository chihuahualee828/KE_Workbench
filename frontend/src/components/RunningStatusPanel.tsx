"use client";

import { Activity, Download, Loader2, RefreshCw, Server, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { Experiment, ModelCacheStatus, ModelRuntimeStatus } from "@/lib/types";

const activeDownloadStates = ["queued", "downloading", "canceling"];
const activeRuntimeStates = ["queued", "loading"];

export function RunningStatusPanel() {
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
  const hasActiveWork = activeDownloads.length > 0 || activeRuntime.length > 0 || runningExperiments.length > 0;

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
    if (!hasActiveWork) return;
    const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, [hasActiveWork]);

  return (
    <section className="panel">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Activity size={17} />
          <h2 className="text-base font-semibold">Running Jobs</h2>
        </div>
        <button className="ghost-button min-h-9 px-2 py-1" onClick={refresh} title="Refresh jobs">
          <RefreshCw size={15} />
          Refresh
        </button>
      </div>

      <div className="grid gap-4">
        <section className="grid gap-2">
          <div className="flex items-center gap-2">
            <Download size={15} />
            <h3 className="text-sm font-semibold">Model Downloads</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-line text-left">
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2 pr-4">Status</th>
                  <th className="py-2 pr-4">Downloaded</th>
                  <th className="py-2 pr-4">Progress</th>
                  <th className="py-2 pr-4">Job</th>
                  <th className="py-2 pr-4">Action</th>
                </tr>
              </thead>
              <tbody>
                {activeDownloads.map((status) => {
                  const job = status.active_job;
                  const canceling = job?.status === "canceling" || status.cache_state === "canceling";
                  return (
                    <tr key={status.model_id} className="border-b border-line/70 align-top">
                      <td className="py-2 pr-4">
                        <div className="font-medium">{status.model_id}</div>
                        <div className="font-mono text-xs text-ink/50">{status.hf_id}</div>
                      </td>
                      <td className="py-2 pr-4">{formatState(job?.status ?? status.cache_state)}</td>
                      <td className="py-2 pr-4">{formatBytes(status.size_bytes)}</td>
                      <td className="py-2 pr-4">
                        <Progress active={!status.cached && activeDownloadStates.includes(status.cache_state)} complete={status.cached} />
                      </td>
                      <td className="py-2 pr-4 text-xs text-ink/70">
                        {job ? `${job.job_id}: ${job.message}` : "n/a"}
                      </td>
                      <td className="py-2 pr-4">
                        {job && activeDownloadStates.includes(job.status) ? (
                          <button className="ghost-button min-h-9 px-2 py-1" onClick={() => cancelDownload(job.job_id)} disabled={canceling || busyJob === job.job_id} title="Cancel download and delete partial cache">
                            {canceling || busyJob === job.job_id ? <Loader2 className="animate-spin" size={15} /> : <XCircle size={15} />}
                            {canceling || busyJob === job.job_id ? "Canceling" : "Cancel"}
                          </button>
                        ) : null}
                      </td>
                    </tr>
                  );
                })}
                {activeDownloads.length === 0 ? (
                  <tr>
                    <td className="py-3 text-sm text-ink/60" colSpan={6}>No active model downloads.</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
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

        <section className="grid gap-2">
          <div className="flex items-center gap-2">
            <Activity size={15} />
            <h3 className="text-sm font-semibold">Experiments</h3>
          </div>
          <div className="divide-y divide-line rounded-md border border-line">
            {runningExperiments.map((experiment) => (
              <div key={experiment.id} className="grid grid-cols-[1fr_auto] gap-3 px-3 py-2 text-sm">
                <span>{experiment.type.replace("_", " ")}</span>
                <span className="font-mono text-xs text-ink/60">{experiment.status}</span>
              </div>
            ))}
            {runningExperiments.length === 0 ? <div className="px-3 py-3 text-sm text-ink/60">No active experiments.</div> : null}
          </div>
        </section>
      </div>
      {error ? <div className="mt-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
    </section>
  );
}

function Progress({ active, complete }: { active: boolean; complete: boolean }) {
  return (
    <div className="h-2 w-44 overflow-hidden rounded-full bg-line">
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
