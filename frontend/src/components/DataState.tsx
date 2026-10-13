"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { api, invalidateApiReads } from "@/lib/api";
import { getBackendConnection, saveBackendConnection, type BackendConnection } from "@/lib/backendConnection";
import { announceModelRuntimeConnected, announceModelRuntimeDisconnected, announceModelRuntimeStatuses, applyRuntimeJobToRuntime, applyRuntimeStatusToRuntime, listenForModelRuntimeLoads, listenForModelRuntimeStatuses, listenForModelRuntimeStream } from "@/lib/modelRuntimeEvents";
import { cacheResource, downloadResource, experimentResource, historyResource, modelResource, patchResource, runtimeResource, refreshTransfers, refreshExperimentHistory, refreshPatchHistory, reconcilePatchesAfterRuntimeUpdate, resetWorkbenchResources } from "@/lib/workbenchResources";
import { listenForExperimentStarts, listenForExperimentCanceled } from "@/lib/experimentEvents";
import { listenForModelDownloads, listenForModelCacheStatuses, applyDownloadJobToCache, applyCacheStatusToCache, upsertModelDownloadJob } from "@/lib/modelDownloadEvents";
import type { Experiment, ModelRuntimeStatus, EditJob } from "@/lib/types";
import { updatePatchesFromEditJob } from "@/lib/workbenchResources";

type WorkbenchData = ReturnType<typeof useWorkbenchDataStore>;

const WorkbenchDataContext = createContext<WorkbenchData | null>(null);

export function WorkbenchDataProvider({ children }: { children: ReactNode }) {
  const value = useWorkbenchDataStore();
  return <WorkbenchDataContext.Provider value={value}>{children}</WorkbenchDataContext.Provider>;
}

export function useWorkbenchData() {
  const value = useContext(WorkbenchDataContext);
  if (!value) throw new Error("useWorkbenchData must be used inside WorkbenchDataProvider.");
  return value;
}

function useWorkbenchDataStore() {
  const [models, setModels] = modelResource.use();
  const [patches, setPatches] = patchResource.use();
  const [experiments, setExperiments] = experimentResource.use();
  const [runtime, setRuntime] = runtimeResource.use();
  const [, setCache] = cacheResource.use();
  const [downloadJobs, setDownloadJobs] = downloadResource.use();
  const [totals] = historyResource.use();
  const connectionVersion = useRef(0);
  const [status, setStatus] = useState<"loading" | "ready" | "offline">("loading");
  const [backend, setBackend] = useState<BackendConnection>(() => getBackendConnection());
  const [backendHydrated, setBackendHydrated] = useState(false);
  const reconnecting = useRef(false);
  const runtimeSnapshot = useRef<ModelRuntimeStatus[]>([]);
  const broadcastingRuntime = useRef(false);

  const refresh = useCallback(async () => {
    const version = connectionVersion.current;
    try {
      const [modelResult, runtimeResult] = await Promise.all([api.models(), api.modelRuntime(), refreshExperimentHistory(), refreshPatchHistory(), refreshTransfers()]);
      if (version !== connectionVersion.current) return;
      setModels(modelResult);
      setRuntime(runtimeResult);
      const previousRuntime = runtimeSnapshot.current;
      runtimeSnapshot.current = runtimeResult;
      await reconcilePatchesAfterRuntimeUpdate(previousRuntime, runtimeResult);
      if (version !== connectionVersion.current) return;
      announceModelRuntimeConnected();
      setStatus("ready");
    } catch (error) {
      if (version !== connectionVersion.current) return;
      setModels([]);
      setPatches([]);
      setExperiments([]);
      setRuntime([]);
      announceModelRuntimeDisconnected();
      setStatus("offline");
      throw error;
    }
  }, []);

  const changeBackend = useCallback(async (connection: BackendConnection) => {
    const saved = saveBackendConnection(connection);
    connectionVersion.current += 1;
    invalidateApiReads();
    resetWorkbenchResources();
    setBackend(saved);
    setStatus("loading");
    setModels([]);
    setPatches([]);
    setExperiments([]);
    setRuntime([]);
    runtimeSnapshot.current = [];
    announceModelRuntimeDisconnected();
    await refresh();
  }, [refresh]);

  useEffect(() => {
    // Read persisted browser-only settings after hydration so a saved remote
    // backend replaces the build-time fallback before the first connection check.
    setBackend(getBackendConnection());
    setBackendHydrated(true);
  }, []);

  useEffect(() => {
    if (!backendHydrated) return;
    void refresh().catch(() => undefined);
  }, [backendHydrated, refresh]);

  useEffect(() => {
    if (!backendHydrated) return;
    const stopListeningForLoads = listenForModelRuntimeLoads((job) => {
      setRuntime((current) => applyRuntimeJobToRuntime(current, job));
      if (job.status === "completed") {
        void refresh().catch(() => undefined);
      }
    });
    const stopListeningForStatuses = listenForModelRuntimeStatuses((runtimeStatus) => {
      if (broadcastingRuntime.current) return;
      const previousRuntime = runtimeSnapshot.current;
      runtimeSnapshot.current = applyRuntimeStatusToRuntime(previousRuntime, runtimeStatus);
      setRuntime((current) => applyRuntimeStatusToRuntime(current, runtimeStatus));
      void reconcilePatchesAfterRuntimeUpdate(previousRuntime, [runtimeStatus]).catch(() => undefined);
    });
    const stopListeningForRuntimeStream = listenForModelRuntimeStream(
      (statuses) => {
        const previousRuntime = runtimeSnapshot.current;
        runtimeSnapshot.current = statuses;
        announceModelRuntimeConnected();
        broadcastingRuntime.current = true;
        try { announceModelRuntimeStatuses(statuses); } finally { broadcastingRuntime.current = false; }
        setRuntime(statuses);
        setStatus("ready");
        void reconcilePatchesAfterRuntimeUpdate(previousRuntime, statuses).catch(() => undefined);
      },
      () => {
        reconnecting.current = true;
        announceModelRuntimeDisconnected();
        setModels([]);
        setPatches([]);
        setExperiments([]);
        setRuntime([]);
        runtimeSnapshot.current = [];
        setStatus("offline");
      },
      () => {
        if (!reconnecting.current) return;
        reconnecting.current = false;
        void refresh().catch(() => undefined);
      }
    );
    return () => {
      stopListeningForLoads();
      stopListeningForStatuses();
      stopListeningForRuntimeStream();
    };
  }, [backend.baseUrl, backend.token, backendHydrated, refresh]);

  useEffect(() => {
    if (!backendHydrated) return;
    const stopDownloads = listenForModelDownloads((job) => {
      setDownloadJobs((current) => upsertModelDownloadJob(current, job));
      setCache((current) => applyDownloadJobToCache(current, job));
    });
    const stopCache = listenForModelCacheStatuses((value) => setCache((current) => applyCacheStatusToCache(current, value)));
    const stopStarts = listenForExperimentStarts(() => { invalidateApiReads(); void refreshExperimentHistory().catch(() => undefined); });
    const stopCanceled = listenForExperimentCanceled(() => { invalidateApiReads(); void refreshExperimentHistory().catch(() => undefined); });
    return () => { stopDownloads(); stopCache(); stopStarts(); stopCanceled(); };
  }, [backendHydrated]);

  const activeTransfers = downloadJobs.filter((job) => ["queued", "downloading", "validating", "registering", "canceling"].includes(job.status)).map((job) => job.job_id).sort().join(",");
  useEffect(() => {
    if (!activeTransfers) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try { await refreshTransfers(); } catch { /* The next cycle retries. */ }
      if (!stopped) timer = setTimeout(poll, 1500);
    };
    timer = setTimeout(poll, 1500);
    return () => { stopped = true; clearTimeout(timer); };
  }, [activeTransfers, backend.baseUrl, backend.token]);

  const activeJobs = experiments.filter((item) => item.status === "running" && item.job?.id).map((item) => item.job!.id).sort().join(",");
  useEffect(() => {
    if (!activeJobs) return;
    const stops = activeJobs.split(",").map((id) => api.subscribeToEvents(`/api/edit/jobs/${id}/events?include_result=false`, {
      onEvent: (event, data) => {
        if (event !== "job") return;
        const job = JSON.parse(data) as EditJob;
        updatePatchesFromEditJob(job);
        setExperiments((current) => current.map((item) => item.id === job.experiment_id ? {
          ...item, status: job.status as Experiment["status"], job: item.job ? { ...item.job, status: job.status as Experiment["status"], progress: job.progress, progress_current: job.progress_current, progress_total: job.progress_total, progress_phase: job.progress_phase } : item.job
        } : item));
        if (job.status !== "running") {
          invalidateApiReads();
          if (experiments.find((item) => item.id === job.experiment_id)?.type !== "evaluation") void refreshPatchHistory().catch(() => undefined);
          void refreshExperimentHistory().catch(() => undefined);
        }
      }
    }));
    return () => stops.forEach((stop) => stop());
  }, [activeJobs, backend.baseUrl, backend.token]);

  return { models, patches, experiments, runtime, status, backend, changeBackend, refresh, totals };
}

export function ApiStatus({ status }: { status: "loading" | "ready" | "offline" }) {
  const text = status === "ready" ? "Backend connected" : status === "loading" ? "Connecting…" : "Backend offline";
  return (
    <div role="status" className={`status-badge ${status === "ready" ? "border-emerald-200/70 bg-emerald-50 text-emerald-700" : status === "loading" ? "border-line bg-white text-ink/50" : "border-rust/20 bg-rust/5 text-rust"}`} title={status === "offline" ? "The backend is unavailable. Check that the backend service is running." : undefined}>
      <span className={`h-1.5 w-1.5 rounded-full ${status === "ready" ? "bg-emerald-500" : status === "loading" ? "animate-pulse bg-ink/30" : "bg-rust"}`} />
      {text}
    </div>
  );
}
