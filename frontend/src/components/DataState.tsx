"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { api } from "@/lib/api";
import { getBackendConnection, saveBackendConnection, type BackendConnection } from "@/lib/backendConnection";
import { announceModelRuntimeConnected, announceModelRuntimeDisconnected, announceModelRuntimeStatuses, applyRuntimeJobToRuntime, applyRuntimeStatusToRuntime, listenForModelRuntimeLoads, listenForModelRuntimeStatuses, listenForModelRuntimeStream } from "@/lib/modelRuntimeEvents";
import type { Experiment, ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";

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
  const [models, setModels] = useState<SupportedModel[]>([]);
  const [patches, setPatches] = useState<Patch[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeStatus[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "offline">("loading");
  const [backend, setBackend] = useState<BackendConnection>(() => getBackendConnection());
  const [backendHydrated, setBackendHydrated] = useState(false);
  const reconnecting = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const [modelResult, patchResult, experimentResult, runtimeResult] = await Promise.all([api.models(), api.patches(), api.experiments(), api.modelRuntime()]);
      setModels(modelResult);
      setPatches(patchResult.patches.filter((patch) => patch.patch_id));
      setExperiments(experimentResult.filter((experiment) => experiment.id?.startsWith("exp_")));
      setRuntime(runtimeResult);
      announceModelRuntimeConnected();
      setStatus("ready");
    } catch (error) {
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
    setBackend(saved);
    setStatus("loading");
    setModels([]);
    setPatches([]);
    setExperiments([]);
    setRuntime([]);
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
      setRuntime((current) => applyRuntimeStatusToRuntime(current, runtimeStatus));
    });
    const stopListeningForRuntimeStream = listenForModelRuntimeStream(
      (statuses) => {
        announceModelRuntimeConnected();
        announceModelRuntimeStatuses(statuses);
        setRuntime(statuses);
        setStatus("ready");
      },
      () => {
        reconnecting.current = true;
        announceModelRuntimeDisconnected();
        setModels([]);
        setPatches([]);
        setExperiments([]);
        setRuntime([]);
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

  return { models, patches, experiments, runtime, status, backend, changeBackend, refresh };
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
