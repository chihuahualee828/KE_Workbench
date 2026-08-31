"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { announceModelRuntimeConnected, announceModelRuntimeDisconnected, announceModelRuntimeStatuses, applyRuntimeJobToRuntime, applyRuntimeStatusToRuntime, listenForModelRuntimeLoads, listenForModelRuntimeStatuses, listenForModelRuntimeStream } from "@/lib/modelRuntimeEvents";
import type { Experiment, ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";

export function useWorkbenchData() {
  const [models, setModels] = useState<SupportedModel[]>([]);
  const [patches, setPatches] = useState<Patch[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeStatus[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "offline">("loading");
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

  useEffect(() => {
    void refresh().catch(() => undefined);
  }, [refresh]);

  useEffect(() => {
    const stopListeningForLoads = listenForModelRuntimeLoads((job) => {
      setRuntime((current) => applyRuntimeJobToRuntime(current, job));
    });
    const stopListeningForStatuses = listenForModelRuntimeStatuses((runtimeStatus) => {
      setRuntime((current) => applyRuntimeStatusToRuntime(current, runtimeStatus));
    });
    const stopListeningForRuntimeStream = listenForModelRuntimeStream(
      (statuses) => {
        const shouldRefresh = reconnecting.current;
        reconnecting.current = false;
        announceModelRuntimeConnected();
        announceModelRuntimeStatuses(statuses);
        setRuntime(statuses);
        setStatus("ready");
        if (shouldRefresh) {
          void refresh().catch(() => undefined);
        }
      },
      () => {
        reconnecting.current = true;
        announceModelRuntimeDisconnected();
        setModels([]);
        setPatches([]);
        setExperiments([]);
        setRuntime([]);
        setStatus("offline");
      }
    );
    return () => {
      stopListeningForLoads();
      stopListeningForStatuses();
      stopListeningForRuntimeStream();
    };
  }, [refresh]);

  return { models, patches, experiments, runtime, status, refresh };
}

export function ApiStatus({ status }: { status: "loading" | "ready" | "offline" }) {
  const text = status === "ready" ? "API connected" : status === "loading" ? "Connecting" : "API offline";
  return (
    <div className="inline-flex items-center gap-2 rounded-md border border-line bg-white px-3 py-2 text-sm">
      <span className={`h-2 w-2 rounded-full ${status === "ready" ? "bg-cyan" : status === "loading" ? "bg-rust" : "bg-ink/30"}`} />
      {text}
    </div>
  );
}
