"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import { applyRuntimeJobToRuntime, applyRuntimeStatusToRuntime, listenForModelRuntimeLoads, listenForModelRuntimeStatuses } from "@/lib/modelRuntimeEvents";
import type { Experiment, ModelRuntimeStatus, Patch, SupportedModel } from "@/lib/types";

const fallbackModels: SupportedModel[] = [
  {
    id: "gpt2",
    name: "GPT-2 Small",
    hf_id: "gpt2",
    n_layers: 12,
    supports_generation: true,
    supports_logit_lens: true,
    supports_causal_trace: true,
    supports_rome: false,
    supports_memit: false
  },
  {
    id: "gpt2-xl",
    name: "GPT-2 XL",
    hf_id: "gpt2-xl",
    n_layers: 48,
    supports_generation: true,
    supports_logit_lens: false,
    supports_causal_trace: false,
    supports_rome: true,
    supports_memit: true,
    easyedit: {
      rome_hparams: "hparams/ROME/gpt2-xl.yaml",
      memit_hparams: "hparams/MEMIT/gpt2-xl.yaml"
    }
  }
];

export function useWorkbenchData() {
  const [models, setModels] = useState<SupportedModel[]>(fallbackModels);
  const [patches, setPatches] = useState<Patch[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeStatus[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "offline">("loading");

  const refresh = useCallback(async () => {
    const [modelResult, patchResult, experimentResult, runtimeResult] = await Promise.all([api.models(), api.patches(), api.experiments(), api.modelRuntime()]);
    setModels(modelResult);
    setPatches(patchResult.patches.filter((patch) => patch.patch_id));
    setExperiments(experimentResult.filter((experiment) => experiment.id?.startsWith("exp_")));
    setRuntime(runtimeResult);
    setStatus("ready");
  }, []);

  useEffect(() => {
    let alive = true;
    Promise.all([api.models(), api.patches(), api.experiments(), api.modelRuntime()])
      .then(([modelResult, patchResult, experimentResult, runtimeResult]) => {
        if (!alive) return;
        setModels(modelResult);
        setPatches(patchResult.patches.filter((patch) => patch.patch_id));
        setExperiments(experimentResult.filter((experiment) => experiment.id?.startsWith("exp_")));
        setRuntime(runtimeResult);
        setStatus("ready");
      })
      .catch(() => {
        if (!alive) return;
        setStatus("offline");
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const stopListeningForLoads = listenForModelRuntimeLoads((job) => {
      setRuntime((current) => applyRuntimeJobToRuntime(current, job));
    });
    const stopListeningForStatuses = listenForModelRuntimeStatuses((runtimeStatus) => {
      setRuntime((current) => applyRuntimeStatusToRuntime(current, runtimeStatus));
    });
    return () => {
      stopListeningForLoads();
      stopListeningForStatuses();
    };
  }, []);

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
