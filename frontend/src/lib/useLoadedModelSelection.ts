"use client";

import { useEffect } from "react";

import { listenForActiveRuntimeModel } from "./modelRuntimeEvents";
import type { ModelRuntimeStatus, SupportedModel } from "./types";

export function useLoadedModelSelection(
  models: SupportedModel[],
  runtime: ModelRuntimeStatus[],
  setModelId: (modelId: string) => void
) {
  useEffect(() => {
    const loadedModel = runtime.find((status) => status.loaded && models.some((model) => model.id === status.model_id));
    if (loadedModel) setModelId(loadedModel.model_id);
  }, [models, runtime, setModelId]);

  useEffect(() => {
    return listenForActiveRuntimeModel((modelId) => {
      if (models.some((model) => model.id === modelId)) setModelId(modelId);
    });
  }, [models, setModelId]);
}
