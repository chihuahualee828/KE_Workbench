"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import type { EditJob } from "@/lib/types";
import { useSessionState } from "@/lib/useSessionState";

type EditSessionState = {
  result: Record<string, unknown> | null;
  job: EditJob | null;
};

type WorkbenchSessionValue = {
  edit: EditSessionState;
  setEdit: (update: (current: EditSessionState) => EditSessionState) => void;
  evaluationExperimentId: string | null;
  setEvaluationExperimentId: (experimentId: string | null) => void;
  generationOutput: (scope: string) => string;
  setGenerationOutput: (scope: string, output: string) => void;
};

const WorkbenchSessionContext = createContext<WorkbenchSessionValue | null>(null);

export function WorkbenchSessionProvider({ children }: { children: React.ReactNode }) {
  const [edit, setEditState] = useState<EditSessionState>({ result: null, job: null });
  const [evaluationExperimentId, setEvaluationExperimentId] = useSessionState<string | null>("kew:evaluate:experimentId", null);
  const [generationOutputs, setGenerationOutputs] = useState<Record<string, string>>({});
  useEffect(() => {
    // Earlier versions stored the entire evaluation payload here.  Keep only a
    // small tab-scoped experiment id now; result cards are loaded from server pages.
    window.localStorage.removeItem("kew:evaluate:result");
  }, []);
  const setEdit = useCallback((update: (current: EditSessionState) => EditSessionState) => {
    setEditState(update);
  }, []);
  const generationOutput = useCallback((scope: string) => generationOutputs[scope] ?? "", [generationOutputs]);
  const setGenerationOutput = useCallback((scope: string, output: string) => {
    setGenerationOutputs((current) => ({ ...current, [scope]: output }));
  }, []);
  const value = useMemo(() => ({ edit, setEdit, evaluationExperimentId, setEvaluationExperimentId, generationOutput, setGenerationOutput }), [edit, evaluationExperimentId, generationOutput, setEdit, setGenerationOutput]);

  return <WorkbenchSessionContext.Provider value={value}>{children}</WorkbenchSessionContext.Provider>;
}

export function useWorkbenchSession() {
  const value = useContext(WorkbenchSessionContext);
  if (value === null) {
    throw new Error("useWorkbenchSession must be used within WorkbenchSessionProvider.");
  }
  return value;
}
