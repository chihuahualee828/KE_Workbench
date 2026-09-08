"use client";

import { createContext, useCallback, useContext, useMemo, useState } from "react";

import type { EditJob } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";

type EditSessionState = {
  result: Record<string, unknown> | null;
  job: EditJob | null;
};

type WorkbenchSessionValue = {
  edit: EditSessionState;
  setEdit: (update: (current: EditSessionState) => EditSessionState) => void;
  evaluationResult: Record<string, unknown> | null;
  setEvaluationResult: (result: Record<string, unknown> | null) => void;
  generationOutput: (scope: string) => string;
  setGenerationOutput: (scope: string, output: string) => void;
};

const WorkbenchSessionContext = createContext<WorkbenchSessionValue | null>(null);

export function WorkbenchSessionProvider({ children }: { children: React.ReactNode }) {
  const [edit, setEditState] = useState<EditSessionState>({ result: null, job: null });
  const [evaluationResult, setEvaluationResult] = usePersistentState<Record<string, unknown> | null>("kew:evaluate:result", null);
  const [generationOutputs, setGenerationOutputs] = useState<Record<string, string>>({});
  const setEdit = useCallback((update: (current: EditSessionState) => EditSessionState) => {
    setEditState(update);
  }, []);
  const generationOutput = useCallback((scope: string) => generationOutputs[scope] ?? "", [generationOutputs]);
  const setGenerationOutput = useCallback((scope: string, output: string) => {
    setGenerationOutputs((current) => ({ ...current, [scope]: output }));
  }, []);
  const value = useMemo(() => ({ edit, setEdit, evaluationResult, setEvaluationResult, generationOutput, setGenerationOutput }), [edit, evaluationResult, generationOutput, setEdit, setGenerationOutput]);

  return <WorkbenchSessionContext.Provider value={value}>{children}</WorkbenchSessionContext.Provider>;
}

export function useWorkbenchSession() {
  const value = useContext(WorkbenchSessionContext);
  if (value === null) {
    throw new Error("useWorkbenchSession must be used within WorkbenchSessionProvider.");
  }
  return value;
}
