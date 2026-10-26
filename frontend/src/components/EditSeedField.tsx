"use client";

import { DEFAULT_EDIT_SEED, MAX_EDIT_SEED, supportsEditSeed, type EditMethod } from "../lib/editingMethods";
import { usePersistentState } from "../lib/usePersistentState";

export function useEditSeed(method: EditMethod) {
  return usePersistentState(`kew:edit:${method}:seed`, String(DEFAULT_EDIT_SEED));
}

export function EditSeedField({ method, value, onChange, disabled }: {
  method: EditMethod;
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
}) {
  if (!supportsEditSeed(method)) return null;
  return (
    <label className="grid gap-2">
      <span className="label">Random seed</span>
      <input className="control" type="number" name="seed" min={0} max={MAX_EDIT_SEED}
        step={1} value={value} onChange={(event) => onChange(event.target.value)} disabled={disabled} />
      <span className="text-xs text-ink/60">Controls randomness for the whole edit job. Identical results also depend on the model and execution settings.</span>
    </label>
  );
}
