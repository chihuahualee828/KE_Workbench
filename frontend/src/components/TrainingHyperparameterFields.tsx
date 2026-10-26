"use client";

import type { TrainingMethod } from "@/lib/editingMethods";
import { LoraTargetModules } from "./LoraTargetModules";
import { trainingFields } from "@/lib/trainingHyperparameters";

export function TrainingHyperparameterFields({ method, modelId, modelLoaded, values, ready, disabled, warning, onChange, onReset }: {
  method: TrainingMethod;
  modelId: string;
  modelLoaded: boolean;
  values: Record<string, string>;
  ready: boolean;
  disabled: boolean;
  warning?: string | null;
  onChange: (key: string, value: string) => void;
  onReset: () => void;
}) {
  return (
    <details className="min-w-0 rounded-lg border border-line bg-panel p-3">
      <summary className="text-sm font-medium">Training hyperparameters</summary>
      <div className="mt-3 grid min-w-0 grid-cols-1 gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs text-ink/60">Defaults from the selected model’s YAML.</span>
          <button type="button" className="ghost-button min-h-8 px-2 py-1 text-xs" disabled={disabled || !ready} onClick={onReset}>Reset defaults</button>
        </div>
        {warning ? <p role="status" className="text-xs text-rust [overflow-wrap:anywhere]">{warning} Correct the affected fields before applying an edit.</p> : null}
        {!ready ? <p className="text-xs text-ink/60">Waiting for model defaults…</p> : trainingFields(method).map(({ key, label, hint }) => (
          <div key={key} className="grid min-w-0 grid-cols-1 gap-2">
            {key === "target_modules" ? (
              <LoraTargetModules key={modelId} modelId={modelId} modelLoaded={modelLoaded} disabled={disabled} value={values[key] ?? ""} onChange={(value) => onChange(key, value)} />
            ) : <label className="grid min-w-0 grid-cols-1 gap-1">
              <span className="label">{label}</span>
              {key === "objective_optimization" ? (
                <select className="control" value={values[key] ?? ""} disabled={disabled} onChange={(event) => onChange(key, event.target.value)}>
                  {values[key] !== "prompt_last" && values[key] !== "target_new" ? <option value={values[key] ?? ""}>Unsupported YAML value: {values[key]}</option> : null}
                  <option value="prompt_last">prompt_last (FT-L)</option>
                  <option value="target_new">target_new (FT-M)</option>
                </select>
              ) : (
                <input className="control" value={values[key] ?? ""} disabled={disabled} onChange={(event) => onChange(key, event.target.value)} spellCheck={false} />
              )}
              <span className="text-xs text-ink/60 [overflow-wrap:anywhere]">{hint}</span>
            </label>}
          </div>
        ))}
        {method === "lora" ? <p className="text-xs text-ink/60">Uses standard LoRA. Layers restrict where the target modules receive adapters.</p> : null}
      </div>
    </details>
  );
}
