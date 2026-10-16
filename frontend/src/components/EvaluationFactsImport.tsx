"use client";

import { Loader2, Upload, X } from "lucide-react";
import { useId, useMemo, useRef, useState } from "react";

import {
  detectEvaluationFactMapping, EVALUATION_FACT_IMPORT_FIELDS, evaluationFactImportFieldLabel,
  normalizeEvaluationFactImport, parseEvaluationFactImport, REQUIRED_EVALUATION_FACT_IMPORT_FIELDS,
  type EvaluationFactFieldMapping
} from "@/lib/evaluationFactImport";
import type { EvaluationFact } from "@/lib/types";

type PendingImport = { records: Record<string, unknown>[]; mapping: EvaluationFactFieldMapping; fileName: string };

export function EvaluationFactsImport({ disabled = false, onImport, onError }: {
  disabled?: boolean;
  onImport: (facts: EvaluationFact[], fileName: string) => Promise<void>;
  onError: (message: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [pendingImport, setPendingImport] = useState<PendingImport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function upload(file: File | null) {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const records = parseEvaluationFactImport(await file.text());
      setPendingImport({ records, mapping: detectEvaluationFactMapping(records), fileName: file.name });
      onError("");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function commit(mapping: EvaluationFactFieldMapping) {
    if (!pendingImport || busy) return;
    setBusy(true);
    setError("");
    try {
      await onImport(normalizeEvaluationFactImport(pendingImport.records, mapping), pendingImport.fileName);
      setPendingImport(null);
      onError("");
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
      onError(message);
    } finally {
      setBusy(false);
    }
  }

  return <>
    <button aria-label="Upload evaluation facts JSON" className="ghost-button grid size-8 shrink-0 place-items-center p-0" disabled={disabled || busy || Boolean(pendingImport)} onClick={() => inputRef.current?.click()} title="Upload evaluation facts JSON" type="button">
      {busy ? <Loader2 className="animate-spin" size={14} /> : <Upload size={14} />}
    </button>
    <input accept="application/json,.json" aria-label="Evaluation facts JSON file" className="sr-only" disabled={disabled || busy} onChange={(event) => {
      const file = event.currentTarget.files?.[0] ?? null;
      event.currentTarget.value = "";
      void upload(file);
    }} ref={inputRef} tabIndex={-1} type="file" />
    {pendingImport ? <EvaluationFactMappingDialog busy={busy} error={error} pending={pendingImport} onClose={() => setPendingImport(null)} onConfirm={(mapping) => void commit(mapping)} /> : null}
  </>;
}

function EvaluationFactMappingDialog({ busy, error, pending, onClose, onConfirm }: {
  busy: boolean;
  error: string;
  pending: PendingImport;
  onClose: () => void;
  onConfirm: (mapping: EvaluationFactFieldMapping) => void;
}) {
  const id = useId();
  const [mapping, setMapping] = useState(pending.mapping);
  const [validationError, setValidationError] = useState("");
  const keys = useMemo(() => Array.from(new Set(pending.records.flatMap((record) => Object.keys(record)))).sort((left, right) => left.localeCompare(right)), [pending.records]);

  function confirm() {
    const missing = REQUIRED_EVALUATION_FACT_IMPORT_FIELDS.find((field) => !mapping[field].trim());
    if (missing) {
      setValidationError(`Choose a JSON attribute for ${evaluationFactImportFieldLabel(missing)}.`);
      return;
    }
    const unavailable = Object.values(mapping).find((key) => key.trim() && !keys.includes(key));
    if (unavailable) {
      setValidationError(`JSON attribute "${unavailable}" was not found in this file.`);
      return;
    }
    setValidationError("");
    onConfirm(mapping);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/30 p-4 sm:items-center" onMouseDown={(event) => {
      if (!busy && event.target === event.currentTarget) onClose();
    }} role="presentation">
      <section aria-labelledby={`${id}-title`} aria-modal="true" className="my-auto w-full max-w-xl rounded-md border border-line bg-white p-5 shadow-xl" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id={`${id}-title`} className="text-base font-semibold">Map Imported Evaluation Facts</h2>
            <p className="mt-1 text-sm text-ink/60">{pending.records.length} {pending.records.length === 1 ? "fact" : "facts"} detected in {pending.fileName}. Common aliases were selected automatically.</p>
          </div>
          <button aria-label="Close evaluation import mapping" className="ghost-button grid size-8 shrink-0 place-items-center p-0" disabled={busy} onClick={onClose} title="Close" type="button"><X size={16} /></button>
        </div>
        <div className="mt-5 grid gap-3">
          {EVALUATION_FACT_IMPORT_FIELDS.map((field) => (
            <label className="grid gap-1" key={field}>
              <span className="label">{evaluationFactImportFieldLabel(field)} {REQUIRED_EVALUATION_FACT_IMPORT_FIELDS.some((required) => required === field) ? "(required)" : "(optional)"}</span>
              <input className="control font-mono text-sm" disabled={busy} list={`${id}-keys-${field}`} value={mapping[field]} onChange={(event) => {
                setMapping((current) => ({ ...current, [field]: event.target.value }));
                setValidationError("");
              }} />
              <datalist id={`${id}-keys-${field}`}>
                {keys.map((key) => <option key={key} value={key} />)}
              </datalist>
            </label>
          ))}
        </div>
        <p className="mt-4 text-xs leading-5 text-ink/60">Ground truth and metric prompts are optional. Facts without a rephrase use their original prompt; locality and portability require their own prompts.</p>
        {validationError || error ? <div className="mt-4 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust" role="alert">{validationError || error}</div> : null}
        <div className="mt-5 flex justify-end gap-2">
          <button className="ghost-button min-h-9 px-3 py-1 text-sm" disabled={busy} onClick={onClose} type="button">Cancel</button>
          <button className="tool-button min-h-9 px-3 py-1 text-sm" disabled={busy} onClick={confirm} type="button">{busy ? <Loader2 className="animate-spin" size={14} /> : null}Import facts</button>
        </div>
      </section>
    </div>
  );
}
