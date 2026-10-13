"use client";

import { Loader2, X } from "lucide-react";

export function EvaluationProgressIndicator({
  current = 0,
  total = 0,
  preparingLabel = "Preparing EasyEdit…",
  multiFactLabel = "Running evaluation…",
  singleFactLabel = "Evaluating fact…",
  onCancel,
  canceling = false,
  cancelDisabled = false,
  preparing: preparingOverride,
  className = ""
}: {
  current?: number;
  total?: number;
  preparingLabel?: string;
  multiFactLabel?: string;
  singleFactLabel?: string;
  onCancel?: () => void;
  canceling?: boolean;
  cancelDisabled?: boolean;
  preparing?: boolean;
  className?: string;
}) {
  const preparing = preparingOverride ?? current <= 0;
  const determinate = total > 1;
  const percent = determinate ? Math.max(0, Math.min(100, Math.round((current / total) * 100))) : 0;
  const label = preparing ? preparingLabel : determinate ? multiFactLabel : singleFactLabel;

  return <div className={`grid gap-1.5 ${className}`.trim()} role="status" aria-label={preparing ? preparingLabel : determinate ? `${current} of ${total} facts evaluated` : "Evaluation in progress"}>
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-center gap-2 text-xs text-cyan"><Loader2 size={14} className="animate-spin" />{label}</div>
      {onCancel ? <button className="ghost-button min-h-8 shrink-0 px-2 py-1 text-xs text-rust" disabled={canceling || cancelDisabled} onClick={onCancel} title="Cancel evaluation" type="button">{canceling ? <Loader2 size={14} className="animate-spin" /> : <X size={14} />}{canceling ? "Canceling" : "Cancel"}</button> : null}
    </div>
    <div className="h-1.5 overflow-hidden rounded-full bg-line"><div className={determinate ? "h-full bg-cyan transition-[width] duration-300" : "edit-progress-indeterminate h-full bg-cyan"} style={determinate ? { width: `${percent}%` } : undefined} /></div>
    <span className="font-mono text-xs text-ink/60">{preparing ? "Loading evaluator" : determinate ? `${current} / ${total} facts (${percent}%)` : "Working"}</span>
  </div>;
}
