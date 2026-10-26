"use client";

import type { EditJob } from "../lib/types";
import { EvaluationProgressIndicator } from "./EvaluationProgressIndicator";

export function EditJobProgress({ job }: { job: EditJob | null }) {
  if (!job || job.status !== "running") return null;
  if (job.progress_phase === "post_edit_evaluation") return <EvaluationProgressIndicator className="mb-2" preparing={false} current={job.progress_current ?? undefined} total={job.progress_total ?? undefined} multiFactLabel="Post-edit evaluation and generation" singleFactLabel="Evaluating and generating for the final model…" />;
  const completed = job.progress_current ?? 0;
  const total = job.progress_total ?? 0;
  const determinate = total > 0;
  const percent = Math.max(0, Math.min(100, Math.round((determinate ? completed / total : job.progress) * 100)));
  const phase = job.progress >= 0.1 ? "Editing" : "Preparing";
  const batch = job.progress_phase === "batch_editing" || job.progress_phase === "batch_completed";
  const currentBatch = job.progress_phase === "batch_editing" ? Math.min(completed + 1, total) : completed;
  const batchLabel = completed >= total
    ? `Finalizing edit — ${total} / ${total} batches edited`
    : `Batch ${currentBatch} / ${total} — ${completed} completed (${percent}%)`;
  const label = determinate ? (batch ? batchLabel : `${completed} / ${total} facts (${percent}%)`) : phase;

  return (
    <div className="mb-2 grid gap-1.5" aria-label={batch && determinate ? label : determinate ? `${completed} of ${total} sequential edits complete` : `${phase} edit`} role="status">
      <div className="h-1.5 overflow-hidden rounded-full bg-line">
        <div className={determinate ? "h-full bg-cyan transition-[width] duration-300" : "edit-progress-indeterminate h-full bg-cyan"} style={determinate ? { width: `${percent}%` } : undefined} />
      </div>
      <span className="font-mono text-xs text-ink/60">{label}</span>
    </div>
  );
}
