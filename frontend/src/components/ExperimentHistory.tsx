"use client";

import { CheckCircle2, CircleAlert, Clock3, Loader2, Trash2, X } from "lucide-react";
import { createPortal } from "react-dom";
import { type PointerEvent, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { Experiment, ExperimentDetails } from "@/lib/types";

export function ExperimentHistory({
  experiments,
  compact = false,
  limit,
  showControls = true,
  onChanged
}: {
  experiments: Experiment[];
  compact?: boolean;
  limit?: number;
  showControls?: boolean;
  onChanged: () => void | Promise<void>;
}) {
  const [details, setDetails] = useState<ExperimentDetails | null>(null);
  const [confirmingClear, setConfirmingClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [deletingExperimentId, setDeletingExperimentId] = useState("");
  const [error, setError] = useState("");
  const sortedExperiments = useMemo(
    () => [...experiments].sort((left, right) => right.created_at.localeCompare(left.created_at)),
    [experiments]
  );
  const displayedExperiments = limit === undefined ? sortedExperiments : sortedExperiments.slice(0, limit);
  const hasRunningExperiment = experiments.some((experiment) => experiment.status === "running");

  async function openDetails(experiment: Experiment) {
    setError("");
    try {
      setDetails(await api.experiment(experiment.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function clearHistory() {
    setClearing(true);
    setError("");
    try {
      await api.clearExperiments();
      setConfirmingClear(false);
      await onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setClearing(false);
    }
  }

  async function deleteExperiment(experiment: Experiment) {
    if (experiment.status === "running") return;
    setDeletingExperimentId(experiment.id);
    setError("");
    try {
      await api.deleteExperiment(experiment.id);
      setDetails((current) => current?.id === experiment.id ? null : current);
      await onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingExperimentId("");
    }
  }

  return (
    <section className="grid gap-2">
      {showControls ? (
        <div className="flex items-center justify-between gap-3">
          <h3 className="label">Experiments</h3>
          <button
            className="ghost-button min-h-8 px-2 py-1 text-xs"
            disabled={experiments.length === 0 || hasRunningExperiment || clearing}
            onClick={() => setConfirmingClear(true)}
            title={hasRunningExperiment ? "Wait for running experiments to finish before clearing history" : "Clear completed and failed experiment history"}
            type="button"
          >
            <Trash2 size={14} />
            Clear history
          </button>
        </div>
      ) : null}

      {showControls && confirmingClear ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm">
          <span>Delete all completed and failed experiment records?</span>
          <div className="flex gap-2">
            <button className="ghost-button min-h-8 px-2 py-1 text-xs" disabled={clearing} onClick={() => setConfirmingClear(false)} type="button">Cancel</button>
            <button className="ghost-button min-h-8 px-2 py-1 text-xs text-rust" disabled={clearing} onClick={clearHistory} type="button">
              {clearing ? <Loader2 className="animate-spin" size={14} /> : <Trash2 size={14} />}
              {clearing ? "Clearing" : "Clear"}
            </button>
          </div>
        </div>
      ) : null}

      <div className={`divide-y divide-line rounded-md border border-line ${compact ? "max-h-72 overflow-y-auto" : "max-h-[540px] overflow-y-auto"}`}>
        {displayedExperiments.map((experiment) => (
          <ExperimentHistoryCard
            deleting={deletingExperimentId === experiment.id}
            experiment={experiment}
            key={experiment.id}
            swipeToDelete={experiment.status !== "running"}
            onDelete={() => void deleteExperiment(experiment)}
            onOpen={() => void openDetails(experiment)}
          />
        ))}
        {displayedExperiments.length === 0 ? <div className="px-3 py-3 text-sm text-ink/60">No experiments recorded yet.</div> : null}
      </div>

      {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-2 text-xs text-rust">{error}</div> : null}
      {details ? <ExperimentDetailsDialog details={details} onClose={() => setDetails(null)} /> : null}
    </section>
  );
}

function ExperimentHistoryCard({
  experiment,
  deleting,
  swipeToDelete,
  onDelete,
  onOpen
}: {
  experiment: Experiment;
  deleting: boolean;
  swipeToDelete: boolean;
  onDelete: () => void;
  onOpen: () => void;
}) {
  const [deleteRevealed, setDeleteRevealed] = useState(false);
  const pointerStartX = useRef<number | null>(null);
  const suppressNextOpen = useRef(false);

  function handlePointerDown(event: PointerEvent<HTMLButtonElement>) {
    if (!swipeToDelete) return;
    pointerStartX.current = event.clientX;
  }

  function handlePointerUp(event: PointerEvent<HTMLButtonElement>) {
    if (!swipeToDelete || pointerStartX.current === null) return;
    const deltaX = event.clientX - pointerStartX.current;
    pointerStartX.current = null;
    if (deltaX <= -48) {
      suppressNextOpen.current = true;
      setDeleteRevealed(true);
    } else if (deltaX >= 24 && deleteRevealed) {
      suppressNextOpen.current = true;
      setDeleteRevealed(false);
    }
  }

  function handleOpen() {
    if (suppressNextOpen.current) {
      suppressNextOpen.current = false;
      return;
    }
    onOpen();
  }

  const content = (
    <>
      <span className="min-w-0">
        <span className="flex items-center gap-2 font-medium">
          <ExperimentStatusIcon status={experiment.status} />
          <span className="truncate">{formatExperimentType(experiment.type)}</span>
        </span>
        <span className="mt-1 block truncate font-mono text-xs text-ink/60">{experiment.model_id} · {formatTimestamp(experiment.created_at)}</span>
      </span>
      <span className={`rounded-md border px-2 py-1 font-mono text-xs ${statusClassName(experiment.status)}`}>
        {experiment.status}
      </span>
    </>
  );

  if (!swipeToDelete) {
    return (
      <button className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 py-2 text-left text-sm transition hover:bg-panel" onClick={onOpen} type="button">
        {content}
      </button>
    );
  }

  return (
    <div className="relative overflow-hidden bg-rust">
      <button
        aria-label={`Delete ${formatExperimentType(experiment.type)} experiment`}
        className="absolute inset-y-0 right-0 grid w-20 place-items-center bg-rust text-xs font-medium text-white"
        disabled={deleting}
        onClick={onDelete}
        type="button"
      >
        {deleting ? <Loader2 className="animate-spin" size={16} /> : <Trash2 size={17} />}
        <span>Delete</span>
      </button>
      <button
        className="relative grid w-full touch-pan-y grid-cols-[minmax(0,1fr)_auto] items-center gap-3 bg-white px-3 py-2 text-left text-sm transition-transform duration-200 hover:bg-panel"
        onClick={handleOpen}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        style={{ transform: deleteRevealed ? "translateX(-80px)" : undefined }}
        type="button"
      >
        {content}
      </button>
    </div>
  );
}

function ExperimentDetailsDialog({ details, onClose }: { details: ExperimentDetails; onClose: () => void }) {
  if (typeof document === "undefined") return null;
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/30 p-4 sm:items-center"
      data-experiment-details-dialog
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      role="presentation"
    >
      <section aria-modal="true" className="my-auto w-full max-w-3xl overflow-hidden rounded-md border border-line bg-white shadow-xl" role="dialog">
        <div className="max-h-[90vh] overflow-y-auto p-5">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-base font-semibold">
                <ExperimentStatusIcon status={details.status} />
                <span>{formatExperimentType(details.type)}</span>
              </div>
              <div className="mt-1 font-mono text-xs text-ink/60">{details.id}</div>
            </div>
            <button aria-label="Close experiment details" className="ghost-button grid size-8 place-items-center p-0" onClick={onClose} title="Close" type="button"><X size={16} /></button>
          </div>

          <dl className="mt-5 grid gap-3 text-sm sm:grid-cols-2">
            <Detail label="Status" value={details.status} />
            <Detail label="Model" value={details.model_id} />
            <Detail label="Created" value={formatTimestamp(details.created_at)} />
            <Detail label="Updated" value={formatTimestamp(details.updated_at ?? details.created_at)} />
            {details.job ? <Detail label="Job" value={`${details.job.id} · ${details.job.status}`} /> : null}
            {details.error_message ? <Detail label="Error" value={details.error_message} /> : null}
          </dl>

          <JsonBlock label="Request" value={details.input_json ?? {}} />
          {details.result ? <JsonBlock label="Result" value={details.result} /> : null}
          {details.job?.logs.length ? <JsonBlock label="Job logs" value={details.job.logs} /> : null}
        </div>
      </section>
    </div>,
    document.body
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="mt-1 break-words text-sm text-ink/80">{value}</dd>
    </div>
  );
}

function JsonBlock({ label, value }: { label: string; value: unknown }) {
  return (
    <section className="mt-5">
      <h3 className="label">{label}</h3>
      <div className="mt-2 max-h-72 overflow-hidden rounded-md bg-ink">
        <pre className="max-h-72 overflow-auto p-3 text-xs text-white">{JSON.stringify(value, null, 2)}</pre>
      </div>
    </section>
  );
}

function ExperimentStatusIcon({ status }: { status: Experiment["status"] }) {
  if (status === "completed") return <CheckCircle2 className="text-cyan" size={15} />;
  if (status === "failed") return <CircleAlert className="text-rust" size={15} />;
  return <Clock3 className="text-ink/60" size={15} />;
}

function formatExperimentType(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatTimestamp(value: string) {
  return value.replace("T", " ").replace(/\.\d+\+00:00$/, " UTC").replace("+00:00", " UTC");
}

function statusClassName(status: Experiment["status"]) {
  if (status === "completed") return "border-cyan/30 bg-cyan/10 text-cyan";
  if (status === "failed") return "border-rust/40 bg-rust/10 text-rust";
  return "border-line bg-panel text-ink/70";
}
