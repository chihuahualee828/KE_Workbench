"use client";

import { CheckCircle2, CircleAlert, Clock3, Loader2, Trash2, X } from "lucide-react";
import { createPortal } from "react-dom";
import { useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { Experiment, ExperimentDetails } from "@/lib/types";

export function ExperimentHistory({
  experiments,
  compact = false,
  onChanged
}: {
  experiments: Experiment[];
  compact?: boolean;
  onChanged: () => void | Promise<void>;
}) {
  const [details, setDetails] = useState<ExperimentDetails | null>(null);
  const [confirmingClear, setConfirmingClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState("");
  const sortedExperiments = useMemo(
    () => [...experiments].sort((left, right) => right.created_at.localeCompare(left.created_at)),
    [experiments]
  );
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

  return (
    <section className="grid gap-2">
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

      {confirmingClear ? (
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
        {sortedExperiments.map((experiment) => (
          <button
            className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 py-2 text-left text-sm transition hover:bg-panel"
            key={experiment.id}
            onClick={() => openDetails(experiment)}
            type="button"
          >
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
          </button>
        ))}
        {sortedExperiments.length === 0 ? <div className="px-3 py-3 text-sm text-ink/60">No experiments recorded yet.</div> : null}
      </div>

      {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-2 text-xs text-rust">{error}</div> : null}
      {details ? <ExperimentDetailsDialog details={details} onClose={() => setDetails(null)} /> : null}
    </section>
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
      <section aria-modal="true" className="my-auto max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-md border border-line bg-white p-5 shadow-xl" role="dialog">
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
      <pre className="mt-2 max-h-72 overflow-auto rounded-md bg-ink p-3 text-xs text-white">{JSON.stringify(value, null, 2)}</pre>
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
