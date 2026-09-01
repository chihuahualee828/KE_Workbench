"use client";

import { Layers, Save, Trash2, X, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { ModelRuntimeStatus, Patch } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";

export function PatchHistory({
  patches,
  runtime = [],
  onPatchesChanged
}: {
  patches: Patch[];
  runtime?: ModelRuntimeStatus[];
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const [busyPatchId, setBusyPatchId] = useState("");
  const [savingPatchId, setSavingPatchId] = useState("");
  const [loadingPatchId, setLoadingPatchId] = useState("");
  const [deletingPatchId, setDeletingPatchId] = useState("");
  const [message, setMessage] = useState("");
  const [selectedPatch, setSelectedPatch] = useState<Patch | null>(null);
  const [patchPendingDeletion, setPatchPendingDeletion] = useState<Patch | null>(null);
  const [confirmingClear, setConfirmingClear] = useState(false);
  const [clearingHistory, setClearingHistory] = useState(false);
  const [exportDirectory] = usePersistentState("kew:settings:modelExportPath", "");
  const [error, setError] = useState("");
  const loadedModelIds = useMemo(
    () => new Set(runtime.filter((status) => status.loaded).map((status) => status.model_id)),
    [runtime]
  );
  const activePatches = useMemo(
    () =>
      patches
        .filter((patch) => patch.active)
        .sort((left, right) => {
          if (left.base_model !== right.base_model) return left.base_model.localeCompare(right.base_model);
          return (left.active_index ?? 0) - (right.active_index ?? 0);
        }),
    [patches]
  );
  const savedPatches = useMemo(
    () =>
      [...patches].sort(
        (left, right) => new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
      ),
    [patches]
  );
  const latestActiveIndexByModel = useMemo(() => {
    const latest: Record<string, number> = {};
    activePatches.forEach((patch) => {
      if (!patch.active || patch.active_index == null) return;
      latest[patch.base_model] = Math.max(latest[patch.base_model] ?? -1, patch.active_index);
    });
    return latest;
  }, [activePatches]);

  async function discardPatch(patchId: string) {
    setBusyPatchId(patchId);
    setError("");
    try {
      await api.discardPatch(patchId);
      await onPatchesChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyPatchId("");
    }
  }

  async function revertToPatch(patchId: string) {
    setBusyPatchId(patchId);
    setError("");
    try {
      await api.revertToPatch(patchId);
      await onPatchesChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyPatchId("");
    }
  }

  async function savePatch(patchId: string) {
    const targetPath = exportDirectory.trim() ? buildExportPath(exportDirectory, patchId, "patch") : "";
    setSavingPatchId(patchId);
    setMessage("");
    setError("");
    try {
      if (!targetPath) {
        throw new Error("Set the default save directory in Settings first.");
      }
      const result = await api.exportPatch(patchId, { output_path: targetPath });
      setMessage(`Saved patch to ${result.output_path}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSavingPatchId("");
    }
  }

  async function loadPatch(patch: Patch) {
    setLoadingPatchId(patch.patch_id);
    setMessage("");
    setError("");
    try {
      await api.loadPatch(patch.patch_id);
      setSelectedPatch(null);
      setMessage(`Loaded ${patch.patch_id} into ${patch.base_model}.`);
      await onPatchesChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingPatchId("");
    }
  }

  async function deletePatch(patch: Patch) {
    setDeletingPatchId(patch.patch_id);
    setMessage("");
    setError("");
    try {
      await api.deletePatch(patch.patch_id);
      setPatchPendingDeletion(null);
      setSelectedPatch((current) => current?.patch_id === patch.patch_id ? null : current);
      setMessage(`Deleted saved patch ${patch.patch_id}.`);
      await onPatchesChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingPatchId("");
    }
  }

  async function clearHistory() {
    setClearingHistory(true);
    setMessage("");
    setError("");
    try {
      const result = await api.deleteAllPatches();
      setSelectedPatch(null);
      setConfirmingClear(false);
      setMessage(`Deleted ${result.deleted_patch_count} saved patch records.`);
      await onPatchesChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setClearingHistory(false);
    }
  }

  return (
    <section className="panel min-w-0 grid gap-6 overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h2 className="text-base font-semibold">Patch History</h2>
          <span className="label whitespace-nowrap">{activePatches.length} active / {patches.length} saved</span>
        </div>
        <button
          className="ghost-button min-h-8 px-2 py-1 text-xs"
          disabled={patches.length === 0 || clearingHistory}
          onClick={() => setConfirmingClear(true)}
          title="Delete all saved patch records and tensor deltas"
          type="button"
        >
          <Trash2 size={14} />
          Clear History
        </button>
      </div>

      <div className="min-w-0">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold">Active Model Lineage</h3>
          <span className="label whitespace-nowrap">current model state</span>
        </div>
        <PatchTable
          patches={activePatches}
          latestActiveIndexByModel={latestActiveIndexByModel}
          busyPatchId={busyPatchId}
          savingPatchId={savingPatchId}
          loadingPatchId={loadingPatchId}
          deletingPatchId={deletingPatchId}
          loadedModelIds={loadedModelIds}
          canSavePatches={Boolean(exportDirectory.trim())}
          onDiscard={discardPatch}
          onRevertTo={revertToPatch}
          onSavePatch={savePatch}
          onLoadPatch={loadPatch}
          onDeletePatch={setPatchPendingDeletion}
          onViewPatch={setSelectedPatch}
          mode="active"
        />
        {activePatches.length === 0 ? <div className="rounded-md border border-line bg-panel p-4 text-sm text-ink/70">No active patches. Current model state is clean.</div> : null}
      </div>

      <div className="min-w-0 border-t border-line pt-5">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold">Saved Patch Library</h3>
          <span className="label whitespace-nowrap">newest first</span>
        </div>
        <PatchTable
          patches={savedPatches}
          latestActiveIndexByModel={latestActiveIndexByModel}
          busyPatchId={busyPatchId}
          savingPatchId={savingPatchId}
          loadingPatchId={loadingPatchId}
          deletingPatchId={deletingPatchId}
          loadedModelIds={loadedModelIds}
          canSavePatches={Boolean(exportDirectory.trim())}
          onDiscard={discardPatch}
          onRevertTo={revertToPatch}
          onSavePatch={savePatch}
          onLoadPatch={loadPatch}
          onDeletePatch={setPatchPendingDeletion}
          onViewPatch={setSelectedPatch}
          mode="library"
          scrollable
        />
        {patches.length === 0 ? <div className="rounded-md border border-line bg-panel p-4 text-sm text-ink/70">No patches saved yet.</div> : null}
      </div>
      {message ? <div className="rounded-md border border-cyan/30 bg-cyan/10 p-3 text-sm text-cyan">{message}</div> : null}
      {error ? <div className="mt-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      {selectedPatch ? (
        <PatchDetailsDialog
          patch={selectedPatch}
          modelLoaded={runtime.some((status) => status.model_id === selectedPatch.base_model && status.loaded)}
          loading={loadingPatchId === selectedPatch.patch_id}
          onClose={() => setSelectedPatch(null)}
          onLoad={() => loadPatch(selectedPatch)}
        />
      ) : null}
      {confirmingClear ? (
        <ClearPatchHistoryDialog
          clearing={clearingHistory}
          patchCount={patches.length}
          onCancel={() => setConfirmingClear(false)}
          onConfirm={clearHistory}
        />
      ) : null}
      {patchPendingDeletion ? (
        <DeletePatchDialog
          deleting={deletingPatchId === patchPendingDeletion.patch_id}
          patch={patchPendingDeletion}
          onCancel={() => setPatchPendingDeletion(null)}
          onConfirm={() => deletePatch(patchPendingDeletion)}
        />
      ) : null}
    </section>
  );
}

function PatchTable({
  patches,
  latestActiveIndexByModel,
  busyPatchId,
  savingPatchId,
  loadingPatchId,
  deletingPatchId,
  loadedModelIds,
  canSavePatches,
  onDiscard,
  onRevertTo,
  onSavePatch,
  onLoadPatch,
  onDeletePatch,
  onViewPatch,
  mode,
  scrollable = false
}: {
  patches: Patch[];
  latestActiveIndexByModel: Record<string, number>;
  busyPatchId: string;
  savingPatchId: string;
  loadingPatchId: string;
  deletingPatchId: string;
  loadedModelIds: Set<string>;
  canSavePatches: boolean;
  onDiscard: (patchId: string) => void;
  onRevertTo: (patchId: string) => void;
  onSavePatch: (patchId: string) => void;
  onLoadPatch: (patch: Patch) => void;
  onDeletePatch: (patch: Patch) => void;
  onViewPatch: (patch: Patch) => void;
  mode: "active" | "library";
  scrollable?: boolean;
}) {
  if (patches.length === 0) return null;
  return (
    <div className={scrollable ? "max-h-[34rem] overflow-auto pb-1" : "overflow-x-auto pb-1"}>
      <table className="w-full table-fixed border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left">
            <th className="sticky top-0 z-10 w-[130px] bg-white py-2 pr-3">Patch</th>
            <th className="sticky top-0 z-10 w-[88px] bg-white py-2 pr-3">State</th>
            <th className="sticky top-0 z-10 w-[72px] bg-white py-2 pr-3">Method</th>
            <th className="sticky top-0 z-10 hidden w-[188px] bg-white py-2 pr-3 2xl:table-cell">Model</th>
            <th className="sticky top-0 z-10 hidden w-[64px] bg-white py-2 pr-3 lg:table-cell">Layers</th>
            <th className="sticky top-0 z-10 w-[48px] bg-white py-2 pr-3">Edits</th>
            <th className="sticky top-0 z-10 hidden w-[142px] bg-white py-2 pr-3 lg:table-cell">Created</th>
            <th className="sticky top-0 z-10 w-[124px] bg-white py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {patches.map((patch) => {
            const latestIndex = latestActiveIndexByModel[patch.base_model] ?? -1;
            const isLatestActive = Boolean(patch.active && patch.active_index === latestIndex);
            const isBakedLineage = Boolean(patch.lineage_only || patch.baked_into_model);
            const canRevertHere = Boolean(patch.active && !isLatestActive && !isBakedLineage);
            const canDiscardLatest = Boolean(isLatestActive && !isBakedLineage);
            const hasTensorDeltas = Boolean((patch.delta_paths ?? []).some((path) => path.endsWith(".safetensors")));
            const canLoad = Boolean(!patch.active && !isBakedLineage && hasTensorDeltas && loadedModelIds.has(patch.base_model));
            const actionLabel = canRevertHere ? "Revert Here" : "Discard";
            const actionTitle = canDiscardLatest
              ? "Discard the most recently loaded patch and step the model back"
              : patch.active
                ? canRevertHere
                  ? "Discard all loaded patches after this point"
                  : "This baked lineage is already the current model state"
                : "Saved patch artifact; not part of the current active model lineage";
            const loadTitle = isBakedLineage
              ? "This patch is already baked into an imported model"
              : !hasTensorDeltas
                ? "This patch has no saved tensor deltas to load"
                : loadedModelIds.has(patch.base_model)
                  ? `Load this patch into ${patch.base_model}`
                  : `Load ${patch.base_model} before loading this patch`;
            const onClick = canRevertHere ? onRevertTo : onDiscard;
            return (
              <tr key={`${mode}-${patch.patch_id}`} className="border-b border-line/70">
                <td className="py-3 pr-3 font-mono text-xs whitespace-nowrap">
                  <button className="text-left text-ink underline decoration-line underline-offset-4 transition hover:decoration-cyan hover:text-cyan" onClick={() => onViewPatch(patch)} title="View patch details" type="button">
                    {patch.patch_id}
                  </button>
                </td>
                <td className="py-3 pr-3">
                  {patch.active ? (
                    <span className="rounded border border-cyan/30 bg-cyan/10 px-2 py-1 text-xs font-medium text-cyan">
                      {isBakedLineage ? "Baked" : isLatestActive ? "Loaded" : `Loaded #${(patch.active_index ?? 0) + 1}`}
                    </span>
                  ) : (
                    <span className="rounded border border-line bg-panel px-2 py-1 text-xs text-ink/50">Saved</span>
                  )}
                </td>
                <td className="py-3 pr-3 whitespace-nowrap">{patch.method.toUpperCase()}</td>
                <td className="hidden py-3 pr-3 2xl:table-cell">
                  <span className="block truncate" title={patch.base_model}>{patch.base_model}</span>
                </td>
                <td className="hidden py-3 pr-3 lg:table-cell">{patch.modified_layers.join(", ") || "-"}</td>
                <td className="py-3 pr-3">{patch.edits.length}</td>
                <td className="hidden py-3 pr-3 font-mono text-xs whitespace-nowrap lg:table-cell">{formatPatchDate(patch.created_at)}</td>
                <td className="py-3 text-right">
                  <div className="flex justify-end gap-1">
                    <button
                      aria-label={`Export patch ${patch.patch_id}`}
                      className="ghost-button grid size-9 place-items-center p-0"
                      disabled={savingPatchId === patch.patch_id || !canSavePatches}
                      onClick={() => onSavePatch(patch.patch_id)}
                      title={canSavePatches ? "Export this patch metadata and tensor deltas" : "Set the default export directory in Settings first"}
                      type="button"
                    >
                      <Save className={savingPatchId === patch.patch_id ? "animate-pulse" : undefined} size={18} />
                    </button>
                    {patch.active ? (
                      <button
                        aria-label={`${actionLabel} patch ${patch.patch_id}`}
                        className="ghost-button grid size-9 place-items-center p-0"
                        disabled={(!canRevertHere && !canDiscardLatest) || busyPatchId === patch.patch_id}
                        onClick={() => onClick(patch.patch_id)}
                        title={actionTitle}
                        type="button"
                      >
                        <XCircle className={busyPatchId === patch.patch_id ? "animate-pulse" : undefined} size={18} />
                      </button>
                    ) : (
                      <>
                        <button
                          aria-label={`Load patch ${patch.patch_id} into model`}
                          className="ghost-button grid size-9 place-items-center p-0"
                          disabled={!canLoad || loadingPatchId === patch.patch_id}
                          onClick={() => onLoadPatch(patch)}
                          title={loadTitle}
                          type="button"
                        >
                          <Layers className={loadingPatchId === patch.patch_id ? "animate-pulse" : undefined} size={18} />
                        </button>
                        {mode === "library" ? (
                          <button
                            aria-label={`Delete saved patch ${patch.patch_id}`}
                            className="ghost-button grid size-9 place-items-center p-0 text-rust hover:text-rust"
                            disabled={deletingPatchId === patch.patch_id}
                            onClick={() => onDeletePatch(patch)}
                            title="Delete this saved patch and its tensor deltas"
                            type="button"
                          >
                            <Trash2 className={deletingPatchId === patch.patch_id ? "animate-pulse" : undefined} size={18} />
                          </button>
                        ) : null}
                      </>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function buildExportPath(directory: string, id: string, mode: "model" | "patch") {
  const base = directory.trim().replace(/\/+$/, "");
  const safeId = id.replace(/[^a-zA-Z0-9._-]/g, "_");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${base}/${safeId}-${mode}-${timestamp}`;
}

function formatPatchDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const pad = (number: number) => String(number).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function PatchDetailsDialog({
  patch,
  modelLoaded,
  loading,
  onClose,
  onLoad
}: {
  patch: Patch;
  modelLoaded: boolean;
  loading: boolean;
  onClose: () => void;
  onLoad: () => void;
}) {
  const tensorDeltas = (patch.delta_paths ?? []).filter((path) => path.endsWith(".safetensors"));
  const bakedIntoModel = Boolean(patch.lineage_only || patch.baked_into_model);
  const canLoad = !patch.active && !bakedIntoModel && tensorDeltas.length > 0 && modelLoaded;
  const loadTitle = patch.active
    ? "This patch is already part of the active model state"
    : bakedIntoModel
      ? "This imported lineage is already baked into its model weights"
      : tensorDeltas.length === 0
        ? "This patch does not contain saved tensor deltas"
        : modelLoaded
          ? "Load this saved patch into its base model"
          : `Load ${patch.base_model} before loading this patch`;

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink/30 p-4" onMouseDown={onClose} role="presentation">
      <section aria-labelledby="patch-details-title" aria-modal="true" className="max-h-[min(760px,calc(100vh-2rem))] w-full max-w-3xl overflow-y-auto rounded-lg border border-line bg-white p-5 shadow-xl" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <span className="label">Patch Details</span>
            <h2 id="patch-details-title" className="mt-1 break-all font-mono text-base font-semibold text-ink">{patch.patch_id}</h2>
          </div>
          <button className="ghost-button min-h-9 w-9 px-0" onClick={onClose} title="Close patch details" type="button">
            <X size={16} />
          </button>
        </div>

        <dl className="mt-5 grid gap-x-6 gap-y-4 border-y border-line py-4 text-sm sm:grid-cols-2">
          <DetailItem label="Method" value={patch.method.toUpperCase()} />
          <DetailItem label="Base Model" value={patch.base_model} />
          <DetailItem label="Created" value={formatPatchDate(patch.created_at)} />
          <DetailItem label="Modified Layers" value={patch.modified_layers.join(", ") || "Not recorded"} />
          <DetailItem label="Edit Mode" value={patch.sequential_edit ? "Sequential" : "Single edit"} />
          <DetailItem label="Second-Moment Adjustment" value={patch.mom2_adjustment === undefined ? "Not recorded" : patch.mom2_adjustment ? "Enabled" : "Disabled"} />
          <DetailItem label="Tensor Deltas" value={tensorDeltas.length ? `${tensorDeltas.length} saved file${tensorDeltas.length === 1 ? "" : "s"}` : "Not available"} />
          {patch.sequential_edit ? <DetailItem label="Prior Patches" value={patch.parent_patch_ids ? String(patch.parent_patch_ids.length) : "Not recorded"} /> : null}
          {patch.layer_source ? <DetailItem label="Layer Source" value={patch.layer_source.replace(/_/g, " ")} /> : null}
        </dl>

        <section className="mt-5">
          <h3 className="text-sm font-semibold">Edited Facts</h3>
          <div className="mt-2 divide-y divide-line border-y border-line">
            {patch.edits.map((edit, index) => (
              <article className="grid gap-3 py-4" key={`${patch.patch_id}-${index}`}>
                <div className="label">Edit {index + 1}</div>
                <DetailItem label="Prompt" value={buildEditPrompt(edit.subject, edit.relation_prompt)} mono />
                {edit.ground_truth ? <DetailItem label="Previous Value" value={edit.ground_truth} /> : null}
                <DetailItem label="New Value" value={edit.new_object} />
              </article>
            ))}
          </div>
        </section>

        <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
          <div className="text-xs text-ink/60">{loadTitle}</div>
          <button className="tool-button min-h-9 px-3 py-1" disabled={!canLoad || loading} onClick={onLoad} title={loadTitle} type="button">
            <Layers size={15} />
            {loading ? "Loading" : "Load into Model"}
          </button>
        </div>
      </section>
    </div>
  );
}

function ClearPatchHistoryDialog({
  patchCount,
  clearing,
  onCancel,
  onConfirm
}: {
  patchCount: number;
  clearing: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink/30 p-4" onMouseDown={onCancel} role="presentation">
      <section aria-labelledby="clear-patches-title" aria-modal="true" className="w-full max-w-md rounded-lg border border-line bg-white p-5 shadow-xl" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <span className="label">Destructive Action</span>
        <h2 id="clear-patches-title" className="mt-1 text-base font-semibold">Clear all patch history?</h2>
        <p className="mt-3 text-sm leading-6 text-ink/70">
          This reverts active workbench patches, then deletes {patchCount} saved patch record{patchCount === 1 ? "" : "s"} and their tensor delta files. Exported models and their embedded lineage are not changed.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <button className="ghost-button min-h-9 px-3 py-1" disabled={clearing} onClick={onCancel} type="button">Cancel</button>
          <button className="tool-button min-h-9 bg-rust px-3 py-1 hover:bg-rust/90" disabled={clearing} onClick={onConfirm} type="button">
            <Trash2 size={15} />
            {clearing ? "Clearing" : "Clear History"}
          </button>
        </div>
      </section>
    </div>
  );
}

function DeletePatchDialog({
  patch,
  deleting,
  onCancel,
  onConfirm
}: {
  patch: Patch;
  deleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink/30 p-4" onMouseDown={onCancel} role="presentation">
      <section aria-labelledby="delete-patch-title" aria-modal="true" className="w-full max-w-md rounded-lg border border-line bg-white p-5 shadow-xl" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <span className="label">Destructive Action</span>
        <h2 id="delete-patch-title" className="mt-1 text-base font-semibold">Delete saved patch?</h2>
        <p className="mt-3 text-sm leading-6 text-ink/70">
          This permanently deletes <span className="font-mono text-xs">{patch.patch_id}</span> and its saved tensor deltas. It cannot be undone.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <button className="ghost-button min-h-9 px-3 py-1" disabled={deleting} onClick={onCancel} type="button">Cancel</button>
          <button className="tool-button min-h-9 bg-rust px-3 py-1 hover:bg-rust/90" disabled={deleting} onClick={onConfirm} type="button">
            <Trash2 size={15} />
            {deleting ? "Deleting" : "Delete Patch"}
          </button>
        </div>
      </section>
    </div>
  );
}

function DetailItem({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid gap-1">
      <dt className="label">{label}</dt>
      <dd className={`${mono ? "font-mono text-xs" : "text-sm"} whitespace-pre-wrap break-words text-ink`}>{value}</dd>
    </div>
  );
}

function buildEditPrompt(subject: string, relationPrompt: string) {
  return relationPrompt.includes("{}") ? relationPrompt.replace(/\{\}/g, subject) : `${subject} ${relationPrompt}`.trim();
}
