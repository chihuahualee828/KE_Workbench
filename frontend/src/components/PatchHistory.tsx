"use client";

import { GitBranch, Layers, RotateCcw, Save, Trash2, X, XCircle } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { experimentResource, historyResource, loadMorePatches, patchResource } from "@/lib/workbenchResources";
import { patchDeletionBlockReason, patchMutationBlockReason } from "@/lib/patchOperations";
import { api } from "@/lib/api";
import type { ModelRuntimeStatus, Patch, PatchEditPage, SupportedModel } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";
import { VirtualizedList } from "./VirtualizedList";

const PATCH_EDIT_PAGE_SIZE = 50;
const PATCH_FACT_CARD_ESTIMATED_HEIGHT = 172;

function usePatchMutationBlockReason() {
  const [experiments] = experimentResource.use();
  return patchMutationBlockReason(experiments);
}

export function PatchHistory({
  patches,
  models = [],
  runtime = [],
  onPatchesChanged
}: {
  patches: Patch[];
  models?: SupportedModel[];
  runtime?: ModelRuntimeStatus[];
  onPatchesChanged?: () => void | Promise<void>;
}) {
  const [totals] = historyResource.use();
  const blockedReason = usePatchMutationBlockReason();
  const savedPatchCount = Math.max(totals.patches, patches.length);
  const [busyPatchId, setBusyPatchId] = useState("");
  const [savingPatchId, setSavingPatchId] = useState("");
  const [loadingPatchId, setLoadingPatchId] = useState("");
  const [loadedPatchModelId, setLoadedPatchModelId] = useState("");
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
          return (right.active_index ?? 0) - (left.active_index ?? 0);
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
  const modelLabels = useMemo(
    () => new Map(models.map((model) => [model.id, patchModelLabel(model, models)])),
    [models]
  );
  const modelLabel = (modelId: string) => modelLabels.get(modelId) ?? modelId;

  useEffect(() => {
    if (!loadedPatchModelId || loadedModelIds.has(loadedPatchModelId)) return;
    setLoadedPatchModelId("");
    setMessage("");
  }, [loadedModelIds, loadedPatchModelId]);

  async function discardPatch(patchId: string) {
    if (blockedReason) { setError(blockedReason); return; }
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
    if (blockedReason) { setError(blockedReason); return; }
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
    if (blockedReason) { setError(blockedReason); return; }
    setLoadingPatchId(patch.patch_id);
    setMessage("");
    setError("");
    try {
      await api.loadPatch(patch.patch_id);
      await onPatchesChanged?.();
      setSelectedPatch(null);
      setLoadedPatchModelId(patch.base_model);
      setMessage(`Loaded ${patch.patch_id} into ${patch.base_model}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingPatchId("");
    }
  }

  async function deletePatch(patch: Patch) {
    const deletionBlocked = patchDeletionBlockReason(patch.patch_id, patches);
    if (deletionBlocked) { setError(deletionBlocked); return; }
    setDeletingPatchId(patch.patch_id);
    setMessage("");
    setError("");
    try {
      const result = await api.deletePatch(patch.patch_id);
      setPatchPendingDeletion(null);
      setSelectedPatch((current) => current && result.deleted_patch_ids.includes(current.patch_id) ? null : current);
      setMessage(`Deleted ${result.deleted_patch_count} saved patch record${result.deleted_patch_count === 1 ? "" : "s"}.`);
      await onPatchesChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingPatchId("");
    }
  }

  async function clearHistory() {
    if (blockedReason) { setError(blockedReason); return; }
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
          <span className="label whitespace-nowrap">{activePatches.length} active / {savedPatchCount} saved</span>
        </div>
        <button
          className="ghost-button min-h-8 px-2 py-1 text-xs"
          disabled={Boolean(blockedReason) || patches.length === 0 || clearingHistory}
          onClick={() => setConfirmingClear(true)}
          title={blockedReason || "Delete all saved patch records and tensor deltas"}
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
          modelLabel={modelLabel}
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
          modelLabel={modelLabel}
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
          modelLabel={modelLabel(selectedPatch.base_model)}
          modelLoaded={runtime.some((status) => status.model_id === selectedPatch.base_model && status.loaded)}
          loading={loadingPatchId === selectedPatch.patch_id}
          onClose={() => setSelectedPatch(null)}
          onLoad={() => loadPatch(selectedPatch)}
        />
      ) : null}
      {confirmingClear ? (
        <ClearPatchHistoryDialog
          clearing={clearingHistory}
          patchCount={savedPatchCount}
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
  modelLabel,
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
  modelLabel: (modelId: string) => string;
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
  const blockedReason = usePatchMutationBlockReason();
  if (patches.length === 0) return null;
  const lineageGraph = mode === "library" ? buildLineageGraph(patches) : null;
  return (
    <div onScroll={(event) => { const node = event.currentTarget; if (scrollable && node.scrollHeight - node.scrollTop - node.clientHeight < 240) void loadMorePatches().catch(() => undefined); }} className={scrollable ? "max-h-[34rem] overflow-auto pb-1" : "overflow-x-auto pb-1"}>
      <table className="w-full table-fixed border-collapse text-sm">
        <colgroup>
          {lineageGraph ? <col className="w-14" /> : null}
        </colgroup>
        <thead>
          <tr className="border-b border-line text-left">
            {lineageGraph ? (
              <th className="sticky top-0 z-10 w-14 bg-white py-2 text-left" title="Sequential edit lineage">
                <GitBranch className="ml-0.5 text-ink/60" size={16} />
              </th>
            ) : null}
            <th className="sticky top-0 z-10 w-[130px] bg-white py-2 pr-3">Patch</th>
            <th className="sticky top-0 z-10 w-[88px] bg-white py-2 pr-3">State</th>
            <th className="sticky top-0 z-10 w-[72px] bg-white py-2 pr-3">Method</th>
            <th className="sticky top-0 z-10 hidden w-[188px] bg-white py-2 pr-3 2xl:table-cell">Model</th>
            <th className="sticky top-0 z-10 hidden w-[64px] bg-white py-2 pr-3 lg:table-cell">Layers</th>
            <th className="sticky top-0 z-10 w-[48px] bg-white py-2 pr-3">Edits</th>
            <th className="sticky top-0 z-10 hidden w-[142px] bg-white py-2 pr-3 lg:table-cell">Created</th>
            <th className="sticky top-0 z-10 w-[124px] bg-white py-2">
              <div className="ml-auto w-[7.25rem] text-left">Actions</div>
            </th>
          </tr>
        </thead>
        <tbody>
          {patches.map((patch) => {
            const deletionBlocked = patchDeletionBlockReason(patch.patch_id, patches);
            const graphRow = lineageGraph?.rowsByPatchId[patch.patch_id];
            const latestIndex = latestActiveIndexByModel[patch.base_model] ?? -1;
            const isLatestActive = Boolean(patch.active && patch.active_index === latestIndex);
            const isBakedLineage = Boolean(patch.lineage_only || patch.baked_into_model);
            const canRevertHere = Boolean(patch.active && !isLatestActive && !isBakedLineage);
            const canDiscard = Boolean(patch.active && !isBakedLineage);
            const hasTensorDeltas = Boolean((patch.delta_paths ?? []).some((path) => path.endsWith(".safetensors")));
            const canLoad = Boolean(!patch.active && !isBakedLineage && hasTensorDeltas && loadedModelIds.has(patch.base_model));
            const actionTitle = patch.active
              ? canDiscard
                ? "Discard this patch and every dependent patch applied after it"
                : "This baked lineage is already the current model state"
              : "Saved patch artifact; not part of the current active model lineage";
            const loadTitle = isBakedLineage
              ? "This patch is already baked into an imported model"
              : !hasTensorDeltas
                ? "This patch has no saved tensor deltas to load"
                : loadedModelIds.has(patch.base_model)
                  ? `Load this patch into ${patch.base_model}`
                  : `Load ${patch.base_model} before loading this patch`;
            return (
              <tr key={`${mode}-${patch.patch_id}`} className="border-b border-line/70">
                {lineageGraph && graphRow ? (
                  <td
                    className="relative w-14 p-0 align-stretch"
                    title={graphRow.parentPatchId ? `Sequential edit from ${graphRow.parentPatchId}` : "Independent edit root"}
                  >
                    <div className="h-14" />
                    <PatchLineageGraph row={graphRow} laneCount={lineageGraph.laneCount} />
                  </td>
                ) : null}
                <td className="py-3 pr-3 font-mono text-xs whitespace-nowrap">
                  <button className="text-left text-ink underline decoration-line underline-offset-4 transition hover:decoration-cyan hover:text-cyan" onClick={() => onViewPatch(patch)} title="View patch details" type="button">
                    {patch.patch_id}
                  </button>
                </td>
                <td className="py-3 pr-3">
                  {patch.active ? (
                    <span className="rounded border border-cyan/30 bg-cyan/10 px-2 py-1 text-xs font-medium text-cyan">
                      {isBakedLineage ? "Baked" : "Loaded"}
                    </span>
                  ) : (
                    <span className="rounded border border-line bg-panel px-2 py-1 text-xs text-ink/50">Saved</span>
                  )}
                </td>
                <td className="py-3 pr-3 whitespace-nowrap">{patch.method.toUpperCase()}</td>
                <td className="hidden py-3 pr-3 2xl:table-cell">
                  <span className="block truncate" title={modelLabel(patch.base_model)}>{modelLabel(patch.base_model)}</span>
                </td>
                <td className="hidden py-3 pr-3 lg:table-cell">{patch.modified_layers.join(", ") || "-"}</td>
                <td className="py-3 pr-3">{patch.edit_count ?? patch.edits.length}</td>
                <td className="hidden py-3 pr-3 font-mono text-xs whitespace-nowrap lg:table-cell">{formatPatchDate(patch.created_at)}</td>
                <td className="py-3 text-right">
                  <div className="ml-auto flex w-[7.25rem] justify-end gap-1">
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
                      <>
                        {canRevertHere ? (
                          <button
                            aria-label={`Revert to patch ${patch.patch_id}`}
                            className="ghost-button grid size-9 place-items-center p-0"
                            disabled={Boolean(blockedReason) || busyPatchId === patch.patch_id}
                            onClick={() => onRevertTo(patch.patch_id)}
                            title={blockedReason || "Keep this patch and discard every patch applied after it"}
                            type="button"
                          >
                            <RotateCcw className={busyPatchId === patch.patch_id ? "animate-pulse" : undefined} size={18} />
                          </button>
                        ) : null}
                        <button
                          aria-label={`Discard patch ${patch.patch_id}`}
                          className="ghost-button grid size-9 place-items-center p-0"
                          disabled={Boolean(blockedReason) || !canDiscard || busyPatchId === patch.patch_id}
                          onClick={() => onDiscard(patch.patch_id)}
                          title={blockedReason || actionTitle}
                          type="button"
                        >
                          <XCircle className={busyPatchId === patch.patch_id ? "animate-pulse" : undefined} size={18} />
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          aria-label={`Load patch ${patch.patch_id} into model`}
                          className="ghost-button grid size-9 place-items-center p-0"
                          disabled={Boolean(blockedReason) || !canLoad || loadingPatchId === patch.patch_id}
                          onClick={() => onLoadPatch(patch)}
                          title={blockedReason || loadTitle}
                          type="button"
                        >
                          <Layers className={loadingPatchId === patch.patch_id ? "animate-pulse" : undefined} size={18} />
                        </button>
                        {mode === "library" ? (
                          <button
                            aria-label={`Delete saved patch ${patch.patch_id}`}
                            className="ghost-button grid size-9 place-items-center p-0 text-rust hover:text-rust"
                            disabled={Boolean(deletionBlocked) || deletingPatchId === patch.patch_id}
                            onClick={() => onDeletePatch(patch)}
                            title={deletionBlocked || "Delete this saved patch and its tensor deltas"}
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

function patchModelLabel(model: SupportedModel, models: SupportedModel[]) {
  if (model.source !== "local") return model.name;

  const defaultName = models.find(
    (candidate) => candidate.source !== "local" && candidate.id === model.base_model_id
  )?.name;
  if (!defaultName || model.name !== defaultName) return model.name;
  return `${defaultName} (${model.id})`;
}

type LineageGraphRow = {
  lane: number;
  parentLane: number | null;
  parentPatchId: string | null;
  incomingLanes: number[];
  outgoingLanes: number[];
};

function buildLineageGraph(patches: Patch[]): { laneCount: number; rowsByPatchId: Record<string, LineageGraphRow> } {
  const patchIds = new Set(patches.map((patch) => patch.patch_id));
  const lanes: Array<string | null> = [];
  const rowsByPatchId: Record<string, LineageGraphRow> = {};
  let laneCount = 1;

  for (const patch of patches) {
    const directParentId = patch.sequential_edit ? patch.parent_patch_ids?.at(-1) ?? null : null;
    const parentPatchId = directParentId && patchIds.has(directParentId) ? directParentId : null;
    let lane = lanes.indexOf(patch.patch_id);
    const incomingLanes = lanes
      .map((candidate, index) => candidate ? index : -1)
      .filter((index) => index >= 0);
    if (lane < 0) {
      lane = lanes.findIndex((candidate) => candidate === null);
      if (lane < 0) lane = lanes.length;
      lanes[lane] = patch.patch_id;
    }

    let parentLane: number | null = null;

    if (parentPatchId) {
      parentLane = lanes.indexOf(parentPatchId);
      if (parentLane < 0) {
        parentLane = lane;
        lanes[lane] = parentPatchId;
      } else if (parentLane !== lane) {
        lanes[lane] = null;
      }
    } else {
      lanes[lane] = null;
    }

    const outgoingLanes = lanes
      .map((candidate, index) => candidate ? index : -1)
      .filter((index) => index >= 0);
    laneCount = Math.max(laneCount, lanes.length);
    rowsByPatchId[patch.patch_id] = {
      lane,
      parentLane,
      parentPatchId,
      incomingLanes,
      outgoingLanes
    };
  }

  return { laneCount, rowsByPatchId };
}

function PatchLineageGraph({ row, laneCount }: { row: LineageGraphRow; laneCount: number }) {
  const laneWidth = 16;
  const width = Math.max(56, laneCount * laneWidth + 8);
  const laneX = (lane: number) => lane * laneWidth + 10;
  return (
    <>
      <svg aria-hidden="true" className="pointer-events-none absolute inset-0 h-full w-14 text-cyan" preserveAspectRatio="none" viewBox={`0 0 ${width} 100`}>
        {row.incomingLanes.map((lane) => (
          <line key={`in-${lane}`} stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.5" x1={laneX(lane)} x2={laneX(lane)} y1="0" y2="50" />
        ))}
        {row.outgoingLanes.map((lane) => (
          <line key={`out-${lane}`} stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.5" x1={laneX(lane)} x2={laneX(lane)} y1="50" y2="100" />
        ))}
        {row.parentLane !== null && row.parentLane !== row.lane ? (
          <path
            d={`M ${laneX(row.lane)} 50 C ${laneX(row.lane)} 76, ${laneX(row.parentLane)} 76, ${laneX(row.parentLane)} 100`}
            fill="none"
            stroke="currentColor"
            strokeOpacity="0.55"
            strokeWidth="1.5"
          />
        ) : null}
      </svg>
      <span
        aria-hidden="true"
        className="pointer-events-none absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-cyan bg-white"
        style={{ left: `${laneX(row.lane)}px` }}
      />
    </>
  );
}

function PatchDetailsDialog({
  patch,
  modelLabel,
  modelLoaded,
  loading,
  onClose,
  onLoad
}: {
  patch: Patch;
  modelLabel: string;
  modelLoaded: boolean;
  loading: boolean;
  onClose: () => void;
  onLoad: () => void;
}) {
  const blockedReason = usePatchMutationBlockReason();
  const tensorDeltas = (patch.delta_paths ?? []).filter((path) => path.endsWith(".safetensors"));
  const bakedIntoModel = Boolean(patch.lineage_only || patch.baked_into_model);
  const canLoad = !patch.active && !bakedIntoModel && tensorDeltas.length > 0 && modelLoaded;
  const editCount = patch.edit_count ?? patch.edits.length;
  const editPages = usePatchEditPages(patch.patch_id, editCount);
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
          <DetailItem label="Base Model" value={modelLabel} />
          <DetailItem label="Created" value={formatPatchDate(patch.created_at)} />
          <DetailItem label="Modified Layers" value={patch.modified_layers.join(", ") || "Not recorded"} />
          <DetailItem label="Edit Mode" value={patch.sequential_edit ? "Sequential" : "Single edit"} />
          <DetailItem label="Second-Moment Adjustment" value={patch.mom2_adjustment === undefined ? "Not recorded" : patch.mom2_adjustment ? "Enabled" : "Disabled"} />
          <DetailItem label="Tensor Deltas" value={tensorDeltas.length ? `${tensorDeltas.length} saved file${tensorDeltas.length === 1 ? "" : "s"}` : "Not available"} />
          {patch.sequential_edit ? <DetailItem label="Prior Patches" value={patch.parent_patch_ids ? String(patch.parent_patch_ids.length) : "Not recorded"} /> : null}
          {patch.layer_source ? <DetailItem label="Layer Source" value={patch.layer_source.replace(/_/g, " ")} /> : null}
        </dl>

        <section className="mt-5">
          <div className="flex items-baseline justify-between gap-3">
            <h3 className="text-sm font-semibold">Edited Facts</h3>
            <span className="label whitespace-nowrap">{editCount}</span>
          </div>
          {editCount ? (
            <VirtualizedList
              key={patch.patch_id}
              className="mt-2 max-h-[min(420px,45vh)] overflow-y-auto border-y border-line"
              estimatedItemHeight={PATCH_FACT_CARD_ESTIMATED_HEIGHT}
              getItem={editPages.getItem}
              itemCount={editCount}
              onVisibleRangeChange={editPages.ensureRange}
              renderItem={(edit, index) => (
                <article className="grid gap-3 border-b border-line py-4">
                  <div className="label">Edit {index + 1}</div>
                  <DetailItem label="Prompt" value={buildEditPrompt(edit.subject, edit.relation_prompt)} mono />
                  {edit.ground_truth ? <DetailItem label="Previous Value" value={edit.ground_truth} /> : null}
                  <DetailItem label="New Value" value={edit.new_object} />
                </article>
              )}
            />
          ) : <div className="mt-2 rounded-md border border-line bg-panel p-3 text-sm text-ink/60">No edit facts were saved with this patch.</div>}
          {editPages.error ? <p className="mt-2 text-xs text-rust" role="alert">{editPages.error}</p> : null}
        </section>

        <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
          <div className="text-xs text-ink/60">{blockedReason || loadTitle}</div>
          <button className="tool-button min-h-9 px-3 py-1" disabled={Boolean(blockedReason) || !canLoad || loading} onClick={onLoad} title={blockedReason || loadTitle} type="button">
            <Layers size={15} />
            {loading ? "Loading" : "Load into Model"}
          </button>
        </div>
      </section>
    </div>
  );
}

function usePatchEditPages(patchId: string, total: number) {
  const [pages, setPages] = useState<Record<number, PatchEditPage>>({});
  const [error, setError] = useState("");
  const requestKeyRef = useRef(patchId);
  const pending = useRef(new Set<string>());

  const loadPage = useCallback(async (offset: number, requestKey = patchId) => {
    if (!patchId || offset >= total) return;
    const pageKey = `${requestKey}:${offset}`;
    if (pending.current.has(pageKey)) return;
    pending.current.add(pageKey);
    try {
      const page = await api.patchEdits(patchId, offset, PATCH_EDIT_PAGE_SIZE);
      if (requestKeyRef.current !== requestKey) return;
      setPages((current) => ({ ...current, [offset]: page }));
      setError("");
    } catch (err) {
      if (requestKeyRef.current === requestKey) {
        setError(err instanceof Error ? err.message : "Could not load this patch's edit facts.");
      }
    } finally {
      pending.current.delete(pageKey);
    }
  }, [patchId, total]);

  useEffect(() => {
    requestKeyRef.current = patchId;
    setPages({});
    setError("");
    if (patchId && total) void loadPage(0, patchId);
  }, [loadPage, patchId, total]);

  const ensureRange = useCallback((start: number, end: number) => {
    if (!patchId || end < start) return;
    const firstOffset = Math.floor(start / PATCH_EDIT_PAGE_SIZE) * PATCH_EDIT_PAGE_SIZE;
    for (let offset = firstOffset; offset <= end && offset < total; offset += PATCH_EDIT_PAGE_SIZE) {
      if (!pages[offset]) void loadPage(offset, patchId);
    }
  }, [loadPage, pages, patchId, total]);

  const getItem = useCallback(
    (index: number) => pages[Math.floor(index / PATCH_EDIT_PAGE_SIZE) * PATCH_EDIT_PAGE_SIZE]?.items[index % PATCH_EDIT_PAGE_SIZE],
    [pages],
  );

  return { ensureRange, error, getItem };
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
  const blockedReason = usePatchMutationBlockReason();
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
          <button className="tool-button min-h-9 bg-rust px-3 py-1 hover:bg-rust/90" disabled={Boolean(blockedReason) || clearing} onClick={onConfirm} title={blockedReason || undefined} type="button">
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
  const [patches] = patchResource.use();
  const blockedReason = patchDeletionBlockReason(patch.patch_id, patches);
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink/30 p-4" onMouseDown={onCancel} role="presentation">
      <section aria-labelledby="delete-patch-title" aria-modal="true" className="w-full max-w-md rounded-lg border border-line bg-white p-5 shadow-xl" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <span className="label">Destructive Action</span>
        <h2 id="delete-patch-title" className="mt-1 text-base font-semibold">Delete saved patch?</h2>
        <p className="mt-3 text-sm leading-6 text-ink/70">
          This permanently deletes <span className="font-mono text-xs">{patch.patch_id}</span>, every dependent patch, and their saved tensor deltas. It cannot be undone.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <button className="ghost-button min-h-9 px-3 py-1" disabled={deleting} onClick={onCancel} type="button">Cancel</button>
          <button className="tool-button min-h-9 bg-rust px-3 py-1 hover:bg-rust/90" disabled={Boolean(blockedReason) || deleting} onClick={onConfirm} title={blockedReason || undefined} type="button">
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
