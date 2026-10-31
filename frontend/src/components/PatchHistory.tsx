"use client";

import { Check, CheckSquare, GitBranch, Layers, Link2, Loader2, RotateCcw, Save, Search, Trash2, X, XCircle } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { experimentResource, historyResource, patchResource } from "@/lib/workbenchResources";
import { patchDeletionBlockReason, patchMutationBlockReason } from "@/lib/patchOperations";
import { api } from "@/lib/api";
import { formatLayerRanges } from "@/lib/layerRanges";
import type { ModelRuntimeStatus, Patch, PatchEditPage, PatchLibraryQuery, SupportedModel } from "@/lib/types";
import { usePatchLibrary } from "@/lib/usePatchLibrary";
import { buildLineageGraph, lineageDescription, type LineageGraphRow } from "@/lib/patchLineage";
import { usePersistentState } from "@/lib/usePersistentState";
import { usePatchDragSelection } from "@/lib/usePatchDragSelection";
import { PatchDeletionReview, PatchSelectionToolbar, usePatchLibrarySelection } from "./PatchLibrarySelection";
import type { PatchLibrarySelection } from "./PatchLibrarySelection";
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
  const [exportDirectory] = usePersistentState("kew:settings:modelExportPath", "");
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [methodFilter, setMethodFilter] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [sort, setSort] = useState<"newest" | "oldest">("newest");
  useEffect(() => {
    const timer = setTimeout(() => setSearchQuery(search.trim()), 250);
    return () => clearTimeout(timer);
  }, [search]);
  const libraryQuery = useMemo<PatchLibraryQuery>(() => ({ query: searchQuery, method: methodFilter, model_id: modelFilter, sort }), [searchQuery, methodFilter, modelFilter, sort]);
  const library = usePatchLibrary(libraryQuery, patches);
  const savedPatches = library.items;
  const selection = usePatchLibrarySelection(savedPatches, async (result) => {
    setSelectedPatch((current) => current && result.deleted_patch_ids.includes(current.patch_id) ? null : current);
    setMessage(`Deleted ${result.deleted_patch_count} saved patch record${result.deleted_patch_count === 1 ? "" : "s"}.${result.cleanup_pending ? " Some removed tensor files could not be cleaned up on disk." : ""}`);
    try { await onPatchesChanged?.(); } finally { library.retry(); }
  }, library.total, libraryQuery);
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

  return (
    <section className="panel min-w-0 grid gap-6 overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h2 className="text-base font-semibold">Patch History</h2>
          <span className="label whitespace-nowrap">{activePatches.length} active / {savedPatchCount} saved</span>
        </div>
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
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-sm font-semibold">Saved Patch Library</h3>
          <div className="flex items-center gap-3">
            <button className={`ghost-button min-h-9 px-3 py-1.5 text-xs ${selection.editing ? "border-cyan/25 bg-cyan/5 text-cyan shadow-none" : ""}`} aria-pressed={selection.editing} disabled={(!savedPatches.length && !selection.editing) || selection.deleting || Boolean(deletingPatchId || loadingPatchId)} onClick={() => { if (selection.editing) selection.finish(); else { setMessage(""); setError(""); selection.start(); } }} type="button">
              {selection.editing ? <Check size={15} aria-hidden="true" /> : <CheckSquare size={15} aria-hidden="true" />}{selection.editing ? "Done" : "Edit patch history"}
            </button>
          </div>
        </div>
        <fieldset disabled={selection.deleting || Boolean(selection.review) || selection.dragging} className="mb-3 grid min-w-0 gap-2 sm:grid-cols-2 xl:grid-cols-[minmax(200px,1fr)_minmax(120px,auto)_minmax(160px,auto)_minmax(140px,auto)]">
          <legend className="sr-only">Search and filter saved patches</legend>
          <div className="relative min-w-0">
            <Search size={15} aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink/40" />
            <input className="control w-full pl-9 pr-9 text-sm" type="search" aria-label="Search edit facts" placeholder="Search edit facts…" maxLength={500} value={search} onChange={(event) => setSearch(event.target.value)} />
            {search ? <button type="button" className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-ink/50 hover:text-ink" aria-label="Clear fact search" onClick={() => { setSearch(""); setSearchQuery(""); }}><X size={14} /></button> : null}
          </div>
          <select className="control min-w-0 text-sm" aria-label="Filter patches by method" value={methodFilter} onChange={(event) => setMethodFilter(event.target.value)}>
            <option value="">All methods</option>
            {[...new Set([...(library.methods ?? []), ...(methodFilter ? [methodFilter] : [])])].sort().map((method) => <option key={method} value={method}>{method.toUpperCase()}</option>)}
          </select>
          <select className="control min-w-0 max-w-full text-sm" aria-label="Filter patches by model" value={modelFilter} onChange={(event) => setModelFilter(event.target.value)}>
            <option value="">All models</option>
            {[...new Set([...(library.model_ids ?? []), ...(modelFilter ? [modelFilter] : [])])].map((id) => <option key={id} value={id}>{modelLabel(id)}</option>)}
          </select>
          <select className="control min-w-0 text-sm" aria-label="Sort patches by creation date" value={sort} onChange={(event) => setSort(event.target.value as "newest" | "oldest")}>
            <option value="newest">Newest first</option><option value="oldest">Oldest first</option>
          </select>
        </fieldset>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-ink/55">
          <span role="status">{library.loading || search.trim() !== searchQuery ? "Loading patches…" : `${library.total} ${searchQuery || methodFilter || modelFilter ? "matching" : "saved"} patch${library.total === 1 ? "" : "es"}`}</span>
          {search || methodFilter || modelFilter ? <button className="text-cyan hover:underline disabled:opacity-50" disabled={selection.deleting || Boolean(selection.review) || selection.dragging} type="button" onClick={() => { setSearch(""); setSearchQuery(""); setMethodFilter(""); setModelFilter(""); }}>Clear filters</button> : null}
        </div>
        <div className={selection.editing ? "overflow-hidden rounded-xl border border-line bg-white" : undefined}>
          {selection.editing ? <PatchSelectionToolbar selection={selection} patches={savedPatches} resultsPending={library.loading || search.trim() !== searchQuery} /> : null}
          <PatchTable
            key={JSON.stringify(libraryQuery)}
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
            selection={selection.editing ? selection : undefined}
            oldestFirst={sort === "oldest"}
            onLoadMore={library.loadMore}
            scrollable
          />
        </div>
        {!savedPatches.length && !library.loading && !library.error ? <div className="rounded-md border border-line bg-panel p-4 text-sm text-ink/70">{searchQuery || methodFilter || modelFilter ? "No patches match these filters." : "No patches saved yet."}</div> : null}
        {library.error ? <div className="inline-error mt-2" role="alert">{library.error} <button className="underline" type="button" onClick={library.retry}>Retry</button></div> : null}
        {library.loading ? <div className="flex justify-center p-3 text-cyan"><Loader2 className="animate-spin" size={17} aria-label="Loading patches" /></div> : library.hasMore ? <button className="ghost-button mt-3 min-h-8 px-3 py-1 text-xs" type="button" onClick={() => void library.loadMore()}>Load more patches</button> : null}
      </div>
      {message ? <div className="rounded-md border border-cyan/30 bg-cyan/10 p-3 text-sm text-cyan">{message}</div> : null}
      {error ? <div className="mt-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      {!selection.editing && selection.error ? <div role="alert" className="text-sm text-rust">{selection.error}</div> : null}
      {selection.review ? <PatchDeletionReview selection={selection} modelLabel={modelLabel} /> : null}
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
  selection,
  onLoadMore,
  oldestFirst = false,
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
  selection?: PatchLibrarySelection;
  onLoadMore?: () => Promise<void>;
  oldestFirst?: boolean;
  scrollable?: boolean;
}) {
  const blockedReason = usePatchMutationBlockReason();
  const scroller = useRef<HTMLDivElement>(null);
  const drag = usePatchDragSelection(scroller, {
    enabled: Boolean(selection && !selection.deleting && !selection.review),
    orderedIds: patches.map((patch) => patch.patch_id),
    selected: selection?.selected ?? new Set(), included: selection?.origins ?? new Map(),
    blocked: selection?.protectedIds ?? new Set(),
    onChange: (next) => selection?.setSelected(next), onDragging: (value) => selection?.setDragging(value),
  });
  const lineageGraph = useMemo(() => buildLineageGraph(patches), [patches]);
  if (patches.length === 0) return null;
  return (
    <div ref={scroller} onPointerDown={drag.onPointerDown} onPointerMove={drag.onPointerMove} onClickCapture={drag.onClickCapture} onScroll={(event) => { const node = event.currentTarget; if (scrollable && node.scrollHeight - node.scrollTop - node.clientHeight < 240) void onLoadMore?.(); }} className={`overflow-auto pb-1 [scrollbar-gutter:stable] ${scrollable ? "max-h-[34rem]" : ""} ${selection ? "select-none" : ""}`}>
      <table className="w-full min-w-[518px] table-fixed border-collapse text-sm lg:min-w-[724px] 2xl:min-w-[912px]">
        <colgroup>
          {selection ? <col className="w-12" /> : null}
          <col className="w-14" />
          <col className="w-[130px]" />
          <col className="w-[88px]" />
          <col className="w-[72px]" />
          <col className="hidden w-[188px] 2xl:table-column" />
          <col className="hidden w-[64px] lg:table-column" />
          <col className="w-[48px]" />
          <col className="hidden w-[142px] lg:table-column" />
          {!selection ? <col className="w-[124px]" /> : null}
        </colgroup>
        <thead>
          <tr className="border-b border-line text-left">
            {selection ? <th className="sticky top-0 z-10 bg-white py-2"><span className="sr-only">Select for deletion</span></th> : null}
            {lineageGraph ? (
              <th className="sticky top-0 z-10 bg-white py-2 text-left" title="Sequential edit lineage">
                <GitBranch className="ml-0.5 text-ink/60" size={16} />
              </th>
            ) : null}
            <th className="sticky top-0 z-10 bg-white py-2 pr-3">Patch</th>
            <th className="sticky top-0 z-10 bg-white py-2 pr-3">State</th>
            <th className="sticky top-0 z-10 bg-white py-2 pr-3">Method</th>
            <th className="sticky top-0 z-10 hidden bg-white py-2 pr-3 2xl:table-cell">Model</th>
            <th className="sticky top-0 z-10 hidden bg-white py-2 pr-3 lg:table-cell">Layers</th>
            <th className="sticky top-0 z-10 bg-white py-2 pr-3">Edits</th>
            <th className="sticky top-0 z-10 hidden bg-white py-2 pr-3 lg:table-cell">Created</th>
            {!selection ? <th className="sticky top-0 z-10 bg-white py-2">
              <div className="ml-auto w-[7.25rem] text-left">Actions</div>
            </th> : null}
          </tr>
        </thead>
        <tbody>
          {patches.map((patch) => {
            const deletionBlocked = patchDeletionBlockReason(patch.patch_id, patches);
            const explicitlySelected = selection?.selected.has(patch.patch_id) ?? false;
            const included = selection?.origins.has(patch.patch_id) ?? false;
            const automatic = included && !explicitlySelected;
            const includedBy = selection?.origins.get(patch.patch_id) ?? [];
            const selectionBlocked = Boolean(selection?.protectedIds.has(patch.patch_id));
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
              <tr key={`${mode}-${patch.patch_id}`} data-patch-id={patch.patch_id} className={`border-b border-line/70 ${selection ? `transition-colors last:border-b-0 ${included ? "bg-cyan/[0.06] hover:bg-cyan/10" : selectionBlocked ? "bg-panel/30" : "hover:bg-panel/70"}` : ""}`} onClick={(event) => { if (selection && !(event.target as Element).closest("button, a, input, label")) drag.toggle(patch.patch_id, event.shiftKey); }}>
                {selection ? <td className="px-1.5 py-2 text-center">
                  <label className="relative inline-flex size-9 items-center justify-center align-middle">
                    <input type="checkbox" className="peer patch-selection-checkbox" aria-label={`Select patch ${patch.patch_id} for deletion`} aria-describedby={`selection-${patch.patch_id}`} checked={included} disabled={automatic || selectionBlocked || selection.deleting} onChange={(event) => drag.toggle(patch.patch_id, (event.nativeEvent as MouseEvent).shiftKey)} />
                    <Check size={12} strokeWidth={3} className="pointer-events-none absolute text-white opacity-0 peer-checked:opacity-100 peer-disabled:text-cyan" aria-hidden="true" />
                  </label>
                </td> : null}
                {lineageGraph && graphRow ? (
                  <td
                    className="relative w-14 p-0 align-stretch"
                    title={lineageDescription(graphRow)}
                  >
                    <div className="h-14" />
                    <span className="sr-only">{lineageDescription(graphRow)}</span>
                    <PatchLineageGraph row={graphRow} laneCount={lineageGraph.laneCount} oldestFirst={oldestFirst} />
                  </td>
                ) : null}
                <td className="py-3 pr-3 font-mono text-xs whitespace-nowrap">
                  <div className={selection ? "flex min-h-10 flex-col justify-center" : undefined}>
                    <button className="text-left text-ink underline decoration-line underline-offset-4 transition hover:decoration-cyan hover:text-cyan" onClick={() => onViewPatch(patch)} title="View patch details" type="button">
                      {patch.patch_id}
                    </button>
                    {selection ? <span
                      id={`selection-${patch.patch_id}`}
                      className={automatic || selectionBlocked ? `mt-1 flex min-w-0 items-center gap-1 font-sans text-[11px] ${automatic ? "text-cyan/80" : "text-ink/45"}` : "sr-only"}
                      title={automatic ? `Included with ${includedBy.join(", ")}. Unselect ${includedBy.join(", ")} to keep.` : selectionBlocked ? "This patch or a dependent patch is active. Discard it before selecting." : undefined}
                    >
                      {automatic ? <Link2 size={11} className="shrink-0" aria-hidden="true" /> : null}
                      <span className="truncate">{automatic ? `Included with ${includedBy.join(", ")}` : selectionBlocked ? "Discard before selecting" : explicitlySelected ? "Selected by you" : "Keep"}</span>
                    </span> : null}
                  </div>
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
                <td className="hidden py-3 pr-3 lg:table-cell">{formatLayerRanges(patch.modified_layers) || "-"}</td>
                <td className="py-3 pr-3">{patch.edit_count ?? patch.edits.length}</td>
                <td className="hidden py-3 pr-3 font-mono text-xs whitespace-nowrap lg:table-cell">{formatPatchDate(patch.created_at)}</td>
                {!selection ? <td className="py-3 text-right">
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
                </td> : null}
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

function PatchLineageGraph({ row, laneCount, oldestFirst }: { row: LineageGraphRow; laneCount: number; oldestFirst: boolean }) {
  const width = Math.max(56, laneCount * 16 + 18);
  const laneX = (lane: number) => lane * 16 + 10;
  const nodeX = laneX(row.lane);
  return (
    <>
      <svg aria-hidden="true" className="pointer-events-none absolute inset-0 h-full w-14 text-cyan" preserveAspectRatio="none" viewBox={`0 0 ${width} 100`}>
        {row.incomingLanes.map((lane) => <path key={`in-${lane}`} stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.5" fill="none"
          d={row.incomingConnections.includes(lane) ? `M ${laneX(lane)} 0 C ${laneX(lane)} 25, ${nodeX} 25, ${nodeX} 50` : `M ${laneX(lane)} 0 V 50`} />)}
        {row.outgoingLanes.map((lane) => <path key={`out-${lane}`} stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.5" fill="none"
          d={row.outgoingConnections.includes(lane) ? `M ${nodeX} 50 C ${nodeX} 75, ${laneX(lane)} 75, ${laneX(lane)} 100` : `M ${laneX(lane)} 50 V 100`} />)}
        {[row.missingParentId ? (oldestFirst ? 18 : 82) : null, row.missingChildCount ? (oldestFirst ? 82 : 18) : null].filter((y): y is number => y !== null).map((y) => <g key={y}>
          <path d={`M ${nodeX} 50 L ${nodeX + 9} ${y}`} fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3 3" />
          <circle cx={nodeX + 9} cy={y} r="2.5" fill="white" stroke="currentColor" />
        </g>)}
      </svg>
      <span aria-hidden="true" className="pointer-events-none absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-cyan bg-white" style={{ left: `${nodeX / width * 56}px` }} />
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
      <section aria-labelledby="patch-details-title" aria-modal="true" className="patch-details-dialog flex h-[min(760px,calc(100dvh-2rem))] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-line bg-white p-4 shadow-xl sm:p-5" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <div className="flex shrink-0 items-start justify-between gap-4">
          <div className="min-w-0">
            <span className="label">Patch Details</span>
            <h2 id="patch-details-title" className="mt-1 break-all font-mono text-base font-semibold text-ink">{patch.patch_id}</h2>
          </div>
          <button className="ghost-button min-h-9 w-9 shrink-0 px-0" onClick={onClose} title="Close patch details" type="button">
            <X size={16} />
          </button>
        </div>

        <dl className="mt-3 grid shrink-0 grid-cols-2 gap-x-4 gap-y-2 border-y border-line py-3 text-sm sm:grid-cols-4">
          <DetailItem label="Method" value={patch.method.toUpperCase()} />
          <DetailItem label="Base Model" value={modelLabel} />
          <DetailItem label="Created" value={formatPatchDate(patch.created_at)} />
          <DetailItem label="Modified Layers" value={formatLayerRanges(patch.modified_layers) || "Not recorded"} />
          <DetailItem label="Edit Mode" value={patch.sequential_edit ? "Sequential" : "Single edit"} />
          <DetailItem label="Second-Moment Adjustment" value={patch.mom2_adjustment === undefined ? "Not recorded" : patch.mom2_adjustment ? "Enabled" : "Disabled"} />
          <DetailItem label="Tensor Deltas" value={tensorDeltas.length ? `${tensorDeltas.length} saved file${tensorDeltas.length === 1 ? "" : "s"}` : "Not available"} />
          <DetailItem label="Safetensors Size" value={formatTensorSize(patch.tensor_size_bytes)} />
          {patch.sequential_edit ? <DetailItem label="Prior Patches" value={patch.parent_patch_ids ? String(patch.parent_patch_ids.length) : "Not recorded"} /> : null}
          {patch.layer_source ? <DetailItem label="Layer Source" value={patch.layer_source.replace(/_/g, " ")} /> : null}
        </dl>

        <section className="mt-3 flex min-h-0 flex-1 flex-col">
          <div className="flex shrink-0 items-baseline justify-between gap-3">
            <h3 className="text-sm font-semibold">Edited Facts</h3>
            <span className="label whitespace-nowrap">{editCount}</span>
          </div>
          {editCount ? (
            <VirtualizedList
              key={patch.patch_id}
              className="mt-2 min-h-0 flex-1 overflow-y-auto overscroll-contain border-y border-line"
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
          ) : <p className="mt-2 text-sm text-ink/60">No edit facts were saved with this patch.</p>}
          {editPages.error ? <p className="mt-2 shrink-0 text-xs text-rust" role="alert">{editPages.error}</p> : null}
        </section>

        <footer className="mt-3 flex shrink-0 items-center justify-between gap-3 border-t border-line pt-3">
          <div className="min-w-0 flex-1 break-words text-xs text-ink/60">{blockedReason || loadTitle}</div>
          <button className="tool-button min-h-9 shrink-0 whitespace-nowrap px-3 py-1" disabled={Boolean(blockedReason) || !canLoad || loading} onClick={onLoad} title={blockedReason || loadTitle} type="button">
            <Layers size={15} />
            {loading ? "Loading" : "Load into Model"}
          </button>
        </footer>
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
    <div className="grid min-w-0 gap-1">
      <dt className="label">{label}</dt>
      <dd className={`${mono ? "font-mono text-xs" : "text-sm"} whitespace-pre-wrap break-words text-ink`}>{value}</dd>
    </div>
  );
}

function formatTensorSize(bytes: number | null | undefined) {
  if (bytes == null) return "Not available";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function buildEditPrompt(subject: string, relationPrompt: string) {
  return relationPrompt.includes("{}") ? relationPrompt.replace(/\{\}/g, subject) : `${subject} ${relationPrompt}`.trim();
}
