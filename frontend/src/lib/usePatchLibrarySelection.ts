"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { patchSelectionOrigins } from "./patchSelection";
import { protectedPatchIds } from "./patchOperations";
import type { Patch, PatchBatchDeletion, PatchDeletionPreview, PatchLibraryQuery } from "./types";
import { patchLibraryParams } from "./patchLibraryQuery";

export function usePatchLibrarySelection(patches: Patch[], onDeleted: (result: PatchBatchDeletion) => Promise<void>, totalPatchCount?: number, query: PatchLibraryQuery = {}) {
  const scope = patchLibraryParams(query);
  const [editing, setEditing] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [dragging, setDragging] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [selectingAll, setSelectingAll] = useState(false);
  const [selectAllSkipped, setSelectAllSkipped] = useState(0);
  const [selectAllFailed, setSelectAllFailed] = useState(false);
  const [selectableSnapshot, setSelectableSnapshot] = useState<{ scope: string; ids: string[] } | null>(null);
  const selectAllRequest = useRef<AbortController | null>(null);
  const [failure, setFailure] = useState("");
  const [revision, setRevision] = useState(0);
  const [review, setReview] = useState<PatchDeletionPreview | null>(null);
  const [response, setResponse] = useState<{
    key: string; patches: Patch[]; plan?: PatchDeletionPreview; error?: string;
  } | null>(null);
  const key = JSON.stringify([...selected].sort());
  const localOrigins = useMemo(() => patchSelectionOrigins(patches, selected), [patches, selected]);
  const protectedIds = useMemo(() => protectedPatchIds(patches), [patches]);
  const current = response?.key === key && response.patches === patches ? response : null;
  const plan = !dragging && !selectingAll ? current?.plan : undefined;
  const checking = selectingAll || (selected.size > 0 && (!current || dragging));
  const origins = useMemo(() => plan
    ? new Map(plan.patches.map((patch) => [patch.patch_id, patch.included_by]))
    : new Map([...localOrigins].map(([id, seeds]) => [id, seeds.filter((seed) => seed !== id)])), [plan, localOrigins]);
  const allSelectableIds = useMemo(() => {
    // Never mistake selecting the loaded page for selecting the whole library.
    if (selectableSnapshot?.scope === scope) return selectableSnapshot.ids;
    if (totalPatchCount === undefined || patches.length !== totalPatchCount) return null;
    return patches.filter((patch) => !protectedIds.has(patch.patch_id)).map((patch) => patch.patch_id);
  }, [selectableSnapshot, scope, patches, totalPatchCount, protectedIds]);
  const allSelected = Boolean(allSelectableIds?.length && allSelectableIds.every((id) => origins.has(id)));

  useEffect(() => {
    if (!editing || key === "[]" || dragging || deleting || selectingAll) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      api.previewPatchDeletion(JSON.parse(key) as string[], controller.signal).then(
        (next) => { if (!controller.signal.aborted) setResponse({ key, patches, plan: next }); },
        (error: unknown) => {
          if (!controller.signal.aborted) setResponse({ key, patches, error: error instanceof Error ? error.message : String(error) });
        }
      );
    }, 180);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [key, editing, dragging, deleting, selectingAll, patches, revision]);

  useEffect(() => () => selectAllRequest.current?.abort(), []);

  // A response for a previous filter must never select the new filter's rows.
  useEffect(() => {
    cancelSelectAll(); setSelectableSnapshot(null); setSelectAllSkipped(0); setSelectAllFailed(false);
  }, [scope]);

  function cancelSelectAll() {
    selectAllRequest.current?.abort();
    selectAllRequest.current = null;
    setSelectingAll(false);
  }

  async function selectAll() {
    if (!editing || deleting || dragging || review || selectAllRequest.current) return;
    const controller = new AbortController();
    selectAllRequest.current = controller;
    setSelectingAll(true); setSelectAllSkipped(0); setSelectAllFailed(false); setFailure(""); setResponse(null);
    try {
      // The server protects prerequisites even when active descendants are filtered out.
      const result = await api.patchSelection(query, controller.signal);
      if (controller.signal.aborted) return;
      setSelectableSnapshot({ scope, ids: result.patch_ids });
      const next = new Set([...selected, ...result.patch_ids]);
      setSelected(next);
      setSelectAllSkipped(result.blocked_patch_count);
      setResponse(null);
      setRevision((value) => value + 1);
    } catch (error) {
      if (!controller.signal.aborted) {
        setFailure(error instanceof Error ? error.message : String(error));
        setSelectAllFailed(true);
      }
    } finally {
      if (selectAllRequest.current === controller) {
        selectAllRequest.current = null;
        setSelectingAll(false);
      }
    }
  }

  function changeSelection(next: Set<string>) {
    cancelSelectAll();
    setSelectAllSkipped(0); setSelectAllFailed(false); setFailure("");
    if (next.size === selected.size && [...next].every((id) => selected.has(id))) return;
    setSelected(next);
    setResponse(null);
    setReview(null);
  }

  function finish() {
    if (deleting) return;
    cancelSelectAll(); setSelectAllSkipped(0); setSelectAllFailed(false);
    setEditing(false); setSelected(new Set()); setReview(null); setFailure(""); setResponse(null); setSelectableSnapshot(null);
  }

  function toggleAll() {
    if (!editing || deleting || dragging || selectingAll || review) return;
    if (allSelected) changeSelection(new Set());
    else return selectAll();
  }

  async function commit() {
    if (!review || !plan || checking || deleting || review.preview_token !== plan.preview_token || plan.blocked_patch_ids.length) return;
    setDeleting(true); setFailure(""); setSelectAllFailed(false);
    let result: PatchBatchDeletion;
    try {
      result = await api.deletePatchBatch(review.selected_patch_ids, review.preview_token);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
      setReview(null); setResponse(null); setRevision((value) => value + 1); setDeleting(false);
      return;
    }
    setSelected(new Set()); setReview(null); setResponse(null); setEditing(false); setSelectableSnapshot(null);
    try { await onDeleted(result); }
    catch (error) { setFailure(`Patches were deleted, but history could not refresh: ${error instanceof Error ? error.message : String(error)}`); }
    finally { setDeleting(false); }
  }

  return {
    editing, start: () => { setEditing(true); setFailure(""); setResponse(null); setSelectAllSkipped(0); setSelectAllFailed(false); }, finish,
    selected, setSelected: changeSelection, dragging, setDragging, deleting,
    selectingAll, selectAll, selectAllSkipped, allSelected, toggleAll,
    plan, checking, origins, protectedIds, review, setReview, commit,
    error: failure || current?.error || "",
    retry: () => {
      if (selectAllFailed) return selectAll();
      setResponse(null); setRevision((value) => value + 1);
    },
  };
}

export type PatchLibrarySelection = ReturnType<typeof usePatchLibrarySelection>;
