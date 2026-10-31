"use client";

import { useEffect, useRef } from "react";
import { CheckSquare, ChevronRight, Link2, Loader2, MinusSquare, Square, Trash2, X } from "lucide-react";
import type { Patch } from "@/lib/types";
import type { PatchLibrarySelection } from "@/lib/usePatchLibrarySelection";
export { usePatchLibrarySelection } from "@/lib/usePatchLibrarySelection";
export type { PatchLibrarySelection } from "@/lib/usePatchLibrarySelection";

export function PatchSelectionToolbar({ selection, patches, resultsPending = false }: { selection: PatchLibrarySelection; patches: Patch[]; resultsPending?: boolean }) {
  const { plan, selected, origins, checking, deleting, selectingAll, selectAllSkipped, allSelected } = selection;
  const loaded = new Set(patches.map((patch) => patch.patch_id));
  const hiddenCount = plan?.deleted_patch_ids.filter((id) => !loaded.has(id)).length ?? 0;
  const count = plan?.deleted_patch_ids.length ?? origins.size;
  const blocked = Boolean(plan?.blocked_patch_ids.length);
  return (
    <div className="border-b border-line bg-panel/60 px-3 py-3 sm:px-4">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
        <div className="flex min-w-0 flex-wrap items-center gap-2.5 text-sm">
          <button
            className={`ghost-button min-h-8 size-8 shrink-0 p-0 shadow-none ${selected.size ? "border-cyan/20 bg-cyan/10 text-cyan" : "text-ink/40"}`}
            role="checkbox"
            aria-checked={allSelected ? true : selected.size > 0 ? "mixed" : false}
            aria-label={allSelected ? "Clear selection" : "Select all matching patches"}
            aria-busy={selectingAll}
            title={allSelected ? "Clear selection" : "Select all matching patches"}
            disabled={resultsPending || selectingAll || deleting || selection.dragging}
            onClick={() => void selection.toggleAll()}
            type="button"
          >
            {selectingAll ? <Loader2 size={16} className="animate-spin" aria-hidden="true" /> : allSelected ? <CheckSquare size={16} aria-hidden="true" /> : selected.size > 0 ? <MinusSquare size={16} aria-hidden="true" /> : <Square size={16} aria-hidden="true" />}
          </button>
          <span role="status" aria-live="polite" className="whitespace-nowrap font-medium text-ink"><span className="tabular-nums">{selected.size}</span> selected{selectingAll ? <span className="sr-only">. Selecting all matching patches.</span> : null}</span>
          {count > selected.size ? <span className="inline-flex items-center gap-1.5 rounded-full border border-cyan/15 bg-cyan/5 px-2 py-0.5 text-xs font-medium text-cyan"><Link2 size={12} aria-hidden="true" /><span className="tabular-nums">+{count - selected.size}</span> dependent</span> : null}
          {checking && !selectingAll ? <span className="inline-flex items-center gap-1.5 text-xs text-ink/50"><Loader2 size={13} className="animate-spin" aria-hidden="true" />Checking…</span> : null}
        </div>
        <div className="ml-auto flex max-w-full flex-wrap items-center justify-end gap-2">
          <button className="ghost-button min-h-9 border-transparent bg-transparent px-2.5 py-1.5 text-xs shadow-none disabled:bg-transparent" disabled={(!selected.size && !selectingAll) || deleting} onClick={() => selection.setSelected(new Set())} type="button"><X size={14} aria-hidden="true" />Clear selection</button>
          <button className="tool-button tool-button-danger min-h-9 px-3 py-1.5 text-xs" aria-label={`Review deletion of ${count} patch${count === 1 ? "" : "es"}`} disabled={resultsPending || !plan || checking || blocked || deleting || !selected.size} onClick={() => selection.setReview(plan!)} type="button">
            <Trash2 size={14} aria-hidden="true" />Delete{count > 0 ? <span className="min-w-5 rounded bg-white/20 px-1.5 py-0.5 text-center text-[11px] tabular-nums">{count}</span> : null}
          </button>
        </div>
      </div>
      {hiddenCount > 0 ? <div className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-line/80 pt-2.5 text-xs text-ink/60"><span>{hiddenCount} patch{hiddenCount === 1 ? "" : "es"} outside the loaded results</span><button className="inline-flex items-center gap-0.5 rounded font-medium text-cyan transition hover:text-moss disabled:opacity-50" disabled={deleting} onClick={() => selection.setReview(plan!)} type="button">View all<ChevronRight size={13} aria-hidden="true" /></button></div> : null}
      {selectAllSkipped > 0 ? <p className="mt-2 text-xs text-ink/60" role="status">{selectAllSkipped} patch{selectAllSkipped === 1 ? " is" : "es are"} in use and {selectAllSkipped === 1 ? "wasn’t" : "weren’t"} selected.</p> : null}
      {blocked ? <div className="inline-error mt-3" role="alert">This selection includes active patches. Discard them before deleting. <button className="font-medium underline underline-offset-2" onClick={() => selection.setReview(plan!)} type="button">View affected patches</button></div> : null}
      {selection.error ? <div role="alert" className="inline-error mt-3">{selection.error} <button className="font-medium underline underline-offset-2" disabled={deleting || checking} onClick={selection.retry} type="button">Retry</button></div> : null}
    </div>
  );
}

export function PatchDeletionReview({ selection, modelLabel }: { selection: PatchLibrarySelection; modelLabel: (id: string) => string }) {
  const dialogRef = useRef<HTMLElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const review = selection.review;
  const close = () => { if (!selection.deleting) selection.setReview(null); };
  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null;
    cancelRef.current?.focus();
    return () => previousFocus?.focus();
  }, []);
  if (!review) return null;
  const selectedIds = new Set(review.selected_patch_ids);
  const count = review.deleted_patch_ids.length;
  const changed = !selection.plan || selection.plan.preview_token !== review.preview_token;
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink/30 p-4" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
      <section ref={dialogRef} aria-labelledby="batch-delete-title" aria-describedby="batch-delete-description" aria-modal="true" role="dialog" className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-lg border border-line bg-white p-5 shadow-xl" onKeyDown={(event) => {
        if (event.key === "Escape") { event.preventDefault(); close(); }
        if (event.key !== "Tab") return;
        const controls = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], input:not(:disabled)') ?? []);
        const first = controls[0], last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }}>
        <div className="flex items-start justify-between gap-3">
          <div><span className="label">Review deletion</span><h2 id="batch-delete-title" className="mt-1 text-base font-semibold">Delete {count} patch{count === 1 ? "" : "es"}?</h2></div>
          <button aria-label="Close deletion review" className="ghost-button p-1" disabled={selection.deleting} onClick={close} type="button"><X size={18} /></button>
        </div>
        <p id="batch-delete-description" className="mt-3 text-sm text-ink/70">{selectedIds.size} selected + {review.dependent_patch_ids.length} dependent. This permanently removes the listed saved records and tensor files.</p>
        <ul className="my-4 min-h-0 overflow-y-auto divide-y divide-line rounded-md border border-line">
          {review.patches.map((patch) => <li className="grid gap-1 p-3" key={patch.patch_id}>
            <div className="flex flex-wrap items-center justify-between gap-2"><span className="break-all font-mono text-xs">{patch.patch_id}</span><span className={`text-xs ${patch.active ? "text-rust" : "text-cyan"}`}>{patch.active ? "Active — discard first" : selectedIds.has(patch.patch_id) ? "Selected by you" : "Included automatically"}</span></div>
            <div className="break-words text-xs text-ink/70">{modelLabel(patch.base_model)}{!selectedIds.has(patch.patch_id) ? ` · Included with ${patch.included_by.join(", ")}` : ""}</div>
            {patch.baked_into_model ? <div className="text-xs text-ink/70">Removing this record does not undo edits baked into the model.</div> : null}
          </li>)}
        </ul>
        {changed ? <p role="status" className="mb-3 text-sm text-rust">The selection is being checked or history has changed. Return to the library and review the updated selection.</p> : null}
        <div className="flex justify-end gap-2">
          <button ref={cancelRef} className="ghost-button min-h-9 px-3 py-1" disabled={selection.deleting} onClick={close} type="button">Back to selection</button>
          <button className="tool-button tool-button-danger min-h-9 px-3 py-1" disabled={selection.deleting || changed || selection.checking || Boolean(review.blocked_patch_ids.length)} onClick={() => void selection.commit()} type="button"><Trash2 size={15} />{selection.deleting ? "Deleting…" : `Delete ${count} patch${count === 1 ? "" : "es"}`}</button>
        </div>
      </section>
    </div>
  );
}
