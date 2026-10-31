"use client";

import { useEffect, useRef } from "react";
import type { MouseEvent, PointerEvent, RefObject } from "react";
import { selectPatchRange } from "./patchSelection";

type Options = {
  enabled: boolean;
  orderedIds: string[];
  selected: ReadonlySet<string>;
  included: ReadonlyMap<string, string[]>;
  blocked: ReadonlySet<string>;
  onChange: (next: Set<string>) => void;
  onDragging: (dragging: boolean) => void;
};

export function usePatchDragSelection(scroller: RefObject<HTMLDivElement>, options: Options) {
  const latest = useRef(options);
  latest.current = options;
  const anchor = useRef<string | null>(null);
  const gesture = useRef<{
    pointerId: number; anchor: string; before: Set<string>; selecting: boolean;
    startX: number; startY: number; x: number; y: number; moved: boolean; focus: string;
  } | null>(null);
  const frame = useRef(0);
  const suppressClick = useRef(false);

  function updateRange() {
    const current = gesture.current, node = scroller.current;
    if (!current || !node || !current.moved) return;
    const bounds = node.getBoundingClientRect();
    const x = Math.max(bounds.left + 2, Math.min(bounds.right - 2, current.x));
    const y = Math.max(bounds.top + 45, Math.min(bounds.bottom - 2, current.y));
    const row = document.elementFromPoint(x, y)?.closest<HTMLElement>("[data-patch-id]");
    if (row && node.contains(row)) current.focus = row.dataset.patchId!;
    const value = latest.current;
    const next = selectPatchRange(value.orderedIds, current.before, current.anchor, current.focus, current.selecting, value.blocked);
    if (next.size !== value.selected.size || [...next].some((id) => !value.selected.has(id))) value.onChange(next);
  }

  function autoScroll() {
    const current = gesture.current, node = scroller.current;
    if (!current || !node || !current.moved) return;
    const bounds = node.getBoundingClientRect(), margin = 36;
    const speed = current.y < bounds.top + margin ? -Math.min(14, (bounds.top + margin - current.y) / 3)
      : current.y > bounds.bottom - margin ? Math.min(14, (current.y - bounds.bottom + margin) / 3) : 0;
    if (speed) { node.scrollTop += speed; updateRange(); }
    frame.current = requestAnimationFrame(autoScroll);
  }

  function finish(cancel = false) {
    const current = gesture.current;
    gesture.current = null;
    cancelAnimationFrame(frame.current);
    if (!current) return;
    if (current.moved) {
      if (cancel) latest.current.onChange(current.before);
      latest.current.onDragging(false);
    }
    if (scroller.current?.hasPointerCapture(current.pointerId)) scroller.current.releasePointerCapture(current.pointerId);
    // The click following pointerup belongs to the drag, not to the last row.
    setTimeout(() => { suppressClick.current = false; }, 0);
  }

  useEffect(() => {
    const up = () => finish();
    const cancel = () => finish(true);
    const key = (event: KeyboardEvent) => { if (event.key === "Escape") finish(true); };
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", cancel);
    window.addEventListener("blur", cancel);
    window.addEventListener("keydown", key);
    return () => {
      cancelAnimationFrame(frame.current);
      window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", cancel);
      window.removeEventListener("blur", cancel); window.removeEventListener("keydown", key);
    };
    // Mutable refs keep the handlers current without interrupting a drag on render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggle(id: string, shift = false) {
    const value = latest.current;
    if (!value.enabled || value.blocked.has(id) || (!shift && value.included.has(id) && !value.selected.has(id))) return;
    const selecting = !value.selected.has(id);
    if (shift && anchor.current) value.onChange(selectPatchRange(value.orderedIds, value.selected, anchor.current, id, selecting, value.blocked));
    else {
      const next = new Set(value.selected);
      selecting ? next.add(id) : next.delete(id);
      value.onChange(next);
      anchor.current = id;
    }
  }

  return {
    toggle,
    onPointerDown(event: PointerEvent<HTMLDivElement>) {
      if (!latest.current.enabled || event.pointerType !== "mouse" || event.button !== 0 || (event.target as Element).closest("button, a, input, label")) return;
      const row = (event.target as Element).closest<HTMLElement>("[data-patch-id]");
      const id = row?.dataset.patchId;
      if (!id || latest.current.blocked.has(id)) return;
      suppressClick.current = false;
      gesture.current = { pointerId: event.pointerId, anchor: id, focus: id, before: new Set(latest.current.selected), selecting: !latest.current.selected.has(id), startX: event.clientX, startY: event.clientY, x: event.clientX, y: event.clientY, moved: false };
    },
    onPointerMove(event: PointerEvent<HTMLDivElement>) {
      const current = gesture.current;
      if (!current || current.pointerId !== event.pointerId) return;
      current.x = event.clientX; current.y = event.clientY;
      if (!current.moved && Math.hypot(current.x - current.startX, current.y - current.startY) < 5) return;
      event.preventDefault();
      if (!current.moved) {
        current.moved = true; suppressClick.current = true; anchor.current = current.anchor;
        scroller.current?.setPointerCapture(event.pointerId);
        latest.current.onDragging(true);
        frame.current = requestAnimationFrame(autoScroll);
      }
      updateRange();
    },
    onClickCapture(event: MouseEvent<HTMLDivElement>) {
      if (suppressClick.current) { event.preventDefault(); event.stopPropagation(); }
    },
  };
}
