"use client";

import { Crosshair, Maximize2, Minus, Move, Plus } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { TraceCell } from "@/lib/types";

const LABEL_WIDTH = 116;
const CELL_WIDTH = 44;
const CELL_HEIGHT = 42;
const HEADER_HEIGHT = 54;
const MIN_SCALE = 0.15;
const MAX_SCALE = 3;
type View = { x: number; y: number; scale: number };
type Drag = { pointerId: number; x: number; y: number; view: View; moved: boolean; mode: "pan" | "marker" };

export function TransformerGrid({ cells, tokens, selected, onSelect }: { cells: TraceCell[]; tokens: string[]; selected: TraceCell | null; onSelect: (cell: TraceCell) => void }) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<Drag | null>(null);
  const [view, setView] = useState<View>({ x: 12, y: 12, scale: 1 });
  const [hovered, setHovered] = useState<TraceCell | null>(null);
  const layers = useMemo(() => Array.from(new Set(cells.map((cell) => cell.layer))).sort((a, b) => a - b), [cells]);
  const lookup = useMemo(() => new Map(cells.map((cell) => [`${cell.layer}:${cell.token_index}`, cell])), [cells]);
  const maxEffect = useMemo(() => cells.reduce((max, cell) => Math.max(max, Math.abs(cell.indirect_effect)), 0.0001), [cells]);
  const contentWidth = LABEL_WIDTH + layers.length * CELL_WIDTH;
  const contentHeight = HEADER_HEIGHT + tokens.length * CELL_HEIGHT;

  const fit = useCallback((readable = false) => {
    const element = viewportRef.current;
    if (!element) return;
    const scale = Math.max(readable ? 0.8 : MIN_SCALE, Math.min(1.2, (element.clientWidth - 24) / contentWidth, (element.clientHeight - 24) / contentHeight));
    setView({ x: Math.max(12, (element.clientWidth - contentWidth * scale) / 2), y: Math.max(12, (element.clientHeight - contentHeight * scale) / 2), scale });
  }, [contentWidth, contentHeight]);

  useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => fit(true));
    observer.observe(element);
    fit(true);
    return () => observer.disconnect();
  }, [fit]);

  function zoomAt(factor: number, x?: number, y?: number) {
    const element = viewportRef.current;
    if (!element) return;
    const anchorX = x ?? element.clientWidth / 2;
    const anchorY = y ?? element.clientHeight / 2;
    setView((current) => {
      const scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, current.scale * factor));
      return { scale, x: anchorX - (anchorX - current.x) * scale / current.scale, y: anchorY - (anchorY - current.y) * scale / current.scale };
    });
  }

  useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;
    function handleWheel(event: WheelEvent) {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const bounds = element!.getBoundingClientRect();
      zoomAt(event.deltaY < 0 ? 1.12 : 1 / 1.12, event.clientX - bounds.left, event.clientY - bounds.top);
    }
    element.addEventListener("wheel", handleWheel, { passive: false });
    return () => element.removeEventListener("wheel", handleWheel);
  }, []);

  function cellAt(clientX: number, clientY: number) {
    const bounds = viewportRef.current?.getBoundingClientRect();
    if (!bounds) return null;
    const layerIndex = Math.floor(((clientX - bounds.left - view.x) / view.scale - LABEL_WIDTH) / CELL_WIDTH);
    const tokenIndex = Math.floor(((clientY - bounds.top - view.y) / view.scale - HEADER_HEIGHT) / CELL_HEIGHT);
    return lookup.get(`${layers[layerIndex]}:${tokenIndex}`) ?? null;
  }

  function pointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.currentTarget.focus({ preventScroll: true });
    const cell = cellAt(event.clientX, event.clientY);
    const isMarker = cell && cell.layer === selected?.layer && cell.token_index === selected.token_index;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, view, moved: false, mode: isMarker ? "marker" : "pan" };
  }

  function pointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const cell = cellAt(event.clientX, event.clientY);
    setHovered(cell);
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
    if (drag.mode === "marker") {
      if (cell) onSelect(cell);
    } else if (drag.moved) setView({ ...drag.view, x: drag.view.x + dx, y: drag.view.y + dy });
  }

  function endPointer(event: ReactPointerEvent<HTMLDivElement>, canceled = false) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (!canceled && !drag.moved) {
      const cell = cellAt(event.clientX, event.clientY);
      if (cell) onSelect(cell);
    }
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }

  function moveSelection(key: string) {
    const column = selected ? layers.indexOf(selected.layer) : 0;
    const row = selected?.token_index ?? 0;
    const nextColumn = Math.max(0, Math.min(layers.length - 1, column + (key === "ArrowRight" ? 1 : key === "ArrowLeft" ? -1 : 0)));
    const nextRow = Math.max(0, Math.min(tokens.length - 1, row + (key === "ArrowDown" ? 1 : key === "ArrowUp" ? -1 : 0)));
    const cell = lookup.get(`${layers[nextColumn]}:${nextRow}`);
    if (!cell) return;
    onSelect(cell);
  }

  useEffect(() => {
    const element = viewportRef.current;
    if (!element || !selected) return;
    const column = layers.indexOf(selected.layer);
    setView((current) => {
      const x = current.x + (LABEL_WIDTH + column * CELL_WIDTH) * current.scale;
      const y = current.y + (HEADER_HEIGHT + selected.token_index * CELL_HEIGHT) * current.scale;
      return { ...current, x: current.x + (x < 0 ? -x + 12 : Math.min(0, element.clientWidth - x - CELL_WIDTH * current.scale - 12)), y: current.y + (y < 0 ? -y + 12 : Math.min(0, element.clientHeight - y - CELL_HEIGHT * current.scale - 12)) };
    });
  }, [selected, layers]);

  const inspected = hovered ?? selected;
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <span className="flex items-center gap-2 text-xs text-ink/50"><Move size={13} />Drag canvas to pan</span>
        <div className="flex items-center gap-1">
          <button className="ghost-button min-h-8 px-2 py-1" type="button" aria-label="Zoom out" disabled={view.scale <= MIN_SCALE} onClick={() => zoomAt(1 / 1.25)}><Minus size={14} /></button>
          <span className="w-12 text-center text-[11px] tabular-nums text-ink/55">{Math.round(view.scale * 100)}%</span>
          <button className="ghost-button min-h-8 px-2 py-1" type="button" aria-label="Zoom in" disabled={view.scale >= MAX_SCALE} onClick={() => zoomAt(1.25)}><Plus size={14} /></button>
          <button className="ghost-button ml-1 min-h-8 px-2 py-1 text-xs" type="button" onClick={() => fit()}><Maximize2 size={13} />Fit</button>
        </div>
      </div>
      <div ref={viewportRef} tabIndex={0} role="group" aria-label="Token by transformer layer heatmap. Use arrow keys to select a cell, plus or minus to zoom, and zero to fit." className="trace-canvas relative h-[420px] min-h-72 touch-none select-none overflow-hidden rounded-lg border border-line cursor-grab active:cursor-grabbing" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={(event) => endPointer(event)} onPointerCancel={(event) => endPointer(event, true)} onPointerLeave={() => setHovered(null)} onKeyDown={(event) => {
        if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); moveSelection(event.key); }
        if (event.key === "+" || event.key === "=") { event.preventDefault(); zoomAt(1.25); }
        if (event.key === "-") { event.preventDefault(); zoomAt(1 / 1.25); }
        if (event.key === "0") { event.preventDefault(); fit(); }
      }}>
        <svg className="block h-full w-full" role="img" aria-label={`Causal effects across ${tokens.length} tokens and ${layers.length} shared transformer layers`}>
          <g transform={`translate(${view.x} ${view.y}) scale(${view.scale})`}>
            <text x={LABEL_WIDTH} y={16} fontSize={10} fontWeight={600} letterSpacing={1.2} fill="#778196">TRANSFORMER LAYERS →</text>
            <text x={0} y={HEADER_HEIGHT - 12} fontSize={10} fill="#778196">TOKEN / POSITION</text>
            {layers.map((layer, column) => <g key={layer}>
              {(hovered ?? selected)?.layer === layer ? <rect x={LABEL_WIDTH + column * CELL_WIDTH} y={25} width={CELL_WIDTH} height={contentHeight - 25} rx={5} fill="#5b57d6" fillOpacity={0.07} /> : null}
              <text x={LABEL_WIDTH + (column + 0.5) * CELL_WIDTH} y={HEADER_HEIGHT - 12} textAnchor="middle" fontSize={11} fontWeight={500} fill="#637086">{layer}</text>
            </g>)}
            {tokens.map((token, row) => <g key={row}>
              {(hovered ?? selected)?.token_index === row ? <rect x={0} y={HEADER_HEIGHT + row * CELL_HEIGHT} width={contentWidth} height={CELL_HEIGHT} rx={5} fill="#5b57d6" fillOpacity={0.07} /> : null}
              <text x={0} y={HEADER_HEIGHT + (row + 0.5) * CELL_HEIGHT + 4} fontSize={10} fill="#94a0b3">{String(row).padStart(2, "0")}</text>
              <text x={26} y={HEADER_HEIGHT + (row + 0.5) * CELL_HEIGHT + 4} fontSize={11} fill="#39455b" fontFamily="ui-monospace, monospace"><title>{token}</title>{token.length > 10 ? `${token.slice(0, 9)}…` : token.replace(/ /g, "·")}</text>
              {layers.map((layer, column) => {
                const cell = lookup.get(`${layer}:${row}`);
                const intensity = cell ? Math.abs(cell.indirect_effect) / maxEffect : 0;
                const isSelected = selected?.layer === layer && selected.token_index === row;
                const x = LABEL_WIDTH + column * CELL_WIDTH + 3;
                const y = HEADER_HEIGHT + row * CELL_HEIGHT + 3;
                const color = (cell?.indirect_effect ?? 0) < 0 ? "194, 79, 110" : "24, 143, 133";
                return <g key={layer}>
                  <rect x={x} y={y} width={CELL_WIDTH - 6} height={CELL_HEIGHT - 6} rx={6} fill={cell ? `rgba(${color}, ${0.07 + intensity * 0.83})` : "#e9edf3"} stroke={isSelected ? "#5b57d6" : "#ffffff"} strokeWidth={isSelected ? 2.5 : 1}><title>{cell ? `Token ${row}: ${token}, layer ${layer}. Indirect effect: ${cell.indirect_effect.toFixed(4)}` : "Not evaluated"}</title></rect>
                  {isSelected ? <g pointerEvents="none"><circle cx={x + 19} cy={y + 18} r={9} fill="white" stroke="#5b57d6" strokeWidth={1.5} /><circle cx={x + 19} cy={y + 18} r={3.5} fill="#5b57d6" /></g> : null}
                </g>;
              })}
            </g>)}
          </g>
        </svg>
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-[10px] text-ink/55"><span className="flex items-center gap-2"><Crosshair size={12} />Drag marker or use arrow keys to inspect</span><div className="flex items-center gap-2"><span>Negative</span><span className="h-2 w-24 rounded-full" style={{ background: "linear-gradient(to right, #c24f6e, #f0f2f5, #188f85)" }} /><span>Positive effect</span></div></div>
      <p className="mt-2 min-h-5 text-[11px] text-ink/60" aria-live="polite">{inspected ? `Token ${inspected.token_index} “${inspected.token}” · Layer ${inspected.layer} · Effect ${inspected.indirect_effect >= 0 ? "+" : ""}${(inspected.indirect_effect * 100).toFixed(2)} pp` : "Select a cell to inspect its result. Ctrl / ⌘ + scroll to zoom."}</p>
    </div>
  );
}
