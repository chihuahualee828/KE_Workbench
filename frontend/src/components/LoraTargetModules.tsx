"use client";

import { ChevronDown, RefreshCw } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { api } from "@/lib/api";
import { isLoraGroupSelected, toggleLoraGroup, type LoraModuleCatalog } from "@/lib/loraModules";
import { VirtualizedList } from "./VirtualizedList";

export function LoraTargetModules({ modelId, modelLoaded, disabled, value, onChange }: {
  modelId: string;
  modelLoaded: boolean;
  disabled: boolean;
  value: string;
  onChange: (value: string) => void;
}) {
  const [catalog, setCatalog] = useState<LoraModuleCatalog | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const requestId = useRef(0);
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const panelId = useId();
  const selected = value.split(",").map((target) => target.trim()).filter(Boolean);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (!open) return;
    searchRef.current?.focus();
    const closeOutside = (event: Event) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("focusin", closeOutside);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("focusin", closeOutside);
    };
  }, [open]);

  useEffect(() => {
    requestId.current += 1;
    setOpen(false);
    setCatalog(null);
    setLoading(false);
    setError("");
    setSearch("");
    return () => { requestId.current += 1; };
  }, [modelId, modelLoaded]);

  async function inspect() {
    const current = ++requestId.current;
    setLoading(true);
    setError("");
    try {
      const response = await api.loraModules({ model_id: modelId });
      if (current === requestId.current && response.model_id === modelId) setCatalog(response);
    } catch (reason) {
      if (current === requestId.current) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (current === requestId.current) setLoading(false);
    }
  }

  const query = search.trim().toLowerCase();
  const groups = catalog?.groups.filter((group) => group.name.toLowerCase().includes(query) ||
    group.modules.some((module) => module.path.toLowerCase().includes(query) || module.type.toLowerCase().includes(query))) ?? [];

  function toggleOpen() {
    const nextOpen = !open;
    setOpen(nextOpen);
    if (nextOpen && !catalog && !loading) void inspect();
  }

  return (
    <div ref={rootRef} className="relative grid min-w-0 grid-cols-1 gap-1" onKeyDown={(event) => {
      if (event.key === "Escape" && open) {
        event.preventDefault();
        event.stopPropagation();
        setOpen(false);
        triggerRef.current?.focus();
      }
    }}>
      <label className="label" htmlFor={`${panelId}-trigger`}>Target modules</label>
      <button
        ref={triggerRef}
        id={`${panelId}-trigger`}
        type="button"
        className="control flex min-w-0 items-center justify-between gap-2 text-left"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        disabled={!modelLoaded || disabled}
        title={selected.join(", ") || "Select target modules"}
        onClick={toggleOpen}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" && !open) {
            event.preventDefault();
            toggleOpen();
          }
        }}
      >
        <span className="min-w-0 truncate">{selected.length ? selected.join(", ") : "Select target modules"}</span>
        <ChevronDown size={16} className={`shrink-0 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {!modelLoaded ? <p className="text-xs text-ink/60">Load this model to select its LoRA targets.</p> : null}
      {open && modelLoaded ? (
        <div id={panelId} role="group" aria-label="Available LoRA target modules" className="absolute inset-x-0 top-full z-30 mt-1 grid max-h-[60vh] min-w-0 grid-cols-1 gap-2 overflow-y-auto overscroll-contain rounded-lg border border-line bg-white p-3 shadow-xl">
          <div className="flex min-w-0 items-center gap-2">
            <input ref={searchRef} className="control min-w-0 text-xs" aria-label="Filter available LoRA modules" placeholder="Search modules…" value={search} onChange={(event) => setSearch(event.target.value)} />
            <button type="button" className="ghost-button min-h-8 shrink-0 p-2" aria-label="Refresh available modules" title="Refresh available modules" disabled={disabled || loading} onClick={() => void inspect()}>
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            </button>
          </div>
          {loading ? <p role="status" className="text-xs text-ink/60">Inspecting loaded model…</p> : null}
          {error ? <p role="alert" className="text-xs text-rust [overflow-wrap:anywhere]">{error}</p> : null}
          {catalog?.model_id === modelId ? <>
            <p className="text-xs text-ink/60">{catalog.module_count} compatible projections. Edit Layers can restrict the layer selection.</p>
            <div className="grid min-w-0 gap-1">
              {groups.map((group) => {
                const available = group.modules.filter((module) => module.available).length;
                return <div key={group.name} className="min-w-0 rounded-md border border-line p-2">
                  <label className="flex min-w-0 cursor-pointer items-start gap-2 text-xs">
                    <input type="checkbox" className="mt-0.5 shrink-0" checked={isLoraGroupSelected(group, value)} disabled={disabled || loading || !available} onChange={(event) => onChange(toggleLoraGroup(group, value, event.target.checked))} />
                    <span className="min-w-0 [overflow-wrap:anywhere]"><span className="font-mono">{group.name}</span><span className="block text-ink/60">{available} available / {group.modules.length} modules</span></span>
                  </label>
                  <details className="mt-2 min-w-0">
                    <summary className="text-xs text-ink/60">Module paths and shapes</summary>
                    <VirtualizedList
                      className="mt-2 h-40 min-w-0 overflow-y-auto overscroll-contain"
                      itemCount={group.modules.length}
                      estimatedItemHeight={72}
                      getItem={(index) => group.modules[index]}
                      renderItem={(module) => <div className="grid min-w-0 gap-1 border-t border-line py-2 text-[11px]">
                        <span className="font-mono [overflow-wrap:anywhere]">{module.path}</span>
                        <span className="text-ink/60">{module.type} · weight [{module.weight_shape.join(" × ")}]</span>
                        {module.reason ? <span className="text-rust [overflow-wrap:anywhere]">{module.reason}</span> : null}
                      </div>}
                    />
                  </details>
                </div>;
              })}
              {!groups.length ? <p className="text-xs text-ink/60">No matching modules.</p> : null}
            </div>
          </> : null}
          <details className="min-w-0 border-t border-line pt-2">
            <summary className="text-xs text-ink/60">Edit names manually</summary>
            <label className="mt-2 grid min-w-0 grid-cols-1 gap-1">
              <span className="text-xs text-ink/60">Comma-separated module names or full paths</span>
              <input className="control text-xs" value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)} spellCheck={false} />
            </label>
          </details>
          <div className="flex items-center justify-between gap-2 border-t border-line pt-2">
            <span className="text-xs text-ink/60">{selected.length} selected targets</span>
            <button type="button" className="ghost-button min-h-8 px-2 py-1 text-xs" onClick={() => { setOpen(false); triggerRef.current?.focus(); }}>Done</button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
