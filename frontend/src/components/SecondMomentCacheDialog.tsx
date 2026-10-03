"use client";

import { Database, Loader2, RefreshCw, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { Mom2Cache, Mom2Dtype } from "@/lib/types";
import { formatMom2Precision, Mom2PrecisionSelect } from "./Mom2PrecisionSelect";

export function SecondMomentCacheDialog({ modelId, method, statsSource, onClose }: { modelId: string; method: "rome" | "r_rome" | "memit"; statsSource?: "base" | "model"; onClose: () => void }) {
  const [cache, setCache] = useState<Mom2Cache | null>(null);
  const [deletingLayer, setDeletingLayer] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestVersion = useRef(0);

  async function refresh() {
    const revision = ++requestVersion.current;
    setBusy(true);
    setError("");
    try {
      const response = await api.mom2Cache({ model_id: modelId, method, stats_source: statsSource });
      if (revision === requestVersion.current) setCache(response);
    } catch (err) {
      if (revision === requestVersion.current) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (revision === requestVersion.current) setBusy(false);
    }
  }

  async function selectPrecision(dtype: Mom2Dtype) {
    if (busy || !cache || dtype === cache.mom2_dtype) return;
    const revision = ++requestVersion.current;
    setBusy(true);
    setError("");
    try {
      await api.setMom2StatisticsPrecision({ model_id: modelId, method, mom2_dtype: dtype });
      const response = await api.mom2Cache({ model_id: modelId, method, stats_source: statsSource, mom2_dtype: dtype });
      if (revision === requestVersion.current) setCache(response);
    } catch (err) {
      if (revision === requestVersion.current) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (revision === requestVersion.current) setBusy(false);
    }
  }

  async function remove(layer: number) {
    if (busy || !cache) return;
    const revision = ++requestVersion.current;
    setBusy(true);
    setDeletingLayer(layer);
    setError("");
    try {
      const response = await api.deleteMom2Statistics({ model_id: modelId, method, layers: [layer], stats_source: statsSource, mom2_dtype: cache.mom2_dtype });
      if (revision === requestVersion.current) setCache(response);
    } catch (err) {
      if (revision === requestVersion.current) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (revision === requestVersion.current) {
        setDeletingLayer(null);
        setBusy(false);
      }
    }
  }

  useEffect(() => {
    setCache(null);
    void refresh();
    return () => { ++requestVersion.current; };
  }, [method, modelId, statsSource]);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/30 p-4 sm:items-center" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }} role="presentation">
      <section aria-labelledby="second-moment-cache-title" aria-modal="true" className="my-auto max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-md border border-line bg-white p-5 shadow-xl" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Database size={17} />
              <h2 id="second-moment-cache-title" className="text-base font-semibold">Second-Moment Cache</h2>
            </div>
            <p className="mt-1 break-words text-sm text-ink/60">
              {modelId} · {method === "r_rome" ? "R-ROME" : method.toUpperCase()} · {cache?.dataset ?? "Loading dataset..."}
              {cache ? ` · ${cache.stats_source === "base" ? `Base model (${cache.base_model_id})` : "Model statistics"}` : ""}
            </p>
          </div>
          <div className="flex items-center gap-1">
            <button className="ghost-button grid size-8 place-items-center p-0" disabled={busy} onClick={() => void refresh()} title="Refresh cache" type="button">
              <RefreshCw size={15} />
            </button>
            <button className="ghost-button grid size-8 place-items-center p-0" onClick={onClose} title="Close" type="button">
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="mt-4 max-w-xs">
          <Mom2PrecisionSelect profile={cache} disabled={busy} onChange={(dtype) => void selectPrecision(dtype)} />
        </div>
        <div className="mt-4 divide-y divide-line border-y border-line">
          {cache === null && !error ? <div className="flex items-center gap-2 py-5 text-sm text-ink/60"><Loader2 className="animate-spin" size={16} /> Loading cache</div> : null}
          {cache?.statistics.length === 0 ? <div className="py-5 text-sm text-ink/60">No cached {formatMom2Precision(cache.mom2_dtype)} statistics for this model and method.</div> : null}
          {cache?.statistics.map((entry) => (
            <div key={entry.path} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 py-3">
              <div className="min-w-0">
                <div className="text-sm font-medium">Layer {entry.layer} <span className="ml-2 font-mono text-xs text-ink/60">{formatMom2Precision(entry.mom2_dtype)}</span></div>
                <div className="mt-1 font-mono text-xs text-ink/60">{formatBytes(entry.size_bytes)} · {formatDate(entry.modified_at)}</div>
              </div>
              <button aria-label={`Delete ${formatMom2Precision(entry.mom2_dtype)} statistics for layer ${entry.layer}`} className="ghost-button grid size-8 place-items-center p-0 text-rust" disabled={busy} onClick={() => void remove(entry.layer)} title="Delete cached statistics" type="button">
                {deletingLayer === entry.layer ? <Loader2 className="animate-spin" size={15} /> : <Trash2 size={15} />}
              </button>
            </div>
          ))}
        </div>
        {error ? <div className="mt-4 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      </section>
    </div>
  );
}

function formatBytes(value: number) {
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value: number) {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value * 1000));
}
