"use client";

import { Database, Loader2, RefreshCw, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";

type CacheEntry = {
  layer: number;
  path: string;
  filename: string;
  size_bytes: number;
  modified_at: number;
};

type CacheResponse = {
  model_id: string;
  method: string;
  dataset: string;
  stats_dir: string;
  statistics: CacheEntry[];
};

export function SecondMomentCacheDialog({ modelId, method, onClose }: { modelId: string; method: "rome" | "r_rome" | "memit"; onClose: () => void }) {
  const [cache, setCache] = useState<CacheResponse | null>(null);
  const [deletingLayer, setDeletingLayer] = useState<number | null>(null);
  const [error, setError] = useState("");

  async function refresh() {
    setError("");
    try {
      setCache(await api.mom2Cache({ model_id: modelId, method }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function remove(layer: number) {
    setDeletingLayer(layer);
    setError("");
    try {
      setCache(await api.deleteMom2Statistics({ model_id: modelId, method, layers: [layer] }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingLayer(null);
    }
  }

  useEffect(() => {
    void refresh();
  }, [method, modelId]);

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
            <p className="mt-1 text-sm text-ink/60">{modelId} · {method === "r_rome" ? "R-ROME" : method.toUpperCase()} · {cache?.dataset ?? "Loading dataset..."}</p>
          </div>
          <div className="flex items-center gap-1">
            <button className="ghost-button grid size-8 place-items-center p-0" onClick={() => void refresh()} title="Refresh cache" type="button">
              <RefreshCw size={15} />
            </button>
            <button className="ghost-button grid size-8 place-items-center p-0" onClick={onClose} title="Close" type="button">
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="mt-4 divide-y divide-line border-y border-line">
          {cache === null && !error ? <div className="flex items-center gap-2 py-5 text-sm text-ink/60"><Loader2 className="animate-spin" size={16} /> Loading cache</div> : null}
          {cache?.statistics.length === 0 ? <div className="py-5 text-sm text-ink/60">No cached statistics for this model and method.</div> : null}
          {cache?.statistics.map((entry) => (
            <div key={entry.path} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 py-3">
              <div className="min-w-0">
                <div className="text-sm font-medium">Layer {entry.layer}</div>
                <div className="mt-1 font-mono text-xs text-ink/60">{formatBytes(entry.size_bytes)} · {formatDate(entry.modified_at)}</div>
              </div>
              <button aria-label={`Delete statistics for layer ${entry.layer}`} className="ghost-button grid size-8 place-items-center p-0 text-rust" disabled={deletingLayer !== null} onClick={() => void remove(entry.layer)} title="Delete cached statistics" type="button">
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
