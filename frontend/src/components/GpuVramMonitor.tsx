"use client";

import { Cpu } from "lucide-react";

import type { ModelRuntimeStatus } from "@/lib/types";

export function GpuVramMonitor({ runtime, variant = "panel" }: { runtime: ModelRuntimeStatus[]; variant?: "panel" | "sidebar" }) {
  const loadedRuntime = runtime.find((status) => status.loaded);
  const gpu = loadedRuntime?.gpu ?? runtime.find((status) => status.gpu)?.gpu;
  const isMps = Boolean(gpu?.mps_available);
  const gpuAvailable = Boolean(gpu?.cuda_available || isMps);
  const memoryLabel = isMps ? "GPU unified memory" : "GPU VRAM";
  const totalBytes = (isMps ? gpu?.recommended_max_bytes : gpu?.total_bytes) ?? 0;
  const usedBytes = (isMps ? gpu?.driver_allocated_bytes : gpu?.used_bytes) ?? (totalBytes && gpu?.free_bytes !== undefined
    ? Math.max(0, totalBytes - gpu.free_bytes)
    : gpu?.allocated_bytes ?? 0);
  const usedPercent = totalBytes ? Math.min(100, Math.max(0, (usedBytes / totalBytes) * 100)) : 0;

  if (!runtime.length) {
    return variant === "sidebar" ? (
      <section className="rounded-lg border border-white/10 bg-white/[0.04] p-3" aria-label="GPU memory">
        <div className="flex items-center gap-2 text-xs font-medium text-slate-200"><Cpu size={14} />GPU VRAM</div>
        <div className="mt-2 text-[11px] text-slate-500">Waiting for backend status</div>
      </section>
    ) : <div className="mb-4 rounded-md border border-line bg-panel p-3 text-sm text-ink/70">GPU: waiting for backend status.</div>;
  }

  if (variant === "sidebar") {
    return (
      <section className="rounded-lg border border-white/10 bg-white/[0.04] p-3" aria-label="GPU memory">
        <div className="flex items-center justify-between gap-2 text-xs font-medium text-slate-200">
          <span className="flex items-center gap-2"><Cpu size={14} />{memoryLabel}</span>
          <span className={gpuAvailable ? "text-cyan" : "text-slate-500"}>{gpuAvailable ? `${usedPercent.toFixed(0)}%` : "N/A"}</span>
        </div>
        {gpuAvailable ? (
          <>
            <div className="mt-2 truncate text-xs text-slate-300" title={gpu?.device_name}>{gpu?.device_name ?? `CUDA ${gpu?.device ?? "0"}`}</div>
            <div className="mt-1 font-mono text-[11px] text-slate-400">{formatBytes(usedBytes)} / {formatBytes(totalBytes)}</div>
            {isMps && <div className="mt-1 text-[10px] text-slate-400">Worker usage / recommended budget</div>}
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
              <div className="h-full rounded-full bg-cyan transition-[width]" style={{ width: `${usedPercent}%` }} />
            </div>
            {loadedRuntime ? (
              <div className="mt-2 flex min-w-0 items-center gap-2 font-mono text-[10px] text-emerald-300" title={loadedRuntime.model_id}>
                <span className="relative flex h-2 w-2 shrink-0" aria-label="Model active">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/70" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(74,222,128,0.9)]" />
                </span>
                <span className="truncate">{loadedRuntime.model_id} · {formatBytes(loadedRuntime.model_vram_bytes ?? 0)} {isMps ? "weights" : "VRAM"}</span>
              </div>
            ) : <div className="mt-2 font-mono text-[10px] text-slate-500">No model loaded</div>}
          </>
        ) : <div className="mt-2 text-[11px] text-slate-500">GPU acceleration not available to backend</div>}
      </section>
    );
  }

  if (!gpuAvailable) {
    return <div className="mb-4 rounded-md border border-line bg-panel p-3 text-sm text-ink/70">GPU acceleration not available to the backend.</div>;
  }

  return (
    <div className="mb-4 grid gap-3 rounded-md border border-line bg-panel p-3 text-sm md:grid-cols-[minmax(0,0.9fr)_minmax(0,1.4fr)_minmax(0,1fr)]">
      <div>
        <div className="label mb-1">GPU</div>
        <div className="font-medium">{gpu?.device_name ?? `CUDA ${gpu?.device ?? "0"}`}</div>
      </div>
      <div>
        <div className="mb-1 flex items-center justify-between gap-3">
          <div className="label">{memoryLabel}</div>
          <div className="font-mono text-xs text-ink/70">{formatBytes(usedBytes)} / {formatBytes(totalBytes)}</div>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-cyan transition-[width]" style={{ width: `${usedPercent}%` }} /></div>
        <div className="mt-1 font-mono text-xs text-ink/50">{usedPercent.toFixed(1)}% used · {gpu?.memory_source ?? "torch"}</div>
        {isMps && <div className="mt-1 text-xs text-ink/50">Worker usage / recommended budget, shared with system RAM.</div>}
      </div>
      <div>
        <div className="label mb-1">Loaded Model</div>
        <div className="font-mono text-xs text-ink/70">{loadedRuntime ? `${formatBytes(loadedRuntime.model_vram_bytes ?? 0)} (${loadedRuntime.model_id})` : "No model loaded"}</div>
        <div className="mt-1 font-mono text-xs text-ink/50">{formatBytes(gpu?.allocated_bytes ?? 0)} {isMps ? "in tensors" : `allocated / ${formatBytes(gpu?.reserved_bytes ?? 0)} reserved`}</div>
      </div>
    </div>
  );
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
