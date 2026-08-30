"use client";

import { Save, XCircle } from "lucide-react";
import { useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { Patch } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";

export function PatchHistory({ patches, onPatchesChanged }: { patches: Patch[]; onPatchesChanged?: () => void | Promise<void> }) {
  const [busyPatchId, setBusyPatchId] = useState("");
  const [savingPatchId, setSavingPatchId] = useState("");
  const [saveMessage, setSaveMessage] = useState("");
  const [exportDirectory] = usePersistentState("kew:settings:modelExportPath", "");
  const [error, setError] = useState("");
  const activePatches = useMemo(
    () =>
      patches
        .filter((patch) => patch.active)
        .sort((left, right) => {
          if (left.base_model !== right.base_model) return left.base_model.localeCompare(right.base_model);
          return (left.active_index ?? 0) - (right.active_index ?? 0);
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

  async function discardPatch(patchId: string) {
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
    setSaveMessage("");
    setError("");
    try {
      if (!targetPath) {
        throw new Error("Set the default save directory in Settings first.");
      }
      const result = await api.exportPatch(patchId, { output_path: targetPath });
      setSaveMessage(`Saved patch to ${result.output_path}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSavingPatchId("");
    }
  }

  return (
    <section className="panel grid gap-6">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">Patch History</h2>
        <span className="label">{activePatches.length} active / {patches.length} saved</span>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold">Active Model Lineage</h3>
          <span className="label">current model state</span>
        </div>
        <PatchTable
          patches={activePatches}
          latestActiveIndexByModel={latestActiveIndexByModel}
          busyPatchId={busyPatchId}
          savingPatchId={savingPatchId}
          canSavePatches={Boolean(exportDirectory.trim())}
          onDiscard={discardPatch}
          onRevertTo={revertToPatch}
          onSavePatch={savePatch}
          mode="active"
        />
        {activePatches.length === 0 ? <div className="rounded-md border border-line bg-panel p-4 text-sm text-ink/70">No active patches. Current model state is clean.</div> : null}
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold">Saved Patch Library</h3>
          <span className="label">newest first</span>
        </div>
        <PatchTable
          patches={savedPatches}
          latestActiveIndexByModel={latestActiveIndexByModel}
          busyPatchId={busyPatchId}
          savingPatchId={savingPatchId}
          canSavePatches={Boolean(exportDirectory.trim())}
          onDiscard={discardPatch}
          onRevertTo={revertToPatch}
          onSavePatch={savePatch}
          mode="library"
        />
        {patches.length === 0 ? <div className="rounded-md border border-line bg-panel p-4 text-sm text-ink/70">No patches saved yet.</div> : null}
      </div>
      {saveMessage ? <div className="rounded-md border border-cyan/30 bg-cyan/10 p-3 text-sm text-cyan">{saveMessage}</div> : null}
      {error ? <div className="mt-3 rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
    </section>
  );
}

function PatchTable({
  patches,
  latestActiveIndexByModel,
  busyPatchId,
  savingPatchId,
  canSavePatches,
  onDiscard,
  onRevertTo,
  onSavePatch,
  mode
}: {
  patches: Patch[];
  latestActiveIndexByModel: Record<string, number>;
  busyPatchId: string;
  savingPatchId: string;
  canSavePatches: boolean;
  onDiscard: (patchId: string) => void;
  onRevertTo: (patchId: string) => void;
  onSavePatch: (patchId: string) => void;
  mode: "active" | "library";
}) {
  if (patches.length === 0) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[840px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left">
            <th className="py-2 pr-4">Patch</th>
            <th className="py-2 pr-4">State</th>
            <th className="py-2 pr-4">Method</th>
            <th className="py-2 pr-4">Model</th>
            <th className="py-2 pr-4">Layers</th>
            <th className="py-2 pr-4">Edits</th>
            <th className="py-2 pr-4">Created</th>
            <th className="py-2 pr-4">Action</th>
          </tr>
        </thead>
        <tbody>
          {patches.map((patch) => {
            const latestIndex = latestActiveIndexByModel[patch.base_model] ?? -1;
            const isLatestActive = Boolean(patch.active && patch.active_index === latestIndex);
            const isBakedLineage = Boolean(patch.lineage_only || patch.baked_into_model);
            const canRevertHere = Boolean(patch.active && !isLatestActive && !isBakedLineage);
            const canDiscardLatest = Boolean(isLatestActive && !isBakedLineage);
            const actionLabel = canRevertHere ? "Revert Here" : "Discard";
            const actionTitle = canDiscardLatest
              ? "Discard the latest active patch and step the model back"
              : patch.active
                ? canRevertHere
                  ? "Discard all active patches after this point"
                  : "This baked lineage is already the current model state"
                : "Saved patch artifact; not part of the current active model lineage";
            const onClick = canRevertHere ? onRevertTo : onDiscard;
            return (
              <tr key={`${mode}-${patch.patch_id}`} className="border-b border-line/70">
                <td className="py-2 pr-4 font-mono text-xs">{patch.patch_id}</td>
                <td className="py-2 pr-4">
                  {patch.active ? (
                    <span className="rounded border border-cyan/30 bg-cyan/10 px-2 py-1 text-xs font-medium text-cyan">
                      {isBakedLineage ? "Baked" : isLatestActive ? "Latest" : `Active #${(patch.active_index ?? 0) + 1}`}
                    </span>
                  ) : (
                    <span className="rounded border border-line bg-panel px-2 py-1 text-xs text-ink/50">Saved</span>
                  )}
                </td>
                <td className="py-2 pr-4">{patch.method.toUpperCase()}</td>
                <td className="py-2 pr-4">{patch.base_model}</td>
                <td className="py-2 pr-4">{patch.modified_layers.join(", ")}</td>
                <td className="py-2 pr-4">{patch.edits.length}</td>
                <td className="py-2 pr-4">{new Date(patch.created_at).toLocaleString()}</td>
                <td className="py-2 pr-4">
                  <div className="flex flex-wrap gap-2">
                    <button
                      className="ghost-button min-h-8 px-2 py-1 text-xs"
                      disabled={savingPatchId === patch.patch_id || !canSavePatches}
                      onClick={() => onSavePatch(patch.patch_id)}
                      title={canSavePatches ? "Save this patch metadata and tensor deltas" : "Set the default save directory in Settings first"}
                      type="button"
                    >
                      <Save size={14} />
                      {savingPatchId === patch.patch_id ? "Saving" : "Save Patch"}
                    </button>
                    <button
                      className="ghost-button min-h-8 px-2 py-1 text-xs"
                      disabled={(!canRevertHere && !canDiscardLatest) || busyPatchId === patch.patch_id}
                      onClick={() => onClick(patch.patch_id)}
                      title={actionTitle}
                      type="button"
                    >
                      <XCircle size={14} />
                      {busyPatchId === patch.patch_id ? "Updating" : actionLabel}
                    </button>
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
