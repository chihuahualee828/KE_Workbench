import type { Experiment, Patch } from "./types";

export function patchDeletionBlockReason(patchId: string, patches: Patch[]): string {
  const byId = new Map(patches.map((patch) => [patch.patch_id, patch]));
  const protectedIds = new Set<string>();
  const pending = patches.filter((patch) => patch.active).map((patch) => patch.patch_id);
  while (pending.length) {
    const id = pending.pop()!;
    if (protectedIds.has(id)) continue;
    protectedIds.add(id);
    pending.push(...(byId.get(id)?.parent_patch_ids ?? []));
  }
  return protectedIds.has(patchId) ? "This patch or a dependent patch is active. Discard it before deleting." : "";
}

export function patchMutationBlockReason(experiments: Experiment[]): string {
  return experiments.some((experiment) => experiment.status === "running" && (
    experiment.type === "evaluation" || experiment.type.endsWith("_edit")
    || experiment.type.endsWith("_edits") || experiment.type === "prepare_mom2_statistics"
  )) ? "Wait for editing or evaluation to finish before changing patches." : "";
}
