import type { EditJob, Patch } from "./types";

export function applyEditPatchState(patches: Patch[], state: NonNullable<EditJob["patch_state"]>): Patch[] {
  const active = new Map(state.active_patch_ids.map((id, index) => [id, index]));
  return patches.map((patch) => patch.base_model !== state.model_id ? patch : {
    ...patch,
    active: active.has(patch.patch_id),
    active_index: active.get(patch.patch_id) ?? null,
  });
}
