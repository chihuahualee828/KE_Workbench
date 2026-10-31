import type { Patch } from "./types";

/** Local preview only: the server also finds descendants outside loaded pages. */
export function patchSelectionOrigins(patches: Patch[], selected: ReadonlySet<string>): Map<string, string[]> {
  const children = new Map<string, string[]>();
  for (const patch of patches) {
    for (const parent of patch.parent_patch_ids ?? []) {
      children.set(parent, [...(children.get(parent) ?? []), patch.patch_id]);
    }
  }
  const origins = new Map<string, string[]>();
  for (const seed of selected) {
    const pending = [seed], visited = new Set<string>();
    while (pending.length) {
      const id = pending.pop()!;
      if (visited.has(id)) continue;
      visited.add(id);
      origins.set(id, [...(origins.get(id) ?? []), seed]);
      pending.push(...(children.get(id) ?? []));
    }
  }
  return origins;
}

/** Apply each range to the gesture's original selection, so reversing a drag undoes it. */
export function selectPatchRange(
  orderedIds: string[], before: ReadonlySet<string>, anchor: string, focus: string,
  selecting: boolean, blocked: ReadonlySet<string>
): Set<string> {
  const next = new Set(before), start = orderedIds.indexOf(anchor), end = orderedIds.indexOf(focus);
  if (start < 0 || end < 0) return next;
  for (const id of orderedIds.slice(Math.min(start, end), Math.max(start, end) + 1)) {
    if (!selecting) next.delete(id);
    else if (!blocked.has(id)) next.add(id);
  }
  return next;
}
