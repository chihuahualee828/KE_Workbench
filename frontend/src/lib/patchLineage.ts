import type { Patch } from "./types";

export type LineageGraphRow = {
  lane: number;
  parentPatchId: string | null;
  incomingLanes: number[];
  outgoingLanes: number[];
  incomingConnections: number[];
  outgoingConnections: number[];
  missingParentId: string | null;
  missingChildCount: number;
};

/** Lay out actual edges by endpoint positions, independently of date/sort direction. */
export function buildLineageGraph(patches: Patch[]): { laneCount: number; rowsByPatchId: Record<string, LineageGraphRow> } {
  const positions = new Map(patches.map((patch, index) => [patch.patch_id, index]));
  const parentOf = (patch: Patch) => patch.sequential_edit ? patch.parent_patch_ids?.at(-1) ?? null : null;
  const visibleChildren = new Map<string, number>();
  const edges: Array<{ start: number; end: number; lane: number }> = [];
  patches.forEach((patch, index) => {
    const parent = parentOf(patch), position = parent ? positions.get(parent) : undefined;
    if (position === undefined || position === index) return;
    visibleChildren.set(parent!, (visibleChildren.get(parent!) ?? 0) + 1);
    edges.push({ start: Math.min(index, position), end: Math.max(index, position), lane: 0 });
  });
  edges.sort((a, b) => a.start - b.start || a.end - b.end);
  const laneEnds: number[] = [];
  const starts = new Map<number, number[]>(), ends = new Map<number, number[]>();
  for (const edge of edges) {
    let lane = laneEnds.findIndex((end) => end <= edge.start);
    if (lane < 0) lane = laneEnds.length;
    laneEnds[lane] = edge.end; edge.lane = lane;
    starts.set(edge.start, [...(starts.get(edge.start) ?? []), lane]);
    ends.set(edge.end, [...(ends.get(edge.end) ?? []), lane]);
  }
  const active = new Set<number>();
  const rowsByPatchId: Record<string, LineageGraphRow> = {};
  let laneCount = Math.max(1, laneEnds.length);
  patches.forEach((patch, index) => {
    const incomingLanes = [...active];
    const incomingConnections = ends.get(index) ?? [], outgoingConnections = starts.get(index) ?? [];
    let lane = incomingConnections[0] ?? outgoingConnections[0];
    if (lane === undefined) { lane = 0; while (active.has(lane)) lane++; }
    incomingConnections.forEach((value) => active.delete(value));
    outgoingConnections.forEach((value) => active.add(value));
    laneCount = Math.max(laneCount, lane + 1);
    const parent = parentOf(patch);
    rowsByPatchId[patch.patch_id] = { lane, parentPatchId: parent, incomingLanes, outgoingLanes: [...active],
      incomingConnections, outgoingConnections,
      missingParentId: parent && !positions.has(parent) ? parent : null,
      missingChildCount: Math.max(0, (patch.lineage_child_count ?? 0) - (visibleChildren.get(patch.patch_id) ?? 0)),
    };
  });
  return { laneCount, rowsByPatchId };
}

export function lineageDescription(row: LineageGraphRow) {
  const parts = [row.parentPatchId ? `Sequential edit from ${row.parentPatchId}` : "Independent edit root"];
  if (row.missingParentId) parts.push("Parent is outside the loaded results");
  if (row.missingChildCount) parts.push(`${row.missingChildCount} child connection(s) outside the loaded results`);
  return parts.join(". ");
}
