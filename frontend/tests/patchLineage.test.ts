import assert from "node:assert/strict";
import test from "node:test";
import { buildLineageGraph, lineageDescription } from "../src/lib/patchLineage";
import type { Patch } from "../src/lib/types";

const patch = (id: string, parent?: string, children = 0) => ({ patch_id: id,
  sequential_edit: Boolean(parent), parent_patch_ids: parent ? [parent] : [], lineage_child_count: children }) as Patch;

test("lineage connects adjacent endpoints in either chronological direction", () => {
  const chain = [patch("a", undefined, 1), patch("b", "a", 1), patch("c", "b")];
  for (const rows of [chain, [...chain].reverse()]) {
    const { rowsByPatchId: graph } = buildLineageGraph(rows);
    assert.equal(graph[rows[0].patch_id].incomingLanes.length, 0);
    assert.equal(graph[rows[2].patch_id].outgoingLanes.length, 0);
    for (let i = 0; i < 2; i++) {
      assert.deepEqual(graph[rows[i].patch_id].outgoingLanes, graph[rows[i + 1].patch_id].incomingLanes);
    }
    assert.equal(graph.b.incomingConnections.length, 1);
    assert.equal(graph.b.outgoingConnections.length, 1);
  }
});

test("filtering intermediate nodes marks gaps without inventing a direct edge", () => {
  const { rowsByPatchId: graph } = buildLineageGraph([patch("c", "b"), patch("a", undefined, 1)]);
  assert.equal(graph.c.missingParentId, "b");
  assert.equal(graph.a.missingChildCount, 1);
  assert.deepEqual(graph.c.outgoingLanes, []);
  assert.deepEqual(graph.a.incomingLanes, []);
  assert.match(lineageDescription(graph.c), /Parent is outside/);
  const complete = buildLineageGraph([patch("c", "b"), patch("b", "a", 1), patch("a", undefined, 1)]);
  assert.equal(complete.rowsByPatchId.c.missingParentId, null);
  assert.equal(complete.rowsByPatchId.a.missingChildCount, 0);
});

test("branches carry real connections past unrelated rows and tolerate unusual date order", () => {
  for (const rows of [
    [patch("b", "a"), patch("unrelated"), patch("a", undefined, 2), patch("c", "a")],
    [patch("a", undefined, 2), patch("b", "a"), patch("c", "a")],
  ]) {
    const { rowsByPatchId: graph } = buildLineageGraph(rows);
    assert.equal(graph.a.incomingConnections.length + graph.a.outgoingConnections.length, 2);
    assert.equal(graph.a.missingChildCount, 0);
    for (let i = 0; i < rows.length - 1; i++) {
      assert.deepEqual(graph[rows[i].patch_id].outgoingLanes, graph[rows[i + 1].patch_id].incomingLanes);
    }
    if (graph.unrelated) assert.ok(!graph.unrelated.incomingLanes.includes(graph.unrelated.lane));
  }
});
