import assert from "node:assert/strict";
import test from "node:test";
import { patchSelectionOrigins, selectPatchRange } from "../src/lib/patchSelection";
import { protectedPatchIds } from "../src/lib/patchOperations";
import type { Patch } from "../src/lib/types";

const patches = [
  { patch_id: "a", parent_patch_ids: [] }, { patch_id: "b", parent_patch_ids: ["a"] },
  { patch_id: "c", parent_patch_ids: ["b"] }, { patch_id: "d", parent_patch_ids: ["a"] },
  { patch_id: "other", parent_patch_ids: [] }, { patch_id: "join", parent_patch_ids: ["b", "other"] },
] as Patch[];

test("all dependent rows are included, with every selected cause, without selecting parents", () => {
  const origins = patchSelectionOrigins(patches, new Set(["b", "other"]));
  assert.deepEqual([...origins.keys()].sort(), ["b", "c", "join", "other"]);
  assert.deepEqual(origins.get("join"), ["b", "other"]);
  assert.deepEqual(origins.get("c"), ["b"]);
});

test("unselecting an ancestor preserves a separately selected descendant", () => {
  const selected = new Set(["a", "c"]);
  assert.equal(patchSelectionOrigins(patches, selected).size, 5);
  selected.delete("a");
  assert.deepEqual([...patchSelectionOrigins(patches, selected).keys()], ["c"]);
});

test("overlapping branches and cycles never duplicate included rows", () => {
  const cyclic = [...patches, { patch_id: "loop", parent_patch_ids: ["join"] }, { patch_id: "join", parent_patch_ids: ["b", "loop"] }] as Patch[];
  const origins = patchSelectionOrigins(cyclic, new Set(["a", "b"]));
  assert.deepEqual(origins.get("loop"), ["a", "b"]);
  assert.equal(origins.has("other"), false);
});

test("drag range can expand and reverse without retaining rows swept over earlier", () => {
  const ids = ["a", "b", "c", "d", "other"];
  const before = new Set(["other"]);
  assert.deepEqual([...selectPatchRange(ids, before, "b", "d", true, new Set())], ["other", "b", "c", "d"]);
  assert.deepEqual([...selectPatchRange(ids, before, "b", "c", true, new Set())], ["other", "b", "c"]);
  assert.deepEqual([...before], ["other"]);
});

test("range selection skips active ancestors and supports deselection and reverse ordering", () => {
  const blocked = protectedPatchIds(patches.map((patch) => ({ ...patch, active: patch.patch_id === "c" })));
  const ids = patches.map((patch) => patch.patch_id);
  assert.deepEqual([...selectPatchRange(ids, new Set(), "d", "a", true, blocked)], ["d"]);
  assert.deepEqual([...selectPatchRange(ids, new Set(["d", "other"]), "other", "d", false, blocked)], []);
});
