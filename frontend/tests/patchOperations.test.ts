import assert from "node:assert/strict";
import test from "node:test";
import { patchDeletionBlockReason, patchMutationBlockReason } from "../src/lib/patchOperations";
import type { Experiment, Patch } from "../src/lib/types";

test("editing and both evaluation workflows block patch mutations", () => {
  for (const type of ["rome_edit", "r_rome_edit", "rome_sequential_edits", "r_rome_sequential_edits", "kn_edit", "memit_edit", "evaluation", "prepare_mom2_statistics"]) {
    assert.ok(patchMutationBlockReason([{ type, status: "running" } as Experiment]), type);
  }
});

test("finished history does not block patch mutations", () => {
  assert.equal(patchMutationBlockReason([]), "");
  for (const status of ["completed", "failed", "canceled"]) {
    assert.equal(patchMutationBlockReason([{ type: "evaluation", status } as Experiment]), "");
  }
});

test("library deletion stays available for unrelated patches while jobs block loading", () => {
  const patches = [
    { patch_id: "old", active: false },
    { patch_id: "current", active: true, parent_patch_ids: ["parent"] },
    { patch_id: "parent", active: false, parent_patch_ids: ["grandparent"] },
  ] as Patch[];
  assert.ok(patchMutationBlockReason([{ type: "evaluation", status: "running" } as Experiment]));
  assert.equal(patchDeletionBlockReason("old", patches), "");
  for (const id of ["current", "parent", "grandparent"]) {
    assert.ok(patchDeletionBlockReason(id, patches), id);
  }
});

test("deletion protection handles cyclic lineage and updates when patches are discarded", () => {
  const patches = [
    { patch_id: "a", active: true, parent_patch_ids: ["b"] },
    { patch_id: "b", active: false, parent_patch_ids: ["a"] },
  ] as Patch[];
  assert.ok(patchDeletionBlockReason("b", patches));
  patches[0].active = false;
  assert.equal(patchDeletionBlockReason("b", patches), "");
});
