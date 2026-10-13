import assert from "node:assert/strict";
import test from "node:test";
import { applyEditPatchState } from "../src/lib/editPatchState";
import type { Patch } from "../src/lib/types";

const patch = (id: string, model = "demo"): Patch => ({
  patch_id: id, base_model: model, method: "rome", modified_layers: [0], edits: [],
  created_at: "2026-10-08T00:00:00Z", active: true, active_index: 0,
});

test("a backend reset immediately clears active lineage without deleting saved patches", () => {
  const before = [patch("old"), patch("another", "other-model")];
  const after = applyEditPatchState(before, { version: 1, model_id: "demo", active_patch_ids: [] });
  assert.equal(after.length, 2);
  assert.equal(after[0].active, false);
  assert.equal(after[0].active_index, null);
  assert.equal(after[1], before[1]);
  assert.equal(before[0].active, true);
});

test("checkpoint notifications preserve ordered parent dependencies", () => {
  const after = applyEditPatchState([patch("child"), patch("parent"), patch("old")], {
    version: 2, model_id: "demo", active_patch_ids: ["parent", "child"],
  });
  assert.equal(after[0].active_index, 1);
  assert.equal(after[1].active_index, 0);
  assert.equal(after[2].active, false);
});
