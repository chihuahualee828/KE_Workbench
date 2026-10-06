import assert from "node:assert/strict";
import test from "node:test";
import { isLoraGroupSelected, toggleLoraGroup, type LoraModuleGroup } from "../src/lib/loraModules";

const group: LoraModuleGroup = { name: "c_proj", targets: ["c_proj"], modules: ["transformer.h.0.attn.c_proj", "transformer.h.0.mlp.c_proj"].map((path) => ({ path, type: "Conv1D", weight_shape: [4, 4], available: true, reason: null })) };

test("selecting a group consolidates paths and preserves unrelated targets", () => {
  assert.equal(toggleLoraGroup(group, "c_attn, transformer.h.0.attn.c_proj", true), "c_attn, c_proj");
  assert.equal(isLoraGroupSelected(group, "c_attn, c_proj"), true);
  assert.equal(isLoraGroupSelected(group, "attn.c_proj"), false);
  assert.equal(toggleLoraGroup(group, "c_attn, c_proj", false), "c_attn");
});

test("exact path suggestions select only supported modules", () => {
  const mixed = { ...group, targets: [group.modules[0].path], modules: [group.modules[0], { ...group.modules[1], available: false, reason: "Tied weights" }] };
  const value = toggleLoraGroup(mixed, "", true);
  assert.equal(value, group.modules[0].path);
  assert.equal(isLoraGroupSelected(mixed, value), true);
  assert.equal(toggleLoraGroup(mixed, value, false), "");
});
