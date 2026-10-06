import assert from "node:assert/strict";
import test from "node:test";
import { formatLayerRanges } from "../src/lib/layerRanges";

test("compact runs of three or more layers while keeping pairs explicit", () => {
  assert.equal(formatLayerRanges([0, 1, 2, 3, 4, 5]), "0~5");
  assert.equal(formatLayerRanges([0, 1, 3, 4, 5]), "0, 1, 3~5");
  assert.equal(formatLayerRanges([0, 1, 2, 4, 6, 7, 8]), "0~2, 4, 6~8");
  assert.equal(formatLayerRanges([13, 14, 15, 16, 17]), "13~17");
});

test("handle missing runs, single layers, and layer zero", () => {
  assert.equal(formatLayerRanges([]), "");
  assert.equal(formatLayerRanges([0]), "0");
  assert.equal(formatLayerRanges([17]), "17");
  assert.equal(formatLayerRanges([0, 1]), "0, 1");
  assert.equal(formatLayerRanges([0, 2, 4]), "0, 2, 4");
});

test("do not change the configured layer order or the original array", () => {
  const layers = Object.freeze([5, 4, 3, 0, 1, 2]);
  assert.equal(formatLayerRanges(layers), "5, 4, 3, 0~2");
  assert.deepEqual(layers, [5, 4, 3, 0, 1, 2]);
});
