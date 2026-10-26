import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { EditSeedField } from "../src/components/EditSeedField";
import { api } from "../src/lib/api";
import { batchWeightMethods, trainingMethods, editMethods, editSeedOptions, DEFAULT_EDIT_SEED, MAX_EDIT_SEED, parseEditSeed, type EditMethod } from "../src/lib/editingMethods";

test("seed input accepts boundary values and rejects incomplete or invalid numbers", () => {
  assert.equal(DEFAULT_EDIT_SEED, 42);
  for (const seed of [0, 42, MAX_EDIT_SEED]) assert.equal(parseEditSeed(String(seed)), seed);
  for (const value of ["", " ", "-1", "1.5", "NaN", "Infinity", String(MAX_EDIT_SEED + 1)]) {
    assert.throws(() => parseEditSeed(value), /Random seed/);
  }
  const html = renderToStaticMarkup(createElement(EditSeedField, {
    method: "lora", value: "123", onChange: () => {}, disabled: true,
  }));
  assert.match(html, /Random seed/);
  assert.match(html, /value="123"/);
  assert.match(html, /disabled/);
});

test("FT and KN hide the seed control and ignore stale stored values", () => {
  for (const method of Object.keys(editMethods) as EditMethod[]) {
    const html = renderToStaticMarkup(createElement(EditSeedField, {
      method, value: "123", onChange: () => {}, disabled: false,
    }));
    if (method === "ft" || method === "kn") {
      assert.equal(html, "");
      for (const value of ["123", "", "invalid", "-1"]) {
        assert.deepEqual(editSeedOptions(method, value), {});
      }
    } else {
      assert.match(html, /Random seed/);
      assert.deepEqual(editSeedOptions(method, "123"), { seed: 123 });
      assert.throws(() => editSeedOptions(method, "invalid"), /Random seed/);
    }
  }
});

test("editing requests omit the seed for FT/KN and carry it for other methods and series", async () => {
  const original = globalThis.fetch;
  const calls: Array<{ url: string; body: { seed?: number } }> = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), body: JSON.parse(String(init?.body)) });
    return new Response(JSON.stringify({ job_id: "test" }), { status: 200 });
  };
  try {
    const fact = { subject: "Paris", relation_prompt: "{} is in", new_object: "Italy", ground_truth: "France", target_neg: "France" };
    const body = { model_id: "demo", evaluation_prompts: [], ...fact };
    await api.romeEdit({ ...body, ...editSeedOptions("rome", "123") });
    await api.rRomeEdit({ ...body, ...editSeedOptions("r_rome", "123") });
    await api.knEdit({ ...body, ...editSeedOptions("kn", "invalid") });
    const batch = { ...body, edits: [fact, fact], batch_size: 2, target_layer: 0 };
    await api.romeSequentialEdits({ ...batch, ...editSeedOptions("rome", "123") });
    await api.rRomeSequentialEdits({ ...batch, ...editSeedOptions("r_rome", "123") });
    for (const method of Object.keys(batchWeightMethods) as Array<keyof typeof batchWeightMethods>) {
      await api.batchWeightEdit(method, { ...batch, ...editSeedOptions(method, "123") });
    }
    for (const method of Object.keys(trainingMethods) as Array<keyof typeof trainingMethods>) {
      await api.trainingEdit(method, { ...batch, ...editSeedOptions(method, method === "ft" ? "invalid" : "123") });
    }
    assert.equal(calls.length, 15);
    for (const { url, body } of calls) {
      if (/\/edit\/(ft|kn)$/.test(url)) assert.equal(Object.hasOwn(body, "seed"), false);
      else assert.equal(body.seed, 123);
    }
  } finally {
    globalThis.fetch = original;
  }
});
