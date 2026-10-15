import assert from "node:assert/strict";
import test from "node:test";
import React, { type DependencyList } from "react";

import * as factDataset from "../src/lib/factDataset";
import type { EvaluationFact } from "../src/lib/types";
import { useUploadedEvaluationFacts } from "../src/lib/useUploadedEvaluationFacts";

type Options = Parameters<typeof useUploadedEvaluationFacts>[0];
type Source = ReturnType<typeof useUploadedEvaluationFacts>;
type Hook = { value?: unknown; deps?: DependencyList; cleanup?: () => void };

const DEFAULT_OPTIONS: Options = { modelId: "model-a", backendUrl: "http://backend-a", sessionId: "tab-a", method: "" };

function fact(key: string, fields: Partial<EvaluationFact> = {}): EvaluationFact {
  return {
    key, subject: "Paris", relation_prompt: "{} is in", prompt: "Paris is in",
    new_object: "Germany", ground_truth: "France", rephrase_prompt: "Where is Paris?",
    locality_prompt: "The sun is", locality_ground_truth: "hot",
    portability_prompt: "Where is the Eiffel Tower?", portability_ground_truth: "Germany",
    ...fields,
  };
}

/** Exercise the real hook lifecycle while substituting only its browser/storage boundaries. */
async function withUploadedSource(check: (harness: {
  source: () => Source;
  render: (updates?: Partial<Options>) => void;
  remount: (updates?: Partial<Options>) => void;
  flush: () => Promise<void>;
  datasets: Map<string, unknown[]>;
  descriptors: Map<string, string>;
  failWrite: () => void;
  failDescriptor: () => void;
  pauseReads: () => () => void;
  pauseWrites: () => () => void;
}) => Promise<void>) {
  let hooks: Hook[] = [];
  let cursor = 0;
  let dirty = false;
  let effects: Array<() => void> = [];
  let options = { ...DEFAULT_OPTIONS };
  let source: Source;
  const takeHook = () => hooks[cursor] ?? (hooks[cursor] = {});
  const nextHook = () => { const hook = takeHook(); cursor += 1; return hook; };
  const changed = (previous: DependencyList | undefined, next: DependencyList | undefined) =>
    !previous || !next || previous.length !== next.length || next.some((value, index) => !Object.is(value, previous[index]));
  const originalHooks = {
    useState: React.useState, useMemo: React.useMemo,
    useCallback: React.useCallback, useEffect: React.useEffect,
  };
  Object.assign(React, {
    useState: (initialValue: unknown) => {
      const hook = nextHook();
      if (!("value" in hook)) hook.value = typeof initialValue === "function" ? initialValue() : initialValue;
      return [hook.value, (value: unknown) => {
        const next = typeof value === "function" ? value(hook.value) : value;
        if (!Object.is(next, hook.value)) { hook.value = next; dirty = true; }
      }];
    },
    useMemo: (compute: () => unknown, deps: DependencyList) => {
      const hook = nextHook();
      if (changed(hook.deps, deps)) { hook.value = compute(); hook.deps = deps; }
      return hook.value;
    },
    useCallback: (callback: unknown, deps: DependencyList) => {
      const hook = nextHook();
      if (changed(hook.deps, deps)) { hook.value = callback; hook.deps = deps; }
      return hook.value;
    },
    useEffect: (effect: () => void | (() => void), deps?: DependencyList) => {
      const hook = nextHook();
      if (changed(hook.deps, deps)) {
        hook.deps = deps;
        effects.push(() => { hook.cleanup?.(); hook.cleanup = effect() || undefined; });
      }
    },
  });

  const datasets = new Map<string, unknown[]>();
  const descriptors = new Map<string, string>();
  let writeFailure = false;
  let descriptorFailure = false;
  let readBarrier: Promise<void> | null = null;
  let writeBarrier: Promise<void> | null = null;
  const originalStorage = { readAllDatasetValues: factDataset.readAllDatasetValues, replaceDatasetValues: factDataset.replaceDatasetValues };
  Object.assign(factDataset, {
    readAllDatasetValues: async (datasetId: string) => {
      if (readBarrier) await readBarrier;
      return structuredClone(datasets.get(datasetId) ?? []);
    },
    replaceDatasetValues: async (datasetId: string, values: unknown[]) => {
      if (writeBarrier) await writeBarrier;
      if (writeFailure) { writeFailure = false; throw new Error("IndexedDB quota exceeded"); }
      datasets.set(datasetId, structuredClone(values));
      return { id: datasetId, count: values.length, updatedAt: Date.now() };
    },
  });
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  let nextId = 0;
  Object.defineProperty(globalThis, "window", { configurable: true, value: {
    crypto: { randomUUID: () => `upload-${++nextId}` },
    sessionStorage: {
      getItem: (key: string) => descriptors.get(key) ?? null,
      setItem: (key: string, value: string) => {
        if (descriptorFailure) { descriptorFailure = false; throw new Error("sessionStorage quota exceeded"); }
        descriptors.set(key, value);
      },
      removeItem: (key: string) => descriptors.delete(key),
    },
  } });

  const unmount = () => { for (const hook of hooks) hook.cleanup?.(); };
  const render = (updates: Partial<Options> = {}) => {
    options = { ...options, ...updates };
    let commits = 0;
    do {
      assert.ok(++commits < 20, "The uploaded source should settle after committing its effects.");
      dirty = false;
      cursor = 0;
      effects = [];
      source = useUploadedEvaluationFacts(options);
      for (const effect of effects) effect();
    } while (dirty);
  };
  const remount = (updates: Partial<Options> = {}) => { unmount(); hooks = []; render(updates); };
  const flush = async () => { await new Promise<void>((resolve) => setImmediate(resolve)); render(); };
  const pause = (kind: "read" | "write") => {
    let resolve: () => void = () => undefined;
    const barrier = new Promise<void>((done) => { resolve = done; });
    if (kind === "read") readBarrier = barrier;
    else writeBarrier = barrier;
    return () => {
      if (kind === "read") readBarrier = null;
      else writeBarrier = null;
      resolve();
    };
  };

  try {
    render();
    await check({ source: () => source, render, remount, flush, datasets, descriptors,
      failWrite: () => { writeFailure = true; }, failDescriptor: () => { descriptorFailure = true; },
      pauseReads: () => pause("read"), pauseWrites: () => pause("write"),
    });
  } finally {
    unmount();
    Object.assign(React, originalHooks);
    Object.assign(factDataset, originalStorage);
    if (originalWindow) Object.defineProperty(globalThis, "window", originalWindow);
    else Reflect.deleteProperty(globalThis, "window");
  }
}

test("uploaded facts and separate rephrase overrides survive refresh without altering original inputs", async () => {
  await withUploadedSource(async ({ source, flush, remount, descriptors }) => {
    assert.equal(source().ready, false);
    await flush();
    assert.equal(source().ready, true);
    assert.equal(source().uploaded, null);
    assert.equal(source().method, "rome");

    const original = fact("one");
    await source().importFacts([original], "facts.json");
    await flush();
    const descriptor = source().uploaded;
    const sourceKey = source().sourceKey;
    assert.equal(source().total, 1);
    assert.deepEqual(source().getItem(0), original);
    assert.equal(source().getItem(1), undefined);
    assert.equal(source().localityPromptCount, 1);
    assert.equal(source().portabilityPromptCount, 1);
    assert.deepEqual(JSON.parse([...descriptors.values()][0]), descriptor);
    assert.deepEqual(Object.keys(JSON.parse([...descriptors.values()][0])).sort(), ["fileName", "id"]);

    await source().saveOverride(original.key, "Where can Paris be found?");
    await flush();
    assert.equal(source().getItem(0)?.rephrase_override, "Where can Paris be found?");
    assert.equal(source().getItem(0)?.rephrase_prompt, original.rephrase_prompt);
    assert.deepEqual(source().inputFacts, [original]);
    remount();
    assert.equal(source().ready, false);
    await flush();
    assert.equal(source().ready, true);
    assert.deepEqual(source().uploaded, descriptor);
    assert.equal(source().sourceKey, sourceKey);
    assert.equal(source().overrides.one, "Where can Paris be found?");
    assert.deepEqual(source().inputFacts, [original]);
  });
});

test("new uploads overwrite one fixed snapshot, reset overrides, and search only the three fact fields", async () => {
  await withUploadedSource(async ({ source, flush, datasets }) => {
    await flush();
    await source().importFacts([fact("one")], "old.json");
    await flush();
    const previousId = source().uploaded?.id;
    await source().saveOverride("one", "An old manual rephrase");
    await source().importFacts([
      fact("one", { subject: "MATCH subject", ground_truth: "MATCH original", new_object: "MATCH target", locality_prompt: "", portability_prompt: "" }),
      fact("two", { subject: "Second", ground_truth: "Match original" }),
      fact("three", { subject: "Third", prompt: "match only the prompt", rephrase_prompt: "match only the rephrase" }),
    ], "replacement.json");
    await flush();
    assert.notEqual(source().uploaded?.id, previousId);
    assert.equal(datasets.size, 1);
    assert.equal([...datasets.values()][0].length, 1);
    assert.equal(source().total, 3);
    assert.deepEqual(source().overrides, {});
    assert.equal(source().getItem(0)?.rephrase_override, undefined);
    assert.deepEqual(source().findMatches("  match  "), [0, 1]);
    assert.deepEqual(source().findMatches(""), []);
    assert.equal(source().localityPromptCount, 2);
    assert.equal(source().portabilityPromptCount, 2);
  });
});

test("failed dataset or descriptor writes preserve the previous source, including refresh restoration", async () => {
  await withUploadedSource(async ({ source, flush, failWrite, failDescriptor, remount }) => {
    await flush();
    await source().importFacts([fact("one")], "original.json");
    await flush();
    const descriptor = source().uploaded;
    failWrite();
    await assert.rejects(source().importFacts([fact("two")], "failed.json"), /IndexedDB quota/);
    await flush();
    assert.deepEqual(source().uploaded, descriptor);
    assert.equal(source().getItem(0)?.key, "one");
    failDescriptor();
    await assert.rejects(source().importFacts([fact("three")], "also-failed.json"), /sessionStorage quota/);
    await flush();
    assert.deepEqual(source().uploaded, descriptor);
    remount();
    await flush();
    assert.deepEqual(source().uploaded, descriptor);
    assert.equal(source().getItem(0)?.key, "one");
  });
});

test("clear empties persisted facts and overrides and can recover from a missing saved snapshot", async () => {
  await withUploadedSource(async ({ source, flush, remount, datasets, descriptors }) => {
    await flush();
    await source().importFacts([fact("one")], "facts.json");
    await source().saveOverride("one", "Manual rephrase");
    await flush();
    await source().clear();
    await flush();
    assert.equal(source().uploaded, null);
    assert.equal(source().total, 0);
    assert.deepEqual(source().overrides, {});
    assert.equal(descriptors.size, 0);
    assert.deepEqual([...datasets.values()][0], []);
    remount();
    await flush();
    assert.equal(source().ready, true);
    assert.equal(source().uploaded, null);

    await source().importFacts([fact("one")], "lost.json");
    await flush();
    datasets.clear();
    remount();
    await flush();
    assert.equal(source().ready, false);
    assert.match(source().error, /unavailable/);
    await source().clear();
    await flush();
    assert.equal(source().ready, true);
    assert.equal(source().error, "");
    assert.equal(descriptors.size, 0);
  });
});

test("stale asynchronous restore and upload completions cannot replace the newly selected model or backend", async () => {
  await withUploadedSource(async ({ source, flush, render, remount, pauseReads, pauseWrites }) => {
    await flush();
    await source().importFacts([fact("one")], "model-a.json");
    await flush();
    const releaseRead = pauseReads();
    remount();
    await flush();
    assert.equal(source().ready, false);
    render({ modelId: "model-b", backendUrl: "http://backend-b", method: "memit" });
    await flush();
    assert.equal(source().ready, true);
    assert.equal(source().uploaded, null);
    releaseRead();
    await flush();
    assert.equal(source().uploaded, null);
    assert.equal(source().method, "memit");

    const releaseWrite = pauseWrites();
    const pendingUpload = source().importFacts([fact("two")], "model-b.json");
    await flush();
    render({ modelId: "model-c" });
    await flush();
    releaseWrite();
    await pendingUpload;
    await flush();
    assert.equal(source().ready, true);
    assert.equal(source().uploaded, null);
    assert.equal(source().total, 0);
    render({ modelId: "model-b" });
    await flush();
    assert.equal(source().uploaded?.fileName, "model-b.json");
    assert.equal(source().getItem(0)?.key, "two");
  });
});
