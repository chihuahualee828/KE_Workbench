import assert from "node:assert/strict";
import test from "node:test";
import React from "react";
import type { DependencyList } from "react";
import { api } from "../src/lib/api";
import { usePatchLibrary } from "../src/lib/usePatchLibrary";
import type { Patch, PatchLibraryPage, PatchLibraryQuery } from "../src/lib/types";

type Hook = { value?: unknown; deps?: DependencyList; cleanup?: () => void };
type Controller = ReturnType<typeof usePatchLibrary>;
const patch = (id: string) => ({ patch_id: id }) as Patch;
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

async function withLibrary(check: (h: {
  get: () => Controller;
  query: (query: PatchLibraryQuery) => Controller;
  refresh: () => Controller;
}) => Promise<void>) {
  const hooks: Hook[] = [];
  let cursor = 0, dirty = false;
  let effects: Array<() => void> = [];
  let query: PatchLibraryQuery = {}, source: Patch[] = [];
  const next = () => { const index = cursor++; return hooks[index] ?? (hooks[index] = {}); };
  const changed = (old: DependencyList | undefined, deps: DependencyList | undefined) =>
    !old || !deps || old.length !== deps.length || deps.some((item, index) => !Object.is(item, old[index]));
  const original = { useState: React.useState, useRef: React.useRef, useEffect: React.useEffect, useCallback: React.useCallback };
  const originalPage = api.patchPage;
  Object.assign(React, {
    useState(initial: unknown) {
      const hook = next();
      if (!("value" in hook)) hook.value = typeof initial === "function" ? initial() : initial;
      return [hook.value, (update: unknown) => {
        const value = typeof update === "function" ? update(hook.value) : update;
        if (!Object.is(value, hook.value)) { dirty = true; hook.value = value; }
      }];
    },
    useRef(initial: unknown) { const hook = next(); if (!("value" in hook)) hook.value = { current: initial }; return hook.value; },
    useCallback(fn: unknown, deps: DependencyList) { const hook = next(); if (changed(hook.deps, deps)) { hook.value = fn; hook.deps = deps; } return hook.value; },
    useEffect(effect: () => void | (() => void), deps?: DependencyList) {
      const hook = next();
      if (changed(hook.deps, deps)) { hook.deps = deps; effects.push(() => { hook.cleanup?.(); hook.cleanup = effect() || undefined; }); }
    },
  });
  const render = () => {
    let value!: Controller, iterations = 0;
    do {
      assert.ok(++iterations < 20);
      cursor = 0; dirty = false; effects = [];
      value = usePatchLibrary(query, source);
      effects.forEach((effect) => effect());
    } while (dirty);
    return value;
  };
  try { await check({ get: render, query: (nextQuery) => { query = nextQuery; return render(); }, refresh: () => { source = [...source]; return render(); } }); }
  finally { hooks.forEach((hook) => hook.cleanup?.()); Object.assign(React, original); api.patchPage = originalPage; }
}

test("library requests preserve filters and order across pages and refresh the retained range", async () => {
  await withLibrary(async (h) => {
    const calls: Array<{ offset: number; query: PatchLibraryQuery }> = [];
    api.patchPage = async (offset = 0, query = {}) => {
      calls.push({ offset, query });
      return { items: Array.from({ length: offset ? 3 : 50 }, (_, i) => patch(String(offset + i))), total: 53 };
    };
    const filters: PatchLibraryQuery = { query: "fact", model_id: "model", method: "rome", sort: "oldest" };
    h.query(filters); await tick();
    const first = h.get();
    assert.equal(first.items.length, 50);
    await Promise.all([first.loadMore(), first.loadMore()]);
    assert.equal(h.get().items.length, 53);
    assert.equal(h.get().hasMore, false);
    assert.deepEqual(calls.map((call) => call.offset), [0, 50]);
    assert.ok(calls.every((call) => JSON.stringify(call.query) === JSON.stringify(filters)));
    h.refresh(); await tick();
    assert.equal(h.get().items.length, 53);
    assert.deepEqual(calls.map((call) => call.offset), [0, 50, 0, 50]);
  });
});

test("switching filters aborts old requests and ignores their late responses", async () => {
  await withLibrary(async (h) => {
    const requests: Array<{ signal?: AbortSignal; finish: (page: PatchLibraryPage) => void }> = [];
    api.patchPage = (_offset, _query, signal) => new Promise((finish) => requests.push({ signal, finish }));
    h.query({ query: "first" });
    h.query({ query: "second" });
    assert.equal(requests[0].signal?.aborted, true);
    requests[1].finish({ items: [patch("second")], total: 1 }); await tick();
    requests[0].finish({ items: [patch("first")], total: 1 }); await tick();
    assert.deepEqual(h.get().items.map((item) => item.patch_id), ["second"]);
  });
});

test("an old load-more response cannot append to a different sort or query", async () => {
  await withLibrary(async (h) => {
    let finish!: (page: PatchLibraryPage) => void;
    api.patchPage = async (offset, query) => offset
      ? new Promise((resolve) => { finish = resolve; })
      : { items: [patch(query?.sort ?? "newest")], total: 2 };
    h.get(); await tick();
    const pending = h.get().loadMore();
    h.query({ sort: "oldest" }); await tick();
    finish({ items: [patch("old page")], total: 2 }); await pending;
    assert.deepEqual(h.get().items.map((item) => item.patch_id), ["oldest"]);
  });
});

test("failed page requests keep previous rows and allow retry without skipping offsets", async () => {
  await withLibrary(async (h) => {
    let fail = true;
    api.patchPage = async (offset = 0) => {
      if (offset && fail) throw new Error("503 offline");
      return { items: [patch(String(offset))], total: 2 };
    };
    h.get(); await tick();
    await h.get().loadMore();
    assert.match(h.get().error, /offline/);
    assert.deepEqual(h.get().items.map((item) => item.patch_id), ["0"]);
    fail = false; await h.get().loadMore();
    assert.equal(h.get().error, "");
    assert.deepEqual(h.get().items.map((item) => item.patch_id), ["0", "1"]);
  });
});
