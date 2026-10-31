import assert from "node:assert/strict";
import test from "node:test";
import React from "react";
import type { DependencyList } from "react";
import { api } from "../src/lib/api";
import { usePatchLibrarySelection } from "../src/lib/usePatchLibrarySelection";
import type { Patch, PatchDeletionPreview, PatchLibraryQuery } from "../src/lib/types";

type Controller = ReturnType<typeof usePatchLibrarySelection>;
type Hook = { value?: unknown; deps?: DependencyList; cleanup?: () => void };
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const preview = (ids: string[], token = ids.join(",")): PatchDeletionPreview => ({
  selected_patch_ids: ids, deleted_patch_ids: ids, dependent_patch_ids: [], blocked_patch_ids: [],
  preview_token: token,
  patches: ids.map((id) => ({ patch_id: id, base_model: "demo", active: false, baked_into_model: false, included_by: [] })),
});

// Follow the existing hook harness convention, exercising the real controller
// with delayed API responses and effect cleanup rather than browser dependencies.
async function withSelection(check: (harness: {
  get: () => Controller;
  act: (change: (value: Controller) => void) => Controller;
  wait: () => Promise<Controller>;
  refresh: () => Controller;
  filter: (query: PatchLibraryQuery) => Controller;
  deleted: string[][];
}) => Promise<void>, totalPatchCount?: number) {
  const hooks: Hook[] = [];
  let cursor = 0, dirty = false;
  let effects: Array<() => void> = [];
  const take = () => hooks[cursor] ?? (hooks[cursor] = {});
  const next = () => { const hook = take(); cursor++; return hook; };
  const changed = (old: DependencyList | undefined, value: DependencyList | undefined) =>
    !old || !value || old.length !== value.length || value.some((item, i) => !Object.is(old[i], item));
  const original = { useState: React.useState, useMemo: React.useMemo, useEffect: React.useEffect, useRef: React.useRef };
  const originalPreview = api.previewPatchDeletion, originalDelete = api.deletePatchBatch, originalPatches = api.patchSelection;
  const deleted: string[][] = [];
  Object.assign(React, {
    useRef: (initial: unknown) => {
      const hook = next();
      if (!("value" in hook)) hook.value = { current: initial };
      return hook.value;
    },
    useState: (initial: unknown) => {
      const hook = next();
      if (!("value" in hook)) hook.value = typeof initial === "function" ? initial() : initial;
      return [hook.value, (update: unknown) => {
        const value = typeof update === "function" ? update(hook.value) : update;
        if (!Object.is(value, hook.value)) { hook.value = value; dirty = true; }
      }];
    },
    useMemo: (compute: () => unknown, deps: DependencyList) => {
      const hook = next();
      if (changed(hook.deps, deps)) { hook.value = compute(); hook.deps = deps; }
      return hook.value;
    },
    useEffect: (effect: () => void | (() => void), deps?: DependencyList) => {
      const hook = next();
      if (changed(hook.deps, deps)) { hook.deps = deps; effects.push(() => { hook.cleanup?.(); hook.cleanup = effect() || undefined; }); }
    },
  });
  let patches = [{ patch_id: "a" }, { patch_id: "c", parent_patch_ids: ["missing_middle"] }] as Patch[];
  let query: PatchLibraryQuery = {};
  api.previewPatchDeletion = async (ids) => preview(ids);
  api.patchSelection = async () => ({ patch_ids: patches.map((patch) => patch.patch_id), blocked_patch_count: 0 });
  api.deletePatchBatch = async (ids) => ({ deleted_patch_ids: ids, deleted_patch_count: ids.length, cleanup_pending: false });
  let value: Controller;
  function render() {
    let commits = 0;
    do {
      assert.ok(++commits < 20);
      cursor = 0; dirty = false; effects = [];
      value = usePatchLibrarySelection(patches, async (result) => { deleted.push(result.deleted_patch_ids); }, totalPatchCount, query);
      for (const effect of effects) effect();
    } while (dirty);
    return value;
  }
  render();
  try {
    await check({
      get: render,
      act: (change) => { change(render()); return render(); },
      wait: async () => { await delay(210); return render(); },
      refresh: () => { patches = [...patches]; return render(); }, deleted,
      filter: (next) => { query = next; return render(); },
    });
  } finally {
    for (const hook of hooks) hook.cleanup?.();
    Object.assign(React, original);
    api.previewPatchDeletion = originalPreview; api.deletePatchBatch = originalDelete; api.patchSelection = originalPatches;
  }
}

test("server preview includes descendants across unloaded pages and unselecting clears them", async () => {
  await withSelection(async (h) => {
    api.previewPatchDeletion = async (ids) => ({
      ...preview(ids), deleted_patch_ids: ["a", "missing_middle", "c"], dependent_patch_ids: ["missing_middle", "c"],
      patches: [preview(["a"]).patches[0], ...["missing_middle", "c"].map((id) => ({ ...preview([id]).patches[0], included_by: ["a"] }))],
    });
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["a"])));
    assert.equal(h.get().checking, true);
    assert.equal(h.get().origins.has("c"), false);
    const ready = await h.wait();
    assert.deepEqual(ready.origins.get("c"), ["a"]);
    assert.equal(ready.checking, false);
    h.act((s) => s.setSelected(new Set()));
    assert.equal(h.get().origins.size, 0);
    assert.equal(h.get().plan, undefined);
  });
});

test("select all targets matching IDs and preserves selections outside the new filter", async () => {
  await withSelection(async (h) => {
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["already_selected"])));
    h.filter({ query: "Tokyo", method: "rome", model_id: "demo", sort: "oldest" });
    api.patchSelection = async (query) => {
      assert.deepEqual(query, { query: "Tokyo", method: "rome", model_id: "demo", sort: "oldest" });
      return { patch_ids: ["matching"], blocked_patch_count: 0 };
    };
    await h.get().selectAll();
    assert.deepEqual([...h.get().selected], ["already_selected", "matching"]);
    assert.equal(h.get().allSelected, true);
    h.filter({ query: "Paris" });
    assert.equal(h.get().allSelected, false);
    assert.deepEqual([...h.get().selected], ["already_selected", "matching"]);
  });
});

test("filter changes abort pending select-all before it can change selection", async () => {
  await withSelection(async (h) => {
    let resolve!: (value: { patch_ids: string[]; blocked_patch_count: number }) => void;
    let signal: AbortSignal | undefined;
    api.patchSelection = (_query, requestSignal) => { signal = requestSignal; return new Promise((finish) => { resolve = finish; }); };
    h.act((s) => s.start());
    const pending = h.get().selectAll();
    h.filter({ method: "memit" });
    assert.equal(signal?.aborted, true);
    resolve({ patch_ids: ["old_scope"], blocked_patch_count: 0 }); await pending;
    assert.equal(h.get().selected.size, 0);
    assert.equal(h.get().selectingAll, false);
  });
});

test("select all with no new eligible matches still refreshes the existing deletion preview", async () => {
  await withSelection(async (h) => {
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["a"])));
    await h.wait();
    api.patchSelection = async () => ({ patch_ids: [], blocked_patch_count: 1 });
    await h.get().selectAll();
    h.get(); await h.wait();
    assert.deepEqual(h.get().plan?.selected_patch_ids, ["a"]);
    assert.equal(h.get().checking, false);
  });
});

test("pointer movement never starts a preview request; drop requests the final selection", async () => {
  await withSelection(async (h) => {
    const calls: string[][] = [];
    api.previewPatchDeletion = async (ids) => { calls.push(ids); return preview(ids); };
    h.act((s) => s.start()); h.act((s) => s.setDragging(true));
    h.act((s) => s.setSelected(new Set(["a"])));
    await h.wait();
    h.act((s) => s.setSelected(new Set(["a", "c"])));
    await h.wait();
    assert.deepEqual(calls, []);
    h.act((s) => s.setDragging(false));
    await h.wait();
    assert.deepEqual(calls, [["a", "c"]]);
  });
});

test("an obsolete preview cannot overwrite a newer selection even if it resolves last", async () => {
  await withSelection(async (h) => {
    const pending: Array<(value: PatchDeletionPreview) => void> = [];
    api.previewPatchDeletion = () => new Promise((resolve) => pending.push(resolve));
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["a"])));
    await h.wait();
    h.act((s) => s.setSelected(new Set(["c"])));
    await h.wait();
    pending[1](preview(["c"])); pending[0](preview(["a"]));
    await delay(0);
    assert.deepEqual(h.get().plan?.selected_patch_ids, ["c"]);
  });
});

test("a changed preview requires a new review before a commit", async () => {
  await withSelection(async (h) => {
    let version = 0;
    const commits: string[] = [];
    api.previewPatchDeletion = async (ids) => preview(ids, String(++version));
    api.deletePatchBatch = async (ids, token) => { commits.push(token); return { deleted_patch_ids: ids, deleted_patch_count: ids.length, cleanup_pending: false }; };
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["a"])));
    await h.wait(); h.act((s) => s.setReview(s.plan!));
    h.refresh();
    await h.get().commit();
    await h.wait();
    await h.get().commit();
    assert.deepEqual(commits, []);
    h.act((s) => s.setReview(s.plan!));
    await h.get().commit();
    assert.deepEqual(commits, ["2"]);
    assert.deepEqual(h.deleted, [["a"]]);
    assert.equal(h.get().editing, false);
  });
});

test("a stale commit preserves selection and requires review of refreshed dependencies", async () => {
  await withSelection(async (h) => {
    api.deletePatchBatch = async () => { throw new Error("409 Patch history changed"); };
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["a"])));
    await h.wait(); h.act((s) => s.setReview(s.plan!));
    await h.get().commit();
    assert.deepEqual([...h.get().selected], ["a"]);
    assert.equal(h.get().review, null);
    assert.match(h.get().error, /history changed/);
    assert.deepEqual(h.deleted, []);
    assert.equal(h.get().deleting, false);
  });
});

test("select all includes the full library rather than just loaded rows, then uses the normal review", async () => {
  await withSelection(async (h) => {
    api.patchSelection = async () => ({ patch_ids: ["a", "c", "off_page", "independent"], blocked_patch_count: 0 });
    h.act((s) => s.start());
    await h.get().selectAll();
    assert.deepEqual([...h.get().selected], ["a", "c", "off_page", "independent"]);
    assert.equal(h.get().selectAllSkipped, 0);
    assert.equal(h.get().checking, true);
    await h.wait();
    h.act((s) => s.setReview(s.plan!));
    await h.get().commit();
    assert.deepEqual(h.deleted, [["a", "c", "independent", "off_page"]]);
  });
});

test("select all excludes active patches and transitive prerequisites from unloaded pages", async () => {
  await withSelection(async (h) => {
    api.patchSelection = async () => ({ patch_ids: ["a", "unrelated"], blocked_patch_count: 3 });
    h.act((s) => s.start());
    await h.get().selectAll();
    assert.deepEqual([...h.get().selected], ["a", "unrelated"]);
    assert.equal(h.get().selectAllSkipped, 3);
    h.act((s) => s.setSelected(new Set()));
    assert.equal(h.get().selectAllSkipped, 0);
  });
});

for (const action of ["finish", "clear", "manual"] as const) {
  test(`a pending select-all response cannot overwrite ${action}`, async () => {
    await withSelection(async (h) => {
      let release!: (result: { patch_ids: string[]; blocked_patch_count: number }) => void;
      let signal: AbortSignal | undefined;
      api.patchSelection = (_query, value) => { signal = value; return new Promise((resolve) => { release = resolve; }); };
      h.act((s) => s.start());
      const pending = h.get().selectAll();
      assert.equal(h.get().selectingAll, true);
      assert.equal(h.get().plan, undefined);
      if (action === "finish") h.act((s) => s.finish());
      else h.act((s) => s.setSelected(new Set(action === "manual" ? ["c"] : [])));
      assert.equal(signal?.aborted, true);
      release({ patch_ids: ["a", "off_page"], blocked_patch_count: 0 });
      await pending;
      assert.deepEqual([...h.get().selected], action === "manual" ? ["c"] : []);
      assert.equal(h.get().selectingAll, false);
      assert.equal(h.get().editing, action !== "finish");
    });
  });
}

test("failed select all keeps the selection and retry reloads the full library", async () => {
  await withSelection(async (h) => {
    api.patchSelection = async () => { throw new Error("503 unavailable"); };
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["c"])));
    await h.get().selectAll();
    assert.deepEqual([...h.get().selected], ["c"]);
    assert.match(h.get().error, /503/);
    assert.equal(h.get().selectingAll, false);
    api.patchSelection = async () => ({ patch_ids: ["a", "c", "off_page"], blocked_patch_count: 0 });
    await h.get().retry();
    assert.deepEqual([...h.get().selected].sort(), ["a", "c", "off_page"]);
    assert.equal(h.get().error, "");
  });
});

test("the master checkbox selects all from a partial selection and clears a full selection", async () => {
  await withSelection(async (h) => {
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["a"])));
    assert.equal(h.get().allSelected, false);
    await h.get().toggleAll();
    assert.deepEqual([...h.get().selected], ["a", "c"]);
    assert.equal(h.get().allSelected, true);
    await h.get().toggleAll();
    assert.equal(h.get().selected.size, 0);
    assert.equal(h.get().allSelected, false);
    h.act((s) => s.setSelected(new Set(["a", "c"])));
    assert.equal(h.get().allSelected, true);
  }, 2);
});

test("selecting a loaded page keeps the master checkbox partial until the entire library is selected", async () => {
  await withSelection(async (h) => {
    api.patchSelection = async () => ({ patch_ids: ["a", "c", "off_page_1", "off_page_2"], blocked_patch_count: 0 });
    h.act((s) => s.start()); h.act((s) => s.setSelected(new Set(["a", "c"])));
    assert.equal(h.get().allSelected, false);
    await h.get().toggleAll();
    assert.equal(h.get().selected.size, 4);
    assert.equal(h.get().allSelected, true);
    h.act((s) => s.setSelected(new Set(["a"])));
    assert.equal(h.get().allSelected, false);
  }, 4);
});
