import assert from "node:assert/strict";
import test from "node:test";
import { api, invalidateApiReads } from "../src/lib/api";
import { experimentResource, historyResource, loadMoreExperiments, refreshExperimentHistory, resetWorkbenchResources, patchResource, refreshPatchHistory, reconcilePatchesAfterRuntimeUpdate } from "../src/lib/workbenchResources";
import type { Experiment, ModelRuntimeStatus, Patch } from "../src/lib/types";

function experiment(index: number, status: Experiment["status"] = "completed"): Experiment {
  return { id: `exp_${index}`, type: "evaluation", model_id: "demo", status, created_at: String(index) } as Experiment;
}

test("history loads a page and all running jobs, then adds older pages on demand", async () => {
  const original = api.experimentPage;
  const calls: Array<[number, string | undefined]> = [];
  api.experimentPage = async (offset = 0, status) => {
    calls.push([offset, status]);
    return { items: status ? [experiment(99, "running")] : Array.from({ length: 50 }, (_, i) => experiment(offset + i)), total: status ? 1 : 100, offset, limit: 50, status_counts: { completed: 99, running: 1 } };
  };
  try {
    resetWorkbenchResources();
    await Promise.all([refreshExperimentHistory(), refreshExperimentHistory()]);
    assert.equal(calls.length, 2);
    assert.equal(experimentResource.get().length, 51);
    assert.equal(historyResource.get().experiments, 100);
    await loadMoreExperiments();
    assert.equal(calls.length, 3);
    assert.equal(experimentResource.get().length, 100);
    await loadMoreExperiments();
    assert.equal(calls.length, 3);
  } finally { api.experimentPage = original; resetWorkbenchResources(); }
});

test("a resource reset discards pending history from the previous backend", async () => {
  const original = api.experimentPage;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  api.experimentPage = async (offset = 0) => {
    await gate;
    return { items: [experiment(1)], total: 1, offset, limit: 50, status_counts: { completed: 1 } };
  };
  try {
    resetWorkbenchResources();
    const pending = refreshExperimentHistory();
    resetWorkbenchResources();
    release();
    await pending;
    assert.deepEqual(experimentResource.get(), []);
    assert.equal(historyResource.get().experiments, 0);
  } finally { api.experimentPage = original; resetWorkbenchResources(); }
});

test("an in-flight history refresh retries after mutation invalidation", async () => {
  const original = api.experimentPage;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  api.experimentPage = async (offset = 0, status) => {
    const stale = calls++ < 2;
    if (stale) await gate;
    return { items: status ? [] : [experiment(stale ? 1 : 2)], total: status ? 0 : 1, offset, limit: 50, status_counts: { completed: 1 } };
  };
  try {
    resetWorkbenchResources();
    const pending = refreshExperimentHistory();
    invalidateApiReads();
    release();
    await pending;
    assert.equal(calls, 4);
    assert.equal(experimentResource.get()[0].id, "exp_2");
  } finally { api.experimentPage = original; resetWorkbenchResources(); }
});


const runtimeStatus = (loaded: boolean) => ({ model_id: "demo", loaded }) as ModelRuntimeStatus;
const savedPatch = (patch_id: string) => ({ patch_id, base_model: "demo", active: true, active_index: 0 }) as Patch;

test("unload bypasses the pre-cancellation cache and clears old and current series patches", async () => {
  const original = globalThis.fetch;
  let active = true;
  let calls = 0;
  const patches = [savedPatch("previous_run"), savedPatch("current_series")];
  globalThis.fetch = async (input) => {
    calls++;
    const isActive = String(input).includes("/patches/active");
    // Deliberately stale flags in history: the active-stack response wins.
    return new Response(JSON.stringify(isActive ? { patches: active ? patches : [] } : { items: patches, total: 2 }));
  };
  try {
    resetWorkbenchResources(); invalidateApiReads();
    await refreshPatchHistory();
    assert.equal(patchResource.get().filter((patch) => patch.active).length, 2);
    active = false;
    // A general refresh may already have recorded the unloaded runtime state.
    await reconcilePatchesAfterRuntimeUpdate([runtimeStatus(false)], [runtimeStatus(false)]);
    assert.equal(calls, 4);
    assert.equal(patchResource.get().length, 2, "Saved artifacts remain visible");
    assert.ok(patchResource.get().every((patch) => !patch.active && patch.active_index === null));
    await reconcilePatchesAfterRuntimeUpdate([runtimeStatus(false)], [runtimeStatus(false)]);
    assert.equal(calls, 4, "Repeated unloaded snapshots do not keep refetching");
  } finally { globalThis.fetch = original; resetWorkbenchResources(); invalidateApiReads(); }
});

test("unload retries an in-flight patch refresh taken before worker shutdown", async () => {
  const originalPage = api.patchPage, originalActive = api.activePatches;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  const patch = savedPatch("current_series");
  api.patchPage = async () => ({ items: [patch], total: 1 });
  api.activePatches = async () => {
    const stale = calls++ === 0;
    if (stale) await gate;
    return { patches: stale ? [patch] : [] };
  };
  try {
    resetWorkbenchResources();
    const early = refreshPatchHistory();
    const unloaded = reconcilePatchesAfterRuntimeUpdate([runtimeStatus(true)], [runtimeStatus(false)]);
    release();
    await Promise.all([early, unloaded]);
    assert.equal(calls, 2);
    assert.equal(patchResource.get()[0].active, false);
  } finally { api.patchPage = originalPage; api.activePatches = originalActive; resetWorkbenchResources(); }
});

test("runtime reconciliation does not clear loaded models or baked lineage", async () => {
  resetWorkbenchResources();
  const original = api.patchPage;
  api.patchPage = async () => { throw new Error("Unexpected patch refresh"); };
  try {
    patchResource.set([savedPatch("loaded")]);
    await reconcilePatchesAfterRuntimeUpdate([runtimeStatus(true)], [runtimeStatus(true)]);
    assert.equal(patchResource.get()[0].active, true);
    patchResource.set([{ ...savedPatch("baked"), baked_into_model: true }]);
    await reconcilePatchesAfterRuntimeUpdate([runtimeStatus(false)], [runtimeStatus(false)]);
    assert.equal(patchResource.get()[0].active, true);
  } finally { api.patchPage = original; resetWorkbenchResources(); }
});
