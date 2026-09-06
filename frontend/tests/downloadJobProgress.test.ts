import assert from "node:assert/strict";
import test from "node:test";

import { downloadJobCache, formatTransferSize, transferProgress } from "../src/lib/downloadJobProgress";
import type { ModelCacheStatus, ModelDownloadJob } from "../src/lib/types";

function job(status: ModelDownloadJob["status"], extra: Partial<ModelDownloadJob> = {}): ModelDownloadJob {
  return { job_id: "old", model_id: "gpt2-xl", hf_id: "gpt2-xl", status, message: "", created_at: "", updated_at: "", ...extra };
}
function cache(size_bytes: number, active_job?: ModelDownloadJob): ModelCacheStatus {
  return { model_id: "gpt2-xl", hf_id: "gpt2-xl", size_bytes, hub_size_bytes: 100, active_job, cached: false, cache_state: "partial", cache_path: "", snapshot_count: 0, has_model_weights: false, has_config: false, incomplete_file_count: 0 };
}

for (const status of ["completed", "failed", "canceled"] as const) {
  test(`${status} history ignores cache deletion and later downloads`, () => {
    const old = job(status, { cache: cache(status === "completed" ? 100 : 40) });
    const expected = status === "completed" ? "100 B / 100 B (100%)" : "40 B / 100 B (40%)";
    for (const bytes of [0, 4, 80, 100]) {
      const live = cache(bytes, job("downloading", { job_id: "new" }));
      assert.equal(downloadJobCache(old, [live]), old.cache);
      assert.equal(formatTransferSize(old, live), expected);
    }
  });
}

test("live progress requires the exact active job ID", () => {
  const active = job("downloading", { cache: cache(10) });
  assert.equal(transferProgress(active, downloadJobCache(active, [cache(25, active)])).downloadedBytes, 25);
  assert.equal(transferProgress(active, downloadJobCache(active, [cache(70, job("downloading", { job_id: "different" }))])).downloadedBytes, 10);
});

test("per-job byte counters retain failed progress, including a real zero", () => {
  for (const bytes of [0, 40]) {
    const old = job("failed", { progress_current_bytes: bytes, progress_total_bytes: 100 });
    assert.equal(transferProgress(old, cache(90)).downloadedBytes, bytes);
    assert.equal(transferProgress(old, cache(90)).known, true);
  }
});

test("legacy history uses its snapshot or reports missing progress honestly", () => {
  const old = job("completed", { cache: { ...cache(100), hub_size_bytes: null } });
  assert.equal(formatTransferSize(old, cache(0)), "100 B / 100 B (100%)");
  assert.equal(formatTransferSize(job("failed"), cache(80)), "Progress unavailable");
  assert.equal(formatTransferSize(job("completed"), cache(80)), "Progress unavailable");
});

test("local imports keep their own uploaded byte counts", () => {
  const imported = job("completed", { job_type: "local_import", progress_current_bytes: 20, progress_total_bytes: 20 });
  assert.equal(formatTransferSize(imported, cache(0)), "20 B / 20 B (100%)");
});
