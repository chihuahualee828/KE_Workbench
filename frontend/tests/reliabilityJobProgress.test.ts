import assert from "node:assert/strict";
import test from "node:test";
import { api } from "../src/lib/api";
import { watchReliabilityTestJob } from "../src/lib/reliabilityJobProgress";
import { announceExperimentCanceled } from "../src/lib/experimentEvents";
import type { EditJob } from "../src/lib/types";

const initial: EditJob = { job_id: "job_test", experiment_id: "exp_test", status: "running", progress: 0, logs: [], log_count: 0, log_start_count: 0 };

function setup() {
  const originalSubscribe = api.subscribeToEvents;
  const originalFetch = api.easyEditReliabilityJob;
  const originalWindow = globalThis.window;
  Object.defineProperty(globalThis, "window", { configurable: true, value: new EventTarget() });
  let emit!: (event: string, data: string) => void;
  let stopped = 0;
  let requests = 0;
  api.subscribeToEvents = (path, handlers) => {
    assert.equal(path, "/api/edit/jobs/job_test/events?include_result=false");
    emit = handlers.onEvent;
    return () => { stopped++; };
  };
  api.easyEditReliabilityJob = async () => {
    requests++;
    return { ...initial, status: "completed", result: { mean_reliability: 1 } };
  };
  return {
    emit: (job: EditJob) => emit("job", JSON.stringify(job)),
    requests: () => requests,
    stopped: () => stopped,
    restore: () => {
      api.subscribeToEvents = originalSubscribe;
      api.easyEditReliabilityJob = originalFetch;
      Object.defineProperty(globalThis, "window", { configurable: true, value: originalWindow });
    }
  };
}

test("testing uses SSE partial results and fetches the final summary only once", async () => {
  const env = setup();
  try {
    const updates: EditJob[] = [];
    const pending = watchReliabilityTestJob(initial, (job) => updates.push(job));
    env.emit({ ...initial, progress_current: 1, progress_total: 2, result_item_count: 1, result: { mean_reliability: 0.5 } });
    assert.equal(env.requests(), 0);
    assert.equal(updates.at(-1)?.result_item_count, 1);
    assert.equal(updates.at(-1)?.result?.mean_reliability, 0.5);
    env.emit({ ...initial, status: "completed" });
    env.emit({ ...initial, status: "completed" });
    const result = await pending;
    assert.equal(result.result?.mean_reliability, 1);
    assert.equal(env.requests(), 1);
    assert.ok(env.stopped() > 0);
  } finally { env.restore(); }
});

test("testing cancellation stops its listener and ignores late events", async () => {
  const env = setup();
  try {
    const updates: EditJob[] = [];
    const pending = watchReliabilityTestJob(initial, (job) => updates.push(job));
    announceExperimentCanceled({ experimentId: initial.experiment_id, jobId: initial.job_id });
    assert.equal((await pending).status, "canceled");
    const count = updates.length;
    env.emit({ ...initial, status: "completed" });
    assert.equal(updates.length, count);
    assert.equal(env.requests(), 0);
  } finally { env.restore(); }
});

test("testing failure reports the job error without fetching results", async () => {
  const env = setup();
  try {
    const pending = watchReliabilityTestJob(initial, () => {});
    env.emit({ ...initial, status: "failed", logs: ["Inference failed"] });
    await assert.rejects(pending, /Inference failed/);
    assert.equal(env.requests(), 0);
    assert.ok(env.stopped() > 0);
  } finally { env.restore(); }
});
