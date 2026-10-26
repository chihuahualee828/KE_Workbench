import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { EditJobProgress } from "../src/components/EditJobProgress";
import type { EditJob } from "../src/lib/types";

const job: EditJob = { job_id: "job_test", experiment_id: "exp_test", status: "running", progress: 0.1, logs: [], log_count: 0, log_start_count: 0 };
function render(values: Partial<EditJob>) {
  return renderToStaticMarkup(createElement(EditJobProgress, { job: { ...job, ...values } }));
}

test("batch progress shows the active batch and only counts completed batches", () => {
  const first = render({ progress_phase: "batch_editing", progress_current: 0, progress_total: 3 });
  assert.match(first, /Batch 1 \/ 3/);
  assert.match(first, /0 completed \(0%\)/);
  const next = render({ progress_phase: "batch_editing", progress_current: 1, progress_total: 3 });
  assert.match(next, /Batch 2 \/ 3/);
  assert.match(next, /1 completed \(33%\)/);
  assert.doesNotMatch(next, /facts|sequential edits|indeterminate/);
});

test("batch completion does not claim the whole edit job has finished", () => {
  const partial = render({ progress_phase: "batch_completed", progress_current: 1, progress_total: 3 });
  assert.match(partial, /Batch 1 \/ 3/);
  for (const total of [1, 3]) {
    const final = render({ progress_phase: "batch_completed", progress_current: total, progress_total: total });
    assert.match(final, /Finalizing edit/);
    assert.ok(final.includes(`${total} / ${total} batches edited`));
  }
  assert.equal(render({ status: "completed" }), "");
  assert.equal(render({ status: "canceled" }), "");
});

test("ROME series and preparation keep their existing progress labels", () => {
  assert.match(render({ progress_phase: "editing", progress_current: 1, progress_total: 3 }), /1 \/ 3 facts/);
  assert.match(render({ progress: 0.05 }), /Preparing/);
  assert.match(render({ progress_phase: "post_edit_evaluation", progress_current: 1, progress_total: 3 }), /Post-edit evaluation/);
});
