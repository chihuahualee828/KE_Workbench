import { api } from "./api";
import { announceExperimentStarted, listenForExperimentCanceled } from "./experimentEvents";
import type { EditJob } from "./types";

export function watchReliabilityTestJob(initial: EditJob, onUpdate: (job: EditJob) => void): Promise<EditJob> {
  announceExperimentStarted(initial.experiment_id);
  return new Promise((resolve, reject) => {
    let settled = false;
    let finishing = false;
    let current = initial;
    let stopSubscription: () => void = () => undefined;
    let stopCanceledListener: () => void = () => undefined;
    const finish = (job: EditJob, error?: Error) => {
      if (settled) return;
      settled = true;
      stopSubscription();
      stopCanceledListener();
      if (error) reject(error);
      else resolve(job);
    };
    const update = (job: EditJob) => {
      if (settled || finishing) return;
      current = job;
      onUpdate(job);
      if (job.status === "running") return;
      stopSubscription();
      if (job.status === "failed") {
        finish(job, new Error(job.logs[job.logs.length - 1] ?? "Reliability evaluation failed."));
      } else if (job.status === "completed") {
        finishing = true;
        // This endpoint supplies the final summary; result rows stay paginated.
        void api.easyEditReliabilityJob(job.job_id).then((completed) => {
          if (settled) return;
          onUpdate(completed);
          finish(completed);
        }).catch((error) => finish(job, error instanceof Error ? error : new Error(String(error))));
      } else finish(job);
    };
    stopCanceledListener = listenForExperimentCanceled(({ jobId }) => {
      if (jobId !== initial.job_id || settled || finishing) return;
      current = { ...current, status: "canceled" };
      onUpdate(current);
      finish(current);
    });
    update(initial);
    if (settled || finishing) return;
    stopSubscription = api.subscribeToEvents(`/api/edit/jobs/${initial.job_id}/events?include_result=false`, {
      onEvent: (event, data) => {
        if (event !== "job" || settled || finishing) return;
        try { update(JSON.parse(data) as EditJob); }
        catch (error) { finish(current, error instanceof Error ? error : new Error(String(error))); }
      }
    });
  });
}
