const eventName = "kew:experiment-started";
const cancelEventName = "kew:edit-job-canceling";
const canceledEventName = "kew:experiment-canceled";

export type ExperimentCanceledEvent = {
  experimentId: string;
  jobId: string;
};

export function announceExperimentStarted(experimentId: string) {
  window.dispatchEvent(new CustomEvent<string>(eventName, { detail: experimentId }));
}

export function listenForExperimentStarts(callback: (experimentId: string) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<string>).detail);
  }
  window.addEventListener(eventName, handle);
  return () => window.removeEventListener(eventName, handle);
}

export function announceEditJobCanceling(jobId: string) {
  window.dispatchEvent(new CustomEvent<string>(cancelEventName, { detail: jobId }));
}

export function listenForEditJobCanceling(callback: (jobId: string) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<string>).detail);
  }
  window.addEventListener(cancelEventName, handle);
  return () => window.removeEventListener(cancelEventName, handle);
}

export function announceExperimentCanceled(detail: ExperimentCanceledEvent) {
  window.dispatchEvent(new CustomEvent<ExperimentCanceledEvent>(canceledEventName, { detail }));
}

export function listenForExperimentCanceled(callback: (detail: ExperimentCanceledEvent) => void) {
  function handle(event: Event) {
    callback((event as CustomEvent<ExperimentCanceledEvent>).detail);
  }
  window.addEventListener(canceledEventName, handle);
  return () => window.removeEventListener(canceledEventName, handle);
}
