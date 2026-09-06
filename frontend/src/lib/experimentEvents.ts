const eventName = "kew:experiment-started";

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
