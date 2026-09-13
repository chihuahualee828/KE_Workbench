import { backendUrl } from "./backendConnection";

export function jobEventsUrl(jobId: string) {
  return backendUrl(`/api/jobs/${jobId}`).replace(/^http/, "ws");
}
