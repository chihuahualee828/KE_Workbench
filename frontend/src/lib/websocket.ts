export function jobEventsUrl(jobId: string) {
  const base = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
  return `${base.replace(/^http/, "ws")}/api/jobs/${jobId}`;
}

