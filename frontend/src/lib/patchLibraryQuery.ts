import type { PatchLibraryQuery } from "./types";

export function patchLibraryParams(query: PatchLibraryQuery = {}, offset?: number) {
  const params = new URLSearchParams();
  if (offset !== undefined) { params.set("offset", String(offset)); params.set("limit", "50"); }
  if (query.query?.trim()) params.set("query", query.query.trim());
  if (query.method) params.set("method", query.method);
  if (query.model_id) params.set("model_id", query.model_id);
  if (offset !== undefined && query.sort) params.set("sort", query.sort);
  return params.toString();
}
