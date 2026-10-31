import assert from "node:assert/strict";
import test from "node:test";
import { api, apiReadVersion, invalidateApiReads } from "../src/lib/api";
import { saveBackendConnection } from "../src/lib/backendConnection";

test("patch search and selection encode the same filters without mutating read state", async () => {
  const original = globalThis.fetch;
  const urls: URL[] = [];
  globalThis.fetch = async (input) => { urls.push(new URL(String(input))); return new Response(JSON.stringify({ items: [], total: 0, patch_ids: [], blocked_patch_count: 0 })); };
  try {
    const query = { query: "  city & 100%_  ", model_id: "org/model", method: "rome", sort: "oldest" as const };
    const before = apiReadVersion();
    await api.patchPage(50, query);
    await api.patchSelection(query);
    assert.equal(apiReadVersion(), before);
    for (const url of urls) {
      assert.equal(url.searchParams.get("query"), "city & 100%_");
      assert.equal(url.searchParams.get("model_id"), "org/model");
      assert.equal(url.searchParams.get("method"), "rome");
    }
    assert.equal(urls[0].searchParams.get("offset"), "50");
    assert.equal(urls[0].searchParams.get("sort"), "oldest");
    assert.equal(urls[1].searchParams.has("offset"), false);
  } finally { globalThis.fetch = original; invalidateApiReads(); }
});

test("additional edit methods send complete batches and DPO rejected answers", async () => {
  const original = globalThis.fetch;
  const calls: { url: string; body: unknown }[] = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), body: JSON.parse(String(init?.body)) });
    return new Response(JSON.stringify({ job_id: "test" }), { status: 200 });
  };
  try {
    const fact = { subject: "Paris", relation_prompt: "{} is in", new_object: "Italy", target_neg: "France" };
    const body = { model_id: "llama-7b", edits: [fact, fact, fact], batch_size: 2, evaluation_prompts: [] };
    for (const method of ["emmet", "memit_fe", "pmet", "eamet", "alphaedit", "sphere"] as const) {
      await api.batchWeightEdit(method, body);
      assert.ok(calls.at(-1)?.url.endsWith(`/api/edit/${method.replaceAll("_", "-")}`));
      assert.deepEqual(calls.at(-1)?.body, body);
    }
    await api.trainingEdit("dpo", body);
    assert.ok(calls.at(-1)?.url.endsWith("/api/edit/dpo"));
    assert.deepEqual(calls.at(-1)?.body, body);
  } finally { globalThis.fetch = original; invalidateApiReads(); }
});

test("shared GETs coalesce and mutations invalidate their snapshots", async () => {
  const original = globalThis.fetch;
  const calls: string[] = [];
  globalThis.fetch = async (input) => {
    calls.push(String(input));
    return new Response(JSON.stringify([]), { status: 200 });
  };
  try {
    invalidateApiReads();
    await Promise.all([api.modelCache(), api.modelCache(), api.modelCache()]);
    assert.equal(calls.length, 1);
    await api.modelCache();
    assert.equal(calls.length, 1);
    await api.clearModelDownloadJobs();
    await api.modelCache();
    assert.equal(calls.length, 3);
  } finally { globalThis.fetch = original; invalidateApiReads(); }
});

test("a failed shared request can be retried", async () => {
  const original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return calls === 1 ? new Response("", { status: 503 }) : new Response("[]");
  };
  try {
    invalidateApiReads();
    await assert.rejects(api.models());
    assert.deepEqual(await api.models(), []);
    assert.equal(calls, 2);
  } finally { globalThis.fetch = original; invalidateApiReads(); }
});

test("deletion previews are read-only and committing sends the reviewed token", async () => {
  const original = globalThis.fetch;
  const calls: Array<{ url: string; body: unknown }> = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), body: JSON.parse(String(init?.body)) });
    return new Response(JSON.stringify({ preview_token: "reviewed", deleted_patch_ids: ["patch_a", "patch_b"] }));
  };
  try {
    const version = apiReadVersion();
    const plan = await api.previewPatchDeletion(["patch_a"]);
    assert.equal(apiReadVersion(), version);
    await api.deletePatchBatch(["patch_a"], plan.preview_token);
    assert.ok(apiReadVersion() > version);
    assert.ok(calls[0].url.endsWith("/api/patches/deletion-preview"));
    assert.deepEqual(calls[1].body, { patch_ids: ["patch_a"], preview_token: "reviewed" });
  } finally { globalThis.fetch = original; invalidateApiReads(); }
});

test("backend switching rejects old responses and scopes cached reads", async () => {
  const original = globalThis.fetch;
  const stored = new Map<string, string>();
  const priorWindow = globalThis.window;
  Object.defineProperty(globalThis, "window", { configurable: true, value: {
    localStorage: { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => stored.set(key, value), removeItem: (key: string) => stored.delete(key) }
  } });
  let resolve!: (response: Response) => void;
  globalThis.fetch = () => new Promise<Response>((done) => { resolve = done; });
  try {
    invalidateApiReads();
    saveBackendConnection({ baseUrl: "http://one.test", token: "" });
    const old = api.models();
    saveBackendConnection({ baseUrl: "http://two.test", token: "" });
    resolve(new Response("[]"));
    await assert.rejects(old, /connection changed/);
    globalThis.fetch = async (url) => {
      assert.equal(String(url), "http://two.test/api/models");
      return new Response("[]");
    };
    await api.models();
  } finally {
    globalThis.fetch = original;
    Object.defineProperty(globalThis, "window", { configurable: true, value: priorWindow });
    invalidateApiReads();
  }
});

test("job consumers share one connection and unsubscribe independently", async () => {
  const original = globalThis.fetch;
  let calls = 0;
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  let aborted = false;
  globalThis.fetch = async (_input, init) => {
    calls += 1;
    init?.signal?.addEventListener("abort", () => { aborted = true; controller.close(); });
    return new Response(new ReadableStream<Uint8Array>({ start(value) { controller = value; } }));
  };
  const first: string[] = [], second: string[] = [];
  const stopFirst = api.subscribeToEvents("/api/edit/jobs/test/events?include_result=false", { onEvent: (event) => first.push(event) });
  const stopSecond = api.subscribeToEvents("/api/edit/jobs/test/events?include_result=false", { onEvent: (event) => second.push(event) });
  try {
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(calls, 1);
    controller.enqueue(new TextEncoder().encode("event: job\ndata: {}\n\n"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(first, ["job"]);
    assert.deepEqual(second, ["job"]);
    stopFirst();
    assert.equal(aborted, false);
    stopSecond();
    assert.equal(aborted, true);
  } finally { stopFirst(); stopSecond(); globalThis.fetch = original; }
});


test("late job subscribers receive accumulated logs without a second stream", async () => {
  const original = globalThis.fetch;
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  let calls = 0;
  globalThis.fetch = async (_input, init) => {
    calls += 1;
    init?.signal?.addEventListener("abort", () => controller.close());
    return new Response(new ReadableStream<Uint8Array>({ start(value) { controller = value; } }));
  };
  const path = "/api/edit/jobs/replay/events?include_result=false";
  const stop = api.subscribeToEvents(path, { onEvent: () => undefined });
  let stopLate = () => {};
  try {
    await new Promise((resolve) => setTimeout(resolve, 0));
    for (let i = 0; i < 2; i++) {
      controller.enqueue(new TextEncoder().encode(`event: job\ndata: ${JSON.stringify({ logs: [String(i)], log_start_count: i, log_count: i + 1 })}\n\n`));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const received: string[][] = [];
    stopLate = api.subscribeToEvents(path, { onEvent: (_event, data) => received.push(JSON.parse(data).logs) });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(calls, 1);
    assert.deepEqual(received, [["0", "1"]]);
  } finally { stop(); stopLate(); globalThis.fetch = original; }
});
