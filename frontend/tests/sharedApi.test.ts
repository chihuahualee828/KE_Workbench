import assert from "node:assert/strict";
import test from "node:test";
import { api, invalidateApiReads } from "../src/lib/api";
import { saveBackendConnection } from "../src/lib/backendConnection";

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
