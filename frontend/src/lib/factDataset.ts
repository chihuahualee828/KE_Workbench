"use client";

import { useCallback, useEffect, useState } from "react";

const DATABASE_NAME = "knowledge-editing-workbench";
const DATABASE_VERSION = 2;
const DATASET_STORE = "fact-datasets";
const FACT_STORE = "fact-dataset-facts";
const OVERRIDE_STORE = "fact-overrides";

export type FactEditDraft = {
  subject: string;
  relation_prompt: string;
  ground_truth: string;
  new_object: string;
  target_neg?: string;
  rephrase_prompts?: string[];
  /** Original imported values that are not edited in the compact fact form. */
  source_metadata?: Record<string, unknown>;
};

type DatasetMeta = {
  id: string;
  count: number;
  updatedAt: number;
};

type StoredFact<T> = {
  datasetId: string;
  index: number;
  value: T;
};

export type IndexedFactDataset<T> = {
  count: number;
  existing: boolean;
  ready: boolean;
  error: string;
  revision: number;
  status: "loading" | "ready" | "error";
  append: (value: T) => Promise<void>;
  clear: () => Promise<void>;
  getAll: () => Promise<T[]>;
  getJson: () => Promise<string>;
  getRange: (start: number, end: number) => Promise<T[]>;
  remove: (index: number) => Promise<void>;
  replace: (values: T[]) => Promise<void>;
  update: (index: number, updateValue: (current: T) => T) => Promise<void>;
};

type UseIndexedFactDatasetOptions<T> = {
  datasetId: string;
  initialValues: T[];
  legacyStorageKey?: string;
  normalize: (value: unknown) => T;
  cacheGroup?: {
    prefix: string;
    maxDatasets: number;
  };
};

export function useIndexedFactDataset<T>({
  datasetId,
  initialValues,
  legacyStorageKey,
  normalize,
  cacheGroup
}: UseIndexedFactDatasetOptions<T>): IndexedFactDataset<T> {
  const [count, setCount] = useState(0);
  const [error, setError] = useState("");
  const [existing, setExisting] = useState(false);
  const [revision, setRevision] = useState(0);
  const [readyDatasetId, setReadyDatasetId] = useState<string | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    let active = true;
    setStatus("loading");
    setReadyDatasetId(null);
    setExisting(false);
    setError("");

    void initializeDataset({ datasetId, initialValues, legacyStorageKey, normalize, cacheGroup })
      .then(({ meta, existing: didExist }) => {
        if (!active) return;
        setCount(meta.count);
        setExisting(didExist);
        setRevision((current) => current + 1);
        setStatus("ready");
        setReadyDatasetId(datasetId);
      })
      .catch((reason) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : String(reason));
        setStatus("error");
      });

    return () => {
      active = false;
    };
  }, [cacheGroup, datasetId, initialValues, legacyStorageKey, normalize]);

  const replace = useCallback(async (values: T[]) => {
    const meta = await replaceDatasetValues(datasetId, values);
    setCount(meta.count);
    setRevision((current) => current + 1);
  }, [datasetId]);

  const append = useCallback(async (value: T) => {
    const meta = await appendDatasetValue(datasetId, value);
    setCount(meta.count);
    setRevision((current) => current + 1);
  }, [datasetId]);

  const remove = useCallback(async (index: number) => {
    const meta = await removeDatasetValue(datasetId, index);
    setCount(meta.count);
    setRevision((current) => current + 1);
  }, [datasetId]);

  const clear = useCallback(async () => {
    const meta = await replaceDatasetValues<T>(datasetId, []);
    setCount(meta.count);
    setRevision((current) => current + 1);
  }, [datasetId]);

  const update = useCallback(async (index: number, updateValue: (current: T) => T) => {
    await updateDatasetValue(datasetId, index, updateValue);
    setRevision((current) => current + 1);
  }, [datasetId]);

  const getRange = useCallback((start: number, end: number) => readDatasetRange<T>(datasetId, start, end), [datasetId]);
  const getAll = useCallback(() => readAllDatasetValues<T>(datasetId), [datasetId]);
  const getJson = useCallback(async () => JSON.stringify(await readAllDatasetValues<T>(datasetId), null, 2), [datasetId]);

  return { append, clear, count, error, existing, getAll, getJson, getRange, ready: status === "ready" && readyDatasetId === datasetId, remove, replace, revision, status, update };
}

async function initializeDataset<T>({
  datasetId,
  initialValues,
  legacyStorageKey,
  normalize,
  cacheGroup
}: UseIndexedFactDatasetOptions<T>): Promise<{ meta: DatasetMeta; existing: boolean }> {
  const existing = await readDatasetMeta(datasetId);
  if (existing) {
    const meta = await touchDatasetMeta(existing);
    await pruneDatasetGroup(cacheGroup, datasetId);
    return { meta, existing: true };
  }

  let values = initialValues;
  let migrated = false;
  const legacyValue = legacyStorageKey ? window.localStorage.getItem(legacyStorageKey) : null;
  if (legacyStorageKey && legacyValue !== null) {
    try {
      const parsed = JSON.parse(legacyValue);
      if (Array.isArray(parsed)) {
        values = parsed.map(normalize);
        migrated = true;
      }
    } catch {
      // Preserve malformed legacy JSON rather than silently deleting it.
    }
  }

  const meta = await replaceDatasetValues(datasetId, values);
  await pruneDatasetGroup(cacheGroup, datasetId);
  if (migrated && legacyStorageKey) window.localStorage.removeItem(legacyStorageKey);
  return { meta, existing: false };
}

async function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") {
    throw new Error("IndexedDB is unavailable in this browser.");
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onerror = () => reject(request.error ?? new Error("Could not open IndexedDB."));
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(OVERRIDE_STORE)) {
        database.createObjectStore(OVERRIDE_STORE, { keyPath: ["datasetId", "key"] });
      }
      if (!database.objectStoreNames.contains(DATASET_STORE)) {
        database.createObjectStore(DATASET_STORE, { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains(FACT_STORE)) {
        const facts = database.createObjectStore(FACT_STORE, { keyPath: ["datasetId", "index"] });
        facts.createIndex("datasetId", "datasetId", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
  });
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed."));
    request.onsuccess = () => resolve(request.result);
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted."));
    transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed."));
    transaction.oncomplete = () => resolve();
  });
}

async function readDatasetMeta(datasetId: string): Promise<DatasetMeta | null> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(DATASET_STORE, "readonly");
    const value = await requestResult(transaction.objectStore(DATASET_STORE).get(datasetId) as IDBRequest<DatasetMeta | undefined>);
    await transactionDone(transaction);
    return value ?? null;
  } finally {
    database.close();
  }
}

async function touchDatasetMeta(meta: DatasetMeta): Promise<DatasetMeta> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(DATASET_STORE, "readwrite");
    const nextMeta = { ...meta, updatedAt: Date.now() };
    transaction.objectStore(DATASET_STORE).put(nextMeta);
    await transactionDone(transaction);
    return nextMeta;
  } finally {
    database.close();
  }
}

async function pruneDatasetGroup(
  cacheGroup: UseIndexedFactDatasetOptions<unknown>["cacheGroup"],
  protectedDatasetId: string
): Promise<void> {
  if (!cacheGroup || cacheGroup.maxDatasets < 1) return;

  const database = await openDatabase();
  try {
    const transaction = database.transaction([DATASET_STORE, FACT_STORE], "readwrite");
    const metaStore = transaction.objectStore(DATASET_STORE);
    const datasets = await requestResult(metaStore.getAll() as IDBRequest<DatasetMeta[]>);
    const staleDatasets = datasets
      .filter((dataset) => dataset.id.startsWith(cacheGroup.prefix) && dataset.id !== protectedDatasetId)
      .sort((left, right) => left.updatedAt - right.updatedAt)
      .reverse()
      .slice(Math.max(0, cacheGroup.maxDatasets - 1));
    const facts = transaction.objectStore(FACT_STORE);
    for (const dataset of staleDatasets) {
      await deleteDatasetFacts(facts, dataset.id);
      metaStore.delete(dataset.id);
    }
    await transactionDone(transaction);
  } finally {
    database.close();
  }
}

export async function replaceDatasetValues<T>(datasetId: string, values: T[]): Promise<DatasetMeta> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction([DATASET_STORE, FACT_STORE], "readwrite");
    const facts = transaction.objectStore(FACT_STORE);
    await deleteDatasetFacts(facts, datasetId);
    values.forEach((value, index) => facts.put({ datasetId, index, value } satisfies StoredFact<T>));
    const meta: DatasetMeta = { id: datasetId, count: values.length, updatedAt: Date.now() };
    transaction.objectStore(DATASET_STORE).put(meta);
    await transactionDone(transaction);
    return meta;
  } finally {
    database.close();
  }
}

async function appendDatasetValue<T>(datasetId: string, value: T): Promise<DatasetMeta> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction([DATASET_STORE, FACT_STORE], "readwrite");
    const metaStore = transaction.objectStore(DATASET_STORE);
    const meta = await requestResult(metaStore.get(datasetId) as IDBRequest<DatasetMeta | undefined>);
    if (!meta) throw new Error("Fact dataset is not initialized.");
    transaction.objectStore(FACT_STORE).put({ datasetId, index: meta.count, value } satisfies StoredFact<T>);
    const nextMeta = { ...meta, count: meta.count + 1, updatedAt: Date.now() };
    metaStore.put(nextMeta);
    await transactionDone(transaction);
    return nextMeta;
  } finally {
    database.close();
  }
}

async function removeDatasetValue(datasetId: string, index: number): Promise<DatasetMeta> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction([DATASET_STORE, FACT_STORE], "readwrite");
    const metaStore = transaction.objectStore(DATASET_STORE);
    const meta = await requestResult(metaStore.get(datasetId) as IDBRequest<DatasetMeta | undefined>);
    if (!meta) throw new Error("Fact dataset is not initialized.");
    if (index < 0 || index >= meta.count) return meta;

    const facts = transaction.objectStore(FACT_STORE);
    const trailing = await requestResult(
      facts.getAll(IDBKeyRange.bound([datasetId, index], [datasetId, Number.MAX_SAFE_INTEGER])) as IDBRequest<StoredFact<unknown>[]>
    );
    trailing.forEach((fact) => facts.delete([datasetId, fact.index]));
    trailing.slice(1).forEach((fact) => facts.put({ ...fact, index: fact.index - 1 }));
    const nextMeta = { ...meta, count: Math.max(0, meta.count - 1), updatedAt: Date.now() };
    metaStore.put(nextMeta);
    await transactionDone(transaction);
    return nextMeta;
  } finally {
    database.close();
  }
}

async function updateDatasetValue<T>(datasetId: string, index: number, updateValue: (current: T) => T): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(FACT_STORE, "readwrite");
    const facts = transaction.objectStore(FACT_STORE);
    const stored = await requestResult(facts.get([datasetId, index]) as IDBRequest<StoredFact<T> | undefined>);
    if (!stored) throw new Error("Fact no longer exists.");
    facts.put({ ...stored, value: updateValue(stored.value) });
    await transactionDone(transaction);
  } finally {
    database.close();
  }
}

async function readDatasetRange<T>(datasetId: string, start: number, end: number): Promise<T[]> {
  if (end <= start) return [];
  const database = await openDatabase();
  try {
    const transaction = database.transaction(FACT_STORE, "readonly");
    const facts = await requestResult(
      transaction.objectStore(FACT_STORE).getAll(
        IDBKeyRange.bound([datasetId, Math.max(0, start)], [datasetId, Math.max(0, end - 1)])
      ) as IDBRequest<StoredFact<T>[]>
    );
    await transactionDone(transaction);
    return facts.sort((left, right) => left.index - right.index).map((fact) => fact.value);
  } finally {
    database.close();
  }
}

export async function readAllDatasetValues<T>(datasetId: string): Promise<T[]> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(FACT_STORE, "readonly");
    const facts = await requestResult(
      transaction.objectStore(FACT_STORE).index("datasetId").getAll(IDBKeyRange.only(datasetId)) as IDBRequest<StoredFact<T>[]>
    );
    await transactionDone(transaction);
    return facts.sort((left, right) => left.index - right.index).map((fact) => fact.value);
  } finally {
    database.close();
  }
}

async function deleteDatasetFacts(store: IDBObjectStore, datasetId: string): Promise<void> {
  const keys = await requestResult(store.index("datasetId").getAllKeys(IDBKeyRange.only(datasetId)));
  keys.forEach((key) => store.delete(key));
}

/** Read only user-authored overrides. Legacy fact caches are traversed one row at
 * a time, rather than materializing the old dataset in memory. */
export async function readFactOverrides(datasetId: string): Promise<Record<string, string>> {
  const database = await openDatabase();
  const values: Record<string, string> = {};
  try {
    const transaction = database.transaction([FACT_STORE, OVERRIDE_STORE], "readonly");
    const done = transactionDone(transaction);
    const scan = (request: IDBRequest<IDBCursorWithValue | null>, legacy: boolean) => new Promise<void>((resolve, reject) => {
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) { resolve(); return; }
        const row = legacy ? cursor.value.value : cursor.value;
        if (row?.key && typeof row.rephrase_override === "string" && (!legacy || row.rephrase_override)) values[row.key] = row.rephrase_override;
        cursor.continue();
      };
    });
    await scan(transaction.objectStore(FACT_STORE).index("datasetId").openCursor(IDBKeyRange.only(datasetId)), true);
    await scan(transaction.objectStore(OVERRIDE_STORE).openCursor(IDBKeyRange.bound([datasetId, ""], [datasetId, "\uffff"])), false);
    await done;
    return values;
  } finally { database.close(); }
}

export async function writeFactOverride(datasetId: string, key: string, value: string) {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(OVERRIDE_STORE, "readwrite");
    transaction.objectStore(OVERRIDE_STORE).put({ datasetId, key, rephrase_override: value });
    await transactionDone(transaction);
  } finally { database.close(); }
}
