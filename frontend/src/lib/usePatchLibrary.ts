"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Patch, PatchLibraryPage, PatchLibraryQuery } from "./types";

const EMPTY: Patch[] = [];
type State = PatchLibraryPage & { key: string; offset: number; loading: boolean; error: string };

/** Query results are separate from the global active-lineage resource. */
export function usePatchLibrary(query: PatchLibraryQuery, refreshSource: Patch[]) {
  const key = JSON.stringify(query);
  const [revision, setRevision] = useState(0);
  const [state, setState] = useState<State>({ key: "", items: EMPTY, total: 0, offset: 0, loading: true, error: "" });
  const retained = useRef({ key: "", offset: 0 });
  const session = useRef<{ controller: AbortController; pending: boolean } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const current = { controller, pending: true };
    session.current = current;
    const target = retained.current.key === key ? retained.current.offset : 0;
    retained.current = { key, offset: target };
    setState((old) => ({ ...old, key, items: old.key === key ? old.items : EMPTY,
      total: old.key === key ? old.total : 0, offset: 0, loading: true, error: "" }));
    void (async () => {
      try {
        const filters: PatchLibraryQuery = JSON.parse(key);
        const first = await api.patchPage(0, filters, controller.signal);
        let items = first.items, offset = items.length;
        while (!controller.signal.aborted && offset < Math.min(target, first.total)) {
          const page = await api.patchPage(offset, filters, controller.signal);
          if (!page.items.length) break;
          items = [...items, ...page.items]; offset += page.items.length;
        }
        if (controller.signal.aborted) return;
        retained.current = { key, offset };
        setState({ ...first, items: unique(items), key, offset, loading: false, error: "" });
      } catch (error) {
        if (!controller.signal.aborted) setState((old) => ({ ...old, loading: false, error: String(error instanceof Error ? error.message : error) }));
      } finally { current.pending = false; }
    })();
    return () => { controller.abort(); if (session.current === current) session.current = null; };
  }, [key, refreshSource, revision]);

  const loadMore = useCallback(async () => {
    const current = session.current;
    if (!current || current.pending || current.controller.signal.aborted || state.key !== key || state.offset >= state.total) return;
    current.pending = true;
    setState((old) => ({ ...old, loading: true, error: "" }));
    try {
      const page = await api.patchPage(state.offset, JSON.parse(key), current.controller.signal);
      if (current.controller.signal.aborted) return;
      const offset = page.items.length ? state.offset + page.items.length : page.total;
      retained.current = { key, offset };
      setState((old) => ({ ...page, key, items: unique([...old.items, ...page.items]), offset, loading: false, error: "" }));
    } catch (error) {
      if (!current.controller.signal.aborted) setState((old) => ({ ...old, loading: false, error: String(error instanceof Error ? error.message : error) }));
    } finally { current.pending = false; }
  }, [key, state.key, state.offset, state.total]);

  const same = state.key === key;
  return { ...state, items: same ? state.items : EMPTY, total: same ? state.total : 0,
    loading: !same || state.loading, error: same ? state.error : "",
    hasMore: same && state.offset < state.total, loadMore, retry: () => setRevision((value) => value + 1) };
}

function unique(items: Patch[]) {
  return [...new Map(items.map((patch) => [patch.patch_id, patch])).values()];
}
