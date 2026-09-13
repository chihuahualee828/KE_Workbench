"use client";

import { useEffect, useState } from "react";

/** Persist small, tab-scoped UI state across a browser refresh. */
export function useSessionState<T>(key: string, defaultValue: T) {
  const [value, setValue] = useState<T>(defaultValue);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const stored = window.sessionStorage.getItem(key);
    if (stored !== null) {
      try {
        setValue(JSON.parse(stored) as T);
      } catch {
        setValue(defaultValue);
      }
    }
    setHydrated(true);
  }, [defaultValue, key]);

  useEffect(() => {
    if (!hydrated) return;
    if (value === null || value === undefined) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, JSON.stringify(value));
  }, [hydrated, key, value]);

  return [value, setValue] as const;
}
