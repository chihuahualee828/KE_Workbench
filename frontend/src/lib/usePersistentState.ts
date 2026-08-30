"use client";

import { useEffect, useState } from "react";

export function usePersistentState<T>(key: string, defaultValue: T) {
  // Keep the server and the first client render identical. Reading localStorage
  // here would make persisted values replace defaults before React hydrates.
  const [value, setValue] = useState<T>(defaultValue);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem(key);
    if (stored === null) {
      setHydrated(true);
      return;
    }
    try {
      setValue(JSON.parse(stored) as T);
    } catch {
      setValue(defaultValue);
    } finally {
      setHydrated(true);
    }
  }, [defaultValue, key]);

  useEffect(() => {
    if (!hydrated) return;
    window.localStorage.setItem(key, JSON.stringify(value));
  }, [hydrated, key, value]);

  return [value, setValue] as const;
}
