"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { Check, Palette } from "lucide-react";

const themes = [
  { id: "purple", name: "Purple", color: "#5b57d6" },
  { id: "blue", name: "Blue", color: "#2563eb" },
  { id: "teal", name: "Teal", color: "#0f766e" },
  { id: "green", name: "Green", color: "#15803d" },
  { id: "rose", name: "Rose", color: "#be185d" }
] as const;
type Accent = typeof themes[number]["id"];
const storageKey = "kew:appearance:accent";
const validAccent = (value: string | null): Accent => themes.some((theme) => theme.id === value) ? value as Accent : "purple";
const ThemeContext = createContext<{ accent: Accent; select: (accent: Accent) => void; error: string } | null>(null);

export function AccentThemeProvider({ children }: { children: React.ReactNode }) {
  const [accent, setAccent] = useState<Accent>("purple");
  const [error, setError] = useState("");
  function apply(value: Accent) {
    document.documentElement.dataset.accent = value;
    setAccent(value);
  }
  useEffect(() => {
    try { apply(validAccent(localStorage.getItem(storageKey))); } catch { /* Use the default when browser storage is unavailable. */ }
    function sync(event: StorageEvent) {
      if (event.key === storageKey || event.key === null) apply(validAccent(event.newValue));
    }
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);
  function select(value: Accent) {
    apply(value);
    setError("");
    try { localStorage.setItem(storageKey, value); }
    catch { setError("Color applied for this session. Browser storage is unavailable, so it may reset when you reload."); }
  }
  return <ThemeContext.Provider value={{ accent, select, error }}>{children}</ThemeContext.Provider>;
}

export function AccentThemeSettings() {
  const theme = useContext(ThemeContext);
  if (!theme) throw new Error("AccentThemeSettings requires AccentThemeProvider");
  return (
    <section className="panel" aria-labelledby="appearance-title">
      <div className="panel-heading"><h2 id="appearance-title" className="flex items-center gap-2 text-base font-semibold"><Palette size={18} className="text-cyan" />Appearance</h2><button type="button" className="ghost-button min-h-8 px-2 py-1 text-xs" disabled={theme.accent === "purple"} onClick={() => theme.select("purple")}>Reset to default</button></div>
      <fieldset>
        <legend className="text-sm font-medium">Main theme color</legend>
        <p className="mb-4 mt-1 text-xs leading-5 text-ink/60">Personalize buttons, navigation, and highlights. Applies instantly and is saved in this browser.</p>
        <div className="flex flex-wrap gap-3">
          {themes.map((option) => <label key={option.id} className="relative cursor-pointer">
            <input type="radio" name="accent-color" value={option.id} checked={theme.accent === option.id} onChange={() => theme.select(option.id)} className="peer sr-only" />
            <span className="flex min-h-11 items-center gap-2 rounded-lg border border-line px-3 py-2 text-sm peer-checked:border-cyan peer-checked:bg-cyan/5 peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-cyan">
              <span className="flex h-6 w-6 items-center justify-center rounded-full text-white" style={{ backgroundColor: option.color }}>{theme.accent === option.id ? <Check size={14} aria-hidden="true" /> : null}</span>{option.name}
            </span>
          </label>)}
        </div>
      </fieldset>
      <div className="mt-5 flex flex-wrap items-center gap-3 rounded-lg border border-line bg-panel p-3" aria-label="Theme preview"><span className="label">Preview</span><span className="tool-button pointer-events-none">Primary action</span><span className="rounded-lg bg-cyan/10 px-3 py-2 text-xs font-medium text-cyan">Selected item</span></div>
      <p className="mt-3 text-xs text-ink/55" role="status">{theme.error || `${themes.find((option) => option.id === theme.accent)?.name} theme selected.`}</p>
    </section>
  );
}
