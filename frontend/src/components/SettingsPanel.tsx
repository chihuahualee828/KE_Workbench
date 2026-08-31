"use client";

import { KeyRound, Loader2, RefreshCw, Settings } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { AppSettings } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";

export function SettingsPanel() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [hfToken, setHfToken] = useState("");
  const [modelExportPath, setModelExportPath] = usePersistentState("kew:settings:modelExportPath", "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    setError("");
    try {
      setSettings(await api.settings());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function saveHfToken() {
    setError("");
    setBusy(true);
    try {
      await api.setHfToken({ token: hfToken });
      setHfToken("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function clearHfToken() {
    setError("");
    setBusy(true);
    try {
      await api.clearHfToken();
      setHfToken("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="grid gap-4">
      <section className="panel">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <KeyRound size={17} />
            <h2 className="text-base font-semibold">Hugging Face Token</h2>
          </div>
          <span className="font-mono text-xs text-ink/60">{formatTokenStatus(settings)}</span>
        </div>
        <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto_auto]">
          <input
            className="control"
            type="password"
            value={hfToken}
            placeholder="hf_..."
            onChange={(event) => setHfToken(event.target.value)}
          />
          <button className="tool-button" onClick={saveHfToken} disabled={busy || !hfToken.trim()} title="Use token for backend downloads">
            {busy ? <Loader2 className="animate-spin" size={16} /> : <KeyRound size={16} />}
            Save
          </button>
          <button className="ghost-button" onClick={clearHfToken} disabled={busy || settings?.hf_token.source !== "runtime"} title="Clear runtime token">
            Clear
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Settings size={17} />
            <h2 className="text-base font-semibold">Runtime Settings</h2>
          </div>
          <button className="ghost-button min-h-9 px-2 py-1" onClick={refresh} title="Refresh settings">
            <RefreshCw size={15} />
            Refresh
          </button>
        </div>
        <div className="grid gap-2 text-sm md:grid-cols-2">
          <SettingRow label="App Env" value={settings?.app_env ?? "loading"} />
          <SettingRow label="Editing Runtime" value={settings?.editing_runtime ?? "loading"} />
          <SettingRow label="EasyEdit Root" value={settings?.easyedit_root ?? "loading"} />
          <SettingRow label="EasyEdit" value={settings ? settings.easyedit_available ? "available" : "missing" : "loading"} />
          <SettingRow label="Data Dir" value={settings?.data_dir ?? "loading"} />
          <SettingRow label="Install EasyEdit" value={settings?.install_easyedit || "unset"} />
        </div>
      </section>

      <section className="panel">
        <div className="mb-3 flex items-center gap-2">
          <Settings size={17} />
          <h2 className="text-base font-semibold">Patch Export</h2>
        </div>
        <label className="grid gap-2">
          <span className="label">Default Server Directory</span>
          <input
            className="control"
            value={modelExportPath}
            placeholder="/home/nick/models"
            onChange={(event) => setModelExportPath(event.target.value)}
          />
        </label>
      </section>

      {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
    </div>
  );
}

function SettingRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-line bg-panel p-3">
      <div className="label mb-1">{label}</div>
      <div className="break-words font-mono text-xs text-ink/70">{value}</div>
    </div>
  );
}

function formatTokenStatus(settings: AppSettings | null) {
  if (!settings) return "loading";
  if (!settings.hf_token.configured) return "none";
  return settings.hf_token.source === "env" ? "env" : "runtime";
}
