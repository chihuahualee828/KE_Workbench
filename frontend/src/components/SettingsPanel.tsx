"use client";

import { Database, Folder, KeyRound, Loader2, RefreshCw, RotateCcw, Save, Settings, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { getDefaultBackendUrl, type BackendConnection } from "@/lib/backendConnection";
import { announceExperimentStarted } from "@/lib/experimentEvents";
import type { AppSettings, EditJob, ModelRuntimeStatus, SupportedModel } from "@/lib/types";
import { usePersistentState } from "@/lib/usePersistentState";
import { refreshTransfers } from "@/lib/workbenchResources";
import { AccentThemeSettings } from "./AccentTheme";
import { ApiStatus } from "./DataState";
import { SecondMomentCacheDialog } from "./SecondMomentCacheDialog";

export function SettingsPanel({
  models,
  runtime,
  backend,
  apiStatus,
  onBackendChanged
}: {
  models: SupportedModel[];
  runtime: ModelRuntimeStatus[];
  backend: BackendConnection;
  apiStatus: "loading" | "ready" | "offline";
  onBackendChanged: (connection: BackendConnection) => Promise<void>;
}) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [hfToken, setHfToken] = useState("");
  const [backendUrl, setBackendUrl] = useState(backend.baseUrl);
  const [backendToken, setBackendToken] = useState(backend.token);
  const [modelExportPath, setModelExportPath] = usePersistentState("kew:settings:modelExportPath", "");
  const [busy, setBusy] = useState(false);
  const [backendBusy, setBackendBusy] = useState(false);
  const [error, setError] = useState("");
  const [cacheDirectory, setCacheDirectory] = useState("");
  const [cacheBusy, setCacheBusy] = useState(false);
  const [cacheError, setCacheError] = useState("");
  const [cacheSaved, setCacheSaved] = useState(false);

  useEffect(() => {
    setBackendUrl(backend.baseUrl);
    setBackendToken(backend.token);
  }, [backend]);

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

  async function saveCacheDirectory(value: string) {
    setCacheError("");
    setCacheSaved(false);
    setCacheBusy(true);
    try {
      const location = await api.setModelCacheLocation({ download_dir: value });
      setSettings((current) => current ? { ...current, model_cache: location } : current);
      setCacheDirectory(location.download_dir);
      setCacheSaved(true);
      void refreshTransfers().catch(() => undefined);
    } catch (err) {
      setCacheError(err instanceof Error ? err.message : String(err));
    } finally {
      setCacheBusy(false);
    }
  }

  async function saveBackend() {
    setError("");
    setBackendBusy(true);
    try {
      await onBackendChanged({ baseUrl: backendUrl, token: backendToken });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBackendBusy(false);
    }
  }

  async function useDefaultBackend() {
    setBackendUrl(getDefaultBackendUrl());
    setError("");
    setBackendBusy(true);
    try {
      await onBackendChanged({ baseUrl: "", token: backendToken });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBackendBusy(false);
    }
  }

  useEffect(() => {
    let active = true;
    setSettings(null);
    setError("");
    setCacheError("");
    setCacheSaved(false);
    if (apiStatus === "ready") {
      api.settings().then((value) => {
        if (active) setSettings(value);
      }).catch((err) => {
        if (active) setError(err instanceof Error ? err.message : String(err));
      });
    }
    return () => { active = false; };
  }, [apiStatus, backend.baseUrl, backend.token]);

  useEffect(() => {
    setCacheDirectory(settings?.model_cache.download_dir ?? "");
  }, [settings?.model_cache.download_dir]);

  return (
    <div className="grid gap-4">
      <AccentThemeSettings />
      <section className="panel grid gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Settings size={17} />
            <h2 className="text-base font-semibold">Backend Connection</h2>
          </div>
          <ApiStatus status={apiStatus} />
        </div>
        <p className="text-xs leading-5 text-ink/60">Choose the backend used by every workspace request. Changes take effect immediately; no frontend rebuild is required.</p>
        <label className="grid gap-1">
          <span className="label">Backend URL</span>
          <input
            className="control"
            type="url"
            inputMode="url"
            autoComplete="url"
            value={backendUrl}
            placeholder="http://localhost:8000"
            onChange={(event) => setBackendUrl(event.target.value)}
          />
        </label>
        <label className="grid gap-1">
          <span className="label">API Token <span className="normal-case text-ink/45">(optional)</span></span>
          <input
            className="control"
            type="password"
            autoComplete="off"
            value={backendToken}
            placeholder="Sent as Authorization: Bearer …"
            onChange={(event) => setBackendToken(event.target.value)}
          />
        </label>
        <div className="flex flex-wrap justify-end gap-2">
          <button className="ghost-button min-h-9 px-3 py-1" type="button" onClick={useDefaultBackend} disabled={backendBusy} title="Use the NEXT_PUBLIC_API_BASE fallback">
            Use build default
          </button>
          <button className="tool-button min-h-9 px-3 py-1" type="button" onClick={saveBackend} disabled={backendBusy}>
            {backendBusy ? <Loader2 className="animate-spin" size={16} /> : <Settings size={16} />}
            Connect
          </button>
        </div>
        <p className="font-mono text-[11px] text-ink/50">Active backend: {backend.baseUrl}</p>
      </section>
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

      <section className="panel grid min-w-0 gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Folder size={17} />
            <h2 className="text-base font-semibold">Model Cache Location</h2>
          </div>
          <span className="text-xs text-ink/60">{settings ? settings.model_cache.source === "settings" ? "Custom folder" : "Backend default" : "Unavailable"}</span>
        </div>
        <label className="grid min-w-0 gap-1">
          <span className="label" title="Absolute path on the backend machine. In Docker, use a path inside the container, such as /hf-cache.">Backend Download Folder</span>
          <input
            className="control min-w-0 font-mono text-sm"
            value={cacheDirectory}
            placeholder={settings?.model_cache.environment_dir ?? "/absolute/path/to/model-cache"}
            onChange={(event) => { setCacheDirectory(event.target.value); setCacheSaved(false); setCacheError(""); }}
            disabled={apiStatus !== "ready" || !settings || cacheBusy}
            autoComplete="off"
            spellCheck={false}
          />
        </label>
        <div className="flex flex-wrap justify-end gap-2">
          <button
            className="ghost-button min-h-9 px-3 py-1"
            type="button"
            onClick={() => saveCacheDirectory("")}
            disabled={apiStatus !== "ready" || !settings || cacheBusy || settings.model_cache.source !== "settings"}
            title={`Restore the backend default: ${settings?.model_cache.environment_dir ?? "HF_HUB_CACHE"}`}
          >
            <RotateCcw size={16} />
            Reset to Default
          </button>
          <button
            className="tool-button min-h-9 px-3 py-1"
            type="button"
            onClick={() => saveCacheDirectory(cacheDirectory)}
            disabled={apiStatus !== "ready" || !settings || cacheBusy || !cacheDirectory.trim() || cacheDirectory.trim() === settings.model_cache.download_dir}
          >
            {cacheBusy ? <Loader2 className="animate-spin" size={16} /> : <Save size={16} />}
            Apply
          </button>
        </div>
        {settings ? <div className="grid min-w-0 gap-1">
          <span className="label">Cache Search Folders</span>
          <ul className="max-h-32 overflow-y-auto text-xs text-ink/60">
            {settings.model_cache.search_dirs.map((directory) => <li className="break-all py-1 font-mono" key={directory}>{directory}</li>)}
          </ul>
        </div> : null}
        {cacheError ? <p className="break-words text-sm text-rust" role="alert">{cacheError}</p> : null}
        {cacheSaved ? <p className="text-xs text-ink/60" role="status">Download folder saved.</p> : null}
      </section>

      <Mom2StatisticsPanel models={models} runtime={runtime} />

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
        <p className="mt-2 text-xs text-ink/60">Used by patch exports. Full model ZIP exports use the browser save picker.</p>
      </section>

      {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
    </div>
  );
}

function Mom2StatisticsPanel({ models, runtime }: { models: SupportedModel[]; runtime: ModelRuntimeStatus[] }) {
  const [method, setMethod] = usePersistentState<"rome" | "r_rome" | "memit">("kew:settings:mom2:method", "memit");
  const [modelId, setModelId] = usePersistentState("kew:settings:mom2:modelId", "gpt2-xl");
  const [layers, setLayers] = useState("");
  const [dataset, setDataset] = useState("wikipedia");
  const [mom2StatsSource, setMom2StatsSource] = useState<"base" | "model">("model");
  const [status, setStatus] = useState<{
    ready: boolean;
    missing_layers: number[];
    dataset?: string;
    stats_source: "base" | "model";
    available_stats_sources: Array<"base" | "model">;
    base_model_id?: string | null;
  } | null>(null);
  const [job, setJob] = useState<EditJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const importInputRef = useRef<HTMLInputElement>(null);
  const [cacheOpen, setCacheOpen] = useState(false);
  const editableModels = models.filter((model) =>
    method === "rome" ? model.supports_rome : method === "r_rome" ? model.supports_r_rome : model.supports_memit
  );
  const hubEditableModels = editableModels.filter((model) => model.source !== "local");
  const importedEditableModels = editableModels.filter((model) => model.source === "local");
  const selectedModel = editableModels.find((model) => model.id === modelId);
  const maxLayer = Math.max(0, (selectedModel?.n_layers ?? 1) - 1);
  const selectedLayers = parseLayers(layers, maxLayer);
  const modelLoaded = runtime.some((entry) => entry.model_id === modelId && entry.loaded);
  const importedModelHasBaseStats = selectedModel?.source === "local" && Boolean(selectedModel.base_model_id);
  const canPrepareSelectedSource = selectedModel?.source !== "local" || mom2StatsSource === "model";

  useEffect(() => {
    if (editableModels.length && !editableModels.some((model) => model.id === modelId)) {
      setModelId(editableModels[0].id);
    }
  }, [editableModels, modelId, setModelId]);

  useEffect(() => {
    setMom2StatsSource(importedModelHasBaseStats ? "base" : "model");
  }, [importedModelHasBaseStats, modelId]);

  useEffect(() => {
    let active = true;
    if (!selectedModel) return;
    api.editDefaultLayers({ model_id: selectedModel.id, method })
      .then((response) => {
        if (!active) return;
        setLayers(response.layers.join(","));
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      active = false;
    };
  }, [method, selectedModel?.id]);

  useEffect(() => {
    let active = true;
    if (!selectedLayers?.length) {
      setStatus(null);
      return;
    }
    setStatus(null);
    api.mom2StatisticsStatus({ model_id: modelId, method, layers: selectedLayers })
      .then((response) => {
        if (!active) return;
        setStatus(response);
        setDataset(response.dataset);
        setMom2StatsSource(response.stats_source);
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      active = false;
    };
  }, [layers, method, modelId, selectedLayers?.join(",")]);

  async function selectMom2StatsSource(source: "base" | "model") {
    if (busy || source === mom2StatsSource) return;
    setBusy(true);
    setError("");
    try {
      const selection = await api.setMom2StatisticsSource({ model_id: modelId, stats_source: source });
      setMom2StatsSource(selection.stats_source);
      if (selectedLayers?.length) {
        const refreshed = await api.mom2StatisticsStatus({
          model_id: modelId,
          method,
          layers: selectedLayers,
          stats_source: selection.stats_source
        });
        setStatus(refreshed);
        setDataset(refreshed.dataset);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function prepare() {
    if (!modelLoaded || !selectedLayers?.length || busy || !canPrepareSelectedSource) return;
    setBusy(true);
    setError("");
    try {
      let current = await api.prepareMom2Statistics({
        model_id: modelId,
        method,
        layers: selectedLayers,
        stats_source: mom2StatsSource
      });
      announceExperimentStarted(current.experiment_id);
      setJob(current);
      while (current.status === "running") {
        current = await api.waitEditJob(current.job_id, current.logs.length, 60);
        setJob(current);
      }
      if (current.status === "failed") {
        throw new Error(current.logs[current.logs.length - 1] ?? "MOM2 statistics preparation failed.");
      }
      const refreshed = await api.mom2StatisticsStatus({
        model_id: modelId,
        method,
        layers: selectedLayers,
        stats_source: mom2StatsSource
      });
      setStatus(refreshed);
      setDataset(refreshed.dataset);
      setMom2StatsSource(refreshed.stats_source);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function importFiles(files: FileList | null) {
    const selectedFiles = files ? Array.from(files) : [];
    if (!selectedLayers?.length || !selectedFiles.length || busy) return;
    setBusy(true);
    setError("");
    try {
      const response = await api.importMom2Statistics({
        model_id: modelId,
        method,
        dataset,
        layers: selectedLayers,
        files: selectedFiles,
        stats_source: mom2StatsSource
      });
      setStatus(response);
      setDataset(response.dataset);
      setMom2StatsSource(response.stats_source);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel grid gap-3">
      <div className="flex items-center gap-2">
        <Database size={17} />
        <h2 className="text-base font-semibold">Second-Moment Statistics</h2>
      </div>
      <div className="grid gap-2 md:grid-cols-5">
        <label className="grid gap-1">
          <span className="label">Method</span>
          <select className="control" value={method} onChange={(event) => setMethod(event.target.value as "rome" | "r_rome" | "memit") }>
            <option value="rome">ROME</option>
            <option value="r_rome">r-ROME</option>
            <option value="memit">MEMIT</option>
          </select>
        </label>
        <label className="grid gap-1">
          <span className="label">Model</span>
          <select className="control" value={modelId} onChange={(event) => setModelId(event.target.value)}>
            {hubEditableModels.length ? (
              <optgroup label="Hugging Face models">
                {hubEditableModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </optgroup>
            ) : null}
            {importedEditableModels.length ? (
              <optgroup label="Imported models">
                {importedEditableModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </optgroup>
            ) : null}
          </select>
        </label>
        <label className="grid gap-1">
          <span className="label">Layers</span>
          <input className="control" inputMode="numeric" value={layers} onChange={(event) => setLayers(event.target.value)} />
        </label>
        <label className="grid gap-1">
          <span className="label">Dataset</span>
          <select className="control" value={dataset} onChange={(event) => setDataset(event.target.value)}>
            <option value="wikipedia">Wikipedia</option>
          </select>
        </label>
        {importedModelHasBaseStats ? (
          <label className="grid gap-1">
            <span className="label">Statistics Source</span>
            <select className="control" disabled={busy} value={mom2StatsSource} onChange={(event) => void selectMom2StatsSource(event.target.value as "base" | "model")}>
              <option value="base">Base model ({selectedModel?.base_model_id})</option>
              <option value="model">This imported model</option>
            </select>
          </label>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-ink/60">
        <span>Valid layers: 0-{maxLayer}</span>
        <span>{status ? status.ready ? "Statistics ready" : `Missing layers: ${status.missing_layers.join(", ")}` : "Checking cache..."}</span>
      </div>
      <div className="flex flex-wrap justify-end gap-2">
        <button className="ghost-button min-h-9 px-3 py-1" onClick={() => setCacheOpen(true)} title="View and manage cached second-moment statistics" type="button">
          <Database size={16} />
          Manage Cache
        </button>
        <input
          ref={importInputRef}
          className="sr-only"
          type="file"
          accept=".npz,application/octet-stream"
          multiple
          onChange={(event) => {
            void importFiles(event.target.files);
            event.target.value = "";
          }}
        />
        <button
          type="button"
          className="ghost-button min-h-9 px-3 py-1"
          onClick={() => importInputRef.current?.click()}
          disabled={busy || !selectedLayers?.length}
          title="Import precomputed second-moment statistics"
        >
          <Upload size={16} />
          Import NPZ
        </button>
        <button className="tool-button min-h-9 px-3 py-1" onClick={prepare} disabled={busy || !modelLoaded || !selectedLayers?.length || !canPrepareSelectedSource} title={!modelLoaded ? "Load the selected model before preparing statistics" : !canPrepareSelectedSource ? "Select this imported model's statistics before computing from its weights" : "Compute missing second-moment covariance statistics"}>
          {busy ? <Loader2 className="animate-spin" size={16} /> : <Database size={16} />}
          {busy ? "Preparing" : status?.ready ? "Statistics Ready" : "Prepare Statistics"}
        </button>
      </div>
      {job ? <div className="font-mono text-xs text-ink/60">{job.status === "running" ? "Preparing statistics. Progress is available in Jobs." : `Preparation ${job.status}.`}</div> : null}
      {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      {cacheOpen ? <SecondMomentCacheDialog modelId={modelId} method={method} statsSource={mom2StatsSource} onClose={() => setCacheOpen(false)} /> : null}
    </section>
  );
}

function parseLayers(value: string, maxLayer: number): number[] | null {
  const values = value.split(",").map((entry) => entry.trim()).filter(Boolean);
  if (!values.length) return null;
  const layers = values.map(Number);
  if (layers.some((layer) => !Number.isInteger(layer) || layer < 0 || layer > maxLayer)) return null;
  return [...new Set(layers)];
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
