"use client";

import { Database, KeyRound, Loader2, RefreshCw, Settings, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { announceExperimentStarted } from "@/lib/experimentEvents";
import type { AppSettings, EditJob, ModelRuntimeStatus, SupportedModel } from "@/lib/types";
import { useLoadedModelSelection } from "@/lib/useLoadedModelSelection";
import { usePersistentState } from "@/lib/usePersistentState";
import { SecondMomentCacheDialog } from "./SecondMomentCacheDialog";

export function SettingsPanel({ models, runtime }: { models: SupportedModel[]; runtime: ModelRuntimeStatus[] }) {
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
  const [method, setMethod] = usePersistentState<"rome" | "memit">("kew:settings:mom2:method", "memit");
  const [modelId, setModelId] = usePersistentState("kew:settings:mom2:modelId", "gpt2-xl");
  const [layers, setLayers] = useState("");
  const [dataset, setDataset] = useState("wikipedia");
  const [status, setStatus] = useState<{ ready: boolean; missing_layers: number[]; dataset?: string } | null>(null);
  const [job, setJob] = useState<EditJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const importInputRef = useRef<HTMLInputElement>(null);
  const [cacheOpen, setCacheOpen] = useState(false);
  const editableModels = models.filter((model) => method === "rome" ? model.supports_rome : model.supports_memit);
  const selectedModel = editableModels.find((model) => model.id === modelId);
  const maxLayer = Math.max(0, (selectedModel?.n_layers ?? 1) - 1);
  const selectedLayers = parseLayers(layers, maxLayer);
  const modelLoaded = runtime.some((entry) => entry.model_id === modelId && entry.loaded);

  useLoadedModelSelection(editableModels, runtime, setModelId);

  useEffect(() => {
    if (editableModels.length && !editableModels.some((model) => model.id === modelId)) {
      setModelId(editableModels[0].id);
    }
  }, [editableModels, modelId, setModelId]);

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
    api.mom2StatisticsStatus({ model_id: modelId, method, layers: selectedLayers })
      .then((response) => {
        if (!active) return;
        setStatus(response);
        setDataset(response.dataset);
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      active = false;
    };
  }, [layers, method, modelId, selectedLayers?.join(",")]);

  async function prepare() {
    if (!modelLoaded || !selectedLayers?.length || busy) return;
    setBusy(true);
    setError("");
    try {
      let current = await api.prepareMom2Statistics({ model_id: modelId, method, layers: selectedLayers });
      announceExperimentStarted(current.experiment_id);
      setJob(current);
      while (current.status === "running") {
        current = await api.waitEditJob(current.job_id, current.logs.length, 60);
        setJob(current);
      }
      if (current.status === "failed") {
        throw new Error(current.logs[current.logs.length - 1] ?? "MOM2 statistics preparation failed.");
      }
      const refreshed = await api.mom2StatisticsStatus({ model_id: modelId, method, layers: selectedLayers });
      setStatus(refreshed);
      setDataset(refreshed.dataset);
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
        files: selectedFiles
      });
      setStatus(response);
      setDataset(response.dataset);
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
      <div className="grid gap-2 md:grid-cols-4">
        <label className="grid gap-1">
          <span className="label">Method</span>
          <select className="control" value={method} onChange={(event) => setMethod(event.target.value as "rome" | "memit") }>
            <option value="rome">ROME</option>
            <option value="memit">MEMIT</option>
          </select>
        </label>
        <label className="grid gap-1">
          <span className="label">Model</span>
          <select className="control" value={modelId} onChange={(event) => setModelId(event.target.value)}>
            {editableModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
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
        <button className="tool-button min-h-9 px-3 py-1" onClick={prepare} disabled={busy || !modelLoaded || !selectedLayers?.length} title={modelLoaded ? "Compute missing second-moment covariance statistics" : "Load the selected model before preparing statistics"}>
          {busy ? <Loader2 className="animate-spin" size={16} /> : <Database size={16} />}
          {busy ? "Preparing" : status?.ready ? "Statistics Ready" : "Prepare Statistics"}
        </button>
      </div>
      {job ? <div className="font-mono text-xs text-ink/60">{job.status === "running" ? "Preparing statistics. Progress is available in Jobs." : `Preparation ${job.status}.`}</div> : null}
      {error ? <div className="rounded-md border border-rust/40 bg-rust/10 p-3 text-sm text-rust">{error}</div> : null}
      {cacheOpen ? <SecondMomentCacheDialog modelId={modelId} method={method} onClose={() => setCacheOpen(false)} /> : null}
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
