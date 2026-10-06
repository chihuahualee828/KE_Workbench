# Knowledge Editing Workbench

A local, full-stack workspace for exploring and editing factual knowledge in Hugging Face language models. Generate text, inspect layer predictions, trace activations, apply EasyEdit knowledge edits, and evaluate the resulting model—all from one interface.

The workbench runs inference against real model weights and integrates **ROME, r-ROME, Knowledge Neurons (KN), MEMIT, FT, and LoRA** through EasyEdit. Edits are tracked as saved patches with an active model lineage, so you can compare states, reload edits, revert changes, and export an edited model.

## What you can do

| Workspace | Current capabilities |
| --- | --- |
| **Overview** | Manage models, generate text, inspect active and saved patches, and browse recent experiments. |
| **Generate** | Run prompts with greedy decoding, sampling, or beam search against the loaded model and its active edits. |
| **Logit lens** | Inspect layer-by-layer next-token predictions, top candidates, and a target token's rank, logit, and probability. |
| **Causal tracing** | Run a token × layer activation-restoration sweep and explore the results in an interactive heatmap. |
| **Knowledge edits** | Apply single-fact edits, sequential ROME/r-ROME series, or MEMIT/FT/LoRA batches; import fact datasets and test edit reliability. |
| **Evaluation** | Evaluate active edit facts for reliability, generalization, and locality; inspect scores and generated answers. |
| **Activity** | Follow experiment history, job progress, logs, and supported cancellation actions. |
| **Settings** | Configure the backend connection, Hugging Face token, patch export directory, and second-moment statistics. |

## Supported features

### Model management and generation

- **Model catalog with capability filtering.** Each workflow offers models configured for that operation. The catalog includes GPT-2 XL, GPT-J, Baichuan 2, ChatGLM/GLM, InternLM, Llama, Mistral, and Qwen variants; editing methods have different compatibility lists.
- **Download and cache management.** Download Hugging Face weights, follow download progress, cancel downloads, inspect cache sizes, and delete managed cached models.
- **Local model registration and import.** Register a path visible to the backend or choose a model folder in the browser. Browser imports validate the root `config.json`, upload the files, and store compatible models in the managed cache. Imported models can inherit editing configuration from a compatible base model.
- **Explicit Load / Unload controls.** Downloaded files and loaded weights are separate states. The backend retains a model and tokenizer in a dedicated worker between requests; unloading stops the worker to release its model memory.
- **Runtime visibility.** Inspect loaded state, device information, and GPU VRAM usage where available.
- **Generation controls.** Choose greedy decoding, sampling, or beam search; configure output length, temperature, top-p, top-k, and beam count as applicable.

Generation and analysis use the current active model state, including loaded workbench patches. Selectors and runtime controls help keep the chosen model consistent across workflows.

### Logit lens and causal tracing

The **logit lens** projects hidden states at the final prompt position into vocabulary space. Its chart and table show how next-token predictions change across layers. An optional target must encode as a single tokenizer token.

The **causal-tracing view** currently exposes residual-stream restoration for supported GPT-2-family models:

1. Enter a prompt, the subject to corrupt, and an expected answer. A subject-suggestion action can help fill the subject field.
2. Configure the number of corrupted samples and the random seed. The UI uses Gaussian embedding noise at scale 3.
3. Run a full trace across token positions and Transformer layers.
4. Pan, zoom, hover, click cells, or drag the selection marker to inspect the completed results. Keyboard controls support selection, zoom, and fitting the map.

The detail panel shows clean, corrupted, and restored probabilities, the indirect effect, and recovery relative to the clean–corrupted gap. Probabilities refer to the **first token of the expected answer**. Recovery can fall outside 0–100%, and is undefined when the clean–corrupted gap is effectively zero.

Each intervention restores the clean activation from the **same token and layer** being evaluated. Moving the marker browses already evaluated cells; it does not run inference or transport an activation between locations. Hooks are removed after each intervention.

**Causal tracing is analysis only.** It does not create a weight edit or automatically choose a ROME edit layer. ROME uses its EasyEdit hyperparameters unless an explicit layer override is supplied.

### Knowledge editing

| Method | Editing workflow |
| --- | --- |
| **ROME** | Single-fact edits or a sequential series of facts. |
| **r-ROME** | EasyEdit's rebuilding-ROME implementation, with single-fact and sequential-series workflows. |
| **Knowledge Neurons (KN)** | Single-fact editing with a required original answer (`ground_truth`). |
| **MEMIT** | Batch editing of multiple facts, with configured or explicitly selected layers. |
| **FT** | Fine-tune configured layers, with a configurable batch size. |
| **LoRA** | Train low-rank adapters in batches and save reversible merged weight deltas. |

Editing controls include method-compatible model selection, layer configuration where supported, and the option to continue from the active patch lineage. ROME/r-ROME sequential series accumulate their edits in one cumulative patch.

MEMIT exposes a **Batch Size** setting (default `1`): each batch jointly edits up to that many facts through EasyEdit's `batch_edit()`. Batches accumulate within the job and are saved as one patch; **Continue from active patches** controls only the starting model state. Larger batches can change edit quality and memory requirements. Cancellation rolls back the entire MEMIT job, including completed batches.

FT and LoRA use EasyEdit's `FTHyperParams` / `LoRAHyperParams` and `BaseEditor.batch_edit()` on the already loaded model. Both expose **Batch Size** (default `1`) and edit layers in the same batch form. Batches accumulate into one reversible patch; cancellation restores the pre-job weights and active patches. FT preserves all changed parameters, including biases. LoRA uses standard LoRA (overriding upstream profiles that select AdaLoRA), then merges the adapters into a saved weight delta so patch replay and model export use the existing workflow. Expand **Training hyperparameters** to adjust learning rate, training steps, and weight decay. FT also exposes its objective and norm constraint; LoRA exposes target modules, rank, alpha, and dropout. The **Target modules** dropdown inspects the loaded model on first opening and groups supported projections by name. Search and check multiple groups to select targets, or expand a group to inspect full paths, module types, and weight shapes. The dropdown stays open while selecting; Done, Escape, or clicking outside closes it. Tied weights are shown as unavailable; discovery does not run inference or load a second model. Values start from the model's YAML, persist separately per model/method in the browser, and can be reset to defaults. Each saved patch records the effective settings. LoRA accepts an empty layer field (`[]`), preserving EasyEdit's no-layer-restriction behavior; target modules still determine which projections receive adapters. Other training settings come from the model's EasyEdit YAML. Model choices are limited to configured method-specific profiles; inherited local models use their base profile.

For sequential series and MEMIT/FT/LoRA batches, the fact editor supports:

- Editable fact cards and a JSON editor.
- JSON file import with attribute mapping and automatic matching of common field aliases.
- Optional rephrase, locality, and portability prompt metadata, while retaining original imported values in patch metadata.
- Browser-backed dataset storage and virtualized lists for larger collections.
- Reliability testing on the current model, with per-fact progress and generated answers.

**Second-moment statistics** are managed separately. The workbench can inspect, prepare, import, and delete cached statistics. ROME/r-ROME expose a second-moment adjustment option; MEMIT requires it. Compatible imported models can choose statistics associated with their base model or the imported model itself.

### Patches, history, and exports

The workbench distinguishes the **Active Model Lineage** from the **Saved Patch Library**. Patch records include edited facts, method, model, modified layers, metrics, and lineage metadata. Tensor changes are stored as safetensors deltas when available.

You can:

- Inspect patch details and browse edited facts.
- Load saved patches into their corresponding loaded model.
- Discard active edits or revert to an earlier patch state.
- Export an individual patch to a server directory configured in Settings.
- Use **Save Model ZIP** on the edit page to download the current edited model. The API also supports exporting to a server-side path.
- Preserve embedded edit lineage when exporting and reimporting an edited model.
- Delete saved patches and their dependent patches, or clear patch history.

Unloading a model or restarting the backend clears active workbench patch state; saved patch artifacts remain available for later loading. Edits baked into an exported model are part of that model's weights and are represented separately from removable workbench patches.

### Evaluation and experiment tracking

Evaluation resolves facts from the selected model's active patch stack and supports choosing multiple metrics in one run:

- **Reliability:** accuracy on the original edit prompts.
- **Generalization:** accuracy on rephrased prompts, with editable rephrases in the UI.
- **Locality:** preservation of answers to unrelated prompts, using available locality inputs.
- **Portability:** the UI accepts derived-fact prompts, but its displayed portability score is currently a placeholder, not a formal metric.

Results include aggregate metrics, per-fact details, and generated answers where available. Fact search, paged results, and virtualized lists support larger edit collections. Missing evaluation prompts limit which metrics can be meaningfully assessed.

Experiment and job records retain inputs, results, status, progress, and logs. The Activity page and topbar job menu expose ongoing work. Editing and runtime updates use server-sent events and job-wait endpoints. Edit and evaluation cancellation keeps the model loaded. Edit jobs remain **Canceling** until a safe cancellation boundary and cleanup finish: ROME/r-ROME series wait for the current EasyEdit call and retain their last fully saved fact, while MEMIT checks for cancellation before each model forward pass using a temporary workbench hook (no EasyEdit source changes). Canceled MEMIT jobs restore the pre-job weights and active patches without retaining a new patch. Explicit model unload still terminates the worker. Cooperative cancellation cannot interrupt an in-progress backward pass, matrix solve, or stuck accelerator call.

## Quick start

### Docker Compose

The supplied Compose configuration requests an NVIDIA GPU. Use a Docker installation configured for NVIDIA GPU access, and allow enough disk space and GPU memory for the selected model and editing method.

From the repository root:

```bash
cp .env.example .env
INSTALL_EASYEDIT=true docker compose up --build
```

Set `HF_TOKEN` in `.env` if needed for the selected Hugging Face repository, or enter a token in Settings after startup.

| Service | Address |
| --- | --- |
| Frontend | http://localhost:3000 |
| Backend health | http://localhost:8000/api/health |
| Interactive API documentation | http://localhost:8000/docs |

Compose mounts the repository's `data/` directory into the backend at `/app/data`, keeping downloaded models, patches, and experiments outside the container image.

### Local development

Use Python 3.10+ and Node.js 20 (the frontend container uses Node 20). Install a PyTorch build appropriate for your hardware in the backend environment.

From the repository root, create the environment file and clone EasyEdit alongside this repository:

```bash
cp .env.example .env
# Skip the clone if you already have an EasyEdit checkout.
git clone https://github.com/zjunlp/EasyEdit.git ../EasyEdit
export EASYEDIT_ROOT="$(cd ../EasyEdit && pwd)"

cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r "$EASYEDIT_ROOT/requirements.txt"
python -m pip install -e ".[dev,ml]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

The backend reads the repository and backend `.env` files automatically; existing environment variables take precedence. Keep `EASYEDIT_ROOT` set to your actual checkout when restarting the backend. EasyEdit must provide the method-specific hparams and runtime APIs used by the integration; missing setup produces an error rather than a simulated edit.

If editing or evaluation reports an EasyEdit import failure, check the dependency
error and Python path in the message. Installing requirements in another Python
environment does not install them in `backend/.venv`. From `backend/`, verify the
actual integration with:

```bash
.venv/bin/python -c 'from app.ml.editing.easyedit_runtime import _load_easyedit; _load_easyedit("ROMEHyperParams"); print("EasyEdit import OK")'
```

For a pip-based environment, use `.venv/bin/python -m pip install` explicitly;
for a uv-managed environment, use `uv pip install --python .venv/bin/python`.
Restart the backend after repairing its dependencies. On Apple Silicon with
Python 3.10, if SciPy 1.15.3 fails to load `_spropack` with a
`__thread_bss` / zero-fill error, SciPy 1.14.1 is a tested workaround on this setup:

```bash
uv pip install --python .venv/bin/python 'scipy==1.14.1'
```

### Apple Silicon (MPS)

Run the backend natively on macOS using an ARM64 Python 3.10 environment; the
NVIDIA Docker configuration remains unchanged. The shared loader automatically
prefers CUDA, then Apple MPS, then CPU. MPS uses Transformers' automatic dtype
selection from the model config/checkpoint, preserving FP16, BF16, and FP32 weights.
Only FP64 parameters and buffers are converted to FP32 before transfer to MPS.
Lower-precision editing remains dependent on model/method compatibility. Existing model/method restrictions
still apply, and larger models need enough unified memory for weights and working
tensors.

For ROME, R-ROME, MEMIT, KN, and EasyEdit evaluation/statistics, install EasyEdit's
normal dependencies. No changes to the EasyEdit checkout are required. The
workbench converts requested FP64 MOM2 precision to FP32 before invoking EasyEdit.
A scoped PyTorch dtype mode also keeps internal FP64 casts (including MEMIT's
`.double()` calls) in FP32 on MPS. Matrix inverse/solve operations stay on the GPU;
no CPU compute fallback is enabled by the workbench. CUDA precision is unchanged.

This intentionally reduces precision on MPS, so editing results may differ from
CUDA/FP64 results. MPS cache preparation and preflight use matching FP32 filenames;
existing FP64 caches are retained separately. See the [MPS precision notes](backend/compatibility/README.md).

The GPU monitor shows Apple **unified memory**: the model worker's Metal allocation
against the recommended working-set budget, plus tensor allocation. This is not
system-wide GPU usage or dedicated VRAM. Samples are collected in the worker once
per second and published through runtime status updates, including during long
operations. Explicit hard unload still stops the owning process and clears its
runtime patch state; saved model files and patches remain on disk.

Check accelerator availability in the backend environment:

```bash
python -c 'import torch; print(torch.backends.mps.is_available())'
```

### First experiment

1. Open **Overview**, choose a supported model, and download it or register a compatible local copy.
2. Click **Load** and wait for the runtime to report that the model is loaded. A cached model still needs this step.
3. Use **Generate** to record an initial answer. For GPT-2 XL, optionally explore **Logit lens** or **Causal tracing**.
4. Open **Knowledge edits**, choose a method, and enter a subject, relation prompt, and replacement answer. Prepare second-moment statistics if the chosen method/settings require them.
5. Run the edit and inspect its logs, reliability results, and patch record.
6. Open **Evaluation** to test active facts and any supplied rephrase/locality prompts. Use Patch History to compare states or reload saved edits, and export the edited model if needed.

For example, `POST /api/edit/rome` accepts the following after `gpt2-xl` has been loaded:

```json
{
  "model_id": "gpt2-xl",
  "subject": "The Space Needle",
  "relation_prompt": "{} is located in the city of",
  "ground_truth": "Seattle",
  "new_object": "Paris",
  "target_layer": null,
  "evaluation_prompts": [
    "The landmark called the Space Needle can be found in"
  ],
  "sequential_edit": false,
  "mom2_adjustment": false
}
```

Here, `{}` is replaced with the subject, and `target_layer: null` preserves EasyEdit's configured layer. See [examples/](examples/) for sample prompts and edit payloads. The current fact-upload UI accepts JSON; the directory also includes CSV and JSONL examples for reference.

## Configuration and storage

| Setting | Purpose |
| --- | --- |
| `EASYEDIT_ROOT` | Path to the EasyEdit checkout, including its `easyeditor` package and `hparams` directory. |
| `INSTALL_EASYEDIT` | Docker build argument controlling whether EasyEdit and its requirements are installed. |
| `HF_TOKEN` | Hugging Face authentication token; a runtime token can also be set in Settings. |
| `HF_HUB_CACHE` | Native backend Hub cache directory. Defaults to `~/.cache/huggingface/hub`; relative paths resolve against `backend/`. Restart the backend after changing it. |
| `HF_CACHE_HOST_DIR` | Docker host cache directory mounted at `/hf-cache`. Defaults to `./data/huggingface/hub`; recreate the backend container after changing it. |
| `DATA_DIR` | Backend artifact root. Relative values resolve against `backend/`. |
| `NEXT_PUBLIC_API_BASE` | Frontend's initial backend URL; defaults to the local backend in the supplied setup. |
| `EDIT_LOG_TO_TERMINAL` | Also mirror captured EasyEdit output to the backend terminal. |

Settings also provides a backend connection editor, runtime configuration details, patch export preferences, and second-moment cache controls.

To store downloaded models elsewhere, set `HF_HUB_CACHE=/absolute/path/to/huggingface/hub`
in the repository or backend `.env` for native runs. For Docker Compose, set
`HF_CACHE_HOST_DIR` to the host directory instead; the container uses `/hf-cache`.

Settings > Model Cache Location can override the download folder without restarting
the backend. This setting is saved in `DATA_DIR/settings/model_cache.json`; resetting
it restores the environment default. Changing folders does not move existing files:
the app searches the default and previously used folders when loading cached models.
The new folder is created if needed and must be writable. Changes are rejected while
a model download is active. Docker paths refer to the container filesystem; external
folders must be mounted in the container to remain accessible.
Downloads use the selected destination; cache status/deletion and model/tokenizer
loading use the same set of known cache folders.
`HF_HUB_CACHE` takes precedence over legacy `HUGGINGFACE_HUB_CACHE` and
`TRANSFORMERS_CACHE`; otherwise the default follows `HF_HOME` or `XDG_CACHE_HOME`.
Changing directories does not move or delete existing downloads. Stop the backend
before moving the complete cache (including blobs, snapshots, and symlinks), or
download the models again in the new location. Browser-imported models and
second-moment statistics remain under `DATA_DIR`.

With `DATA_DIR=./data`, local development stores artifacts under `backend/data/`. Compose uses the host repository's `data/` directory through its volume mount. Principal storage areas include:

- `models/`: managed model data, including browser imports under `models/imported_models/`.
- `patches/`: patch metadata and tensor deltas.
- `experiments/`: experiment/job records, logs, and result artifacts.
- `model_cache_jobs/`: persisted download job records.
- `stats/`: cached second-moment statistics used by editing methods.
- `uploads/`: upload staging data.

JSON artifacts are the source of truth and are written atomically. A rebuildable SQLite catalog in the system temporary directory indexes patch and experiment summaries and edit facts for paged queries. Some UI preferences and draft datasets are stored in the browser.

Patch metadata stores tensor paths relative to `DATA_DIR`, for example
`patches/patch_8860c4129d/weight_deltas.safetensors`, so the data directory can move
between machines. Older absolute paths are converted when the patch is read and
its tensor file can be located in the current data directory.

## Architecture and API

The frontend uses **Next.js 14, React 18, TypeScript, Tailwind CSS, and Recharts**. The backend uses **FastAPI, PyTorch, Hugging Face Transformers, and EasyEdit**. Model operations share a retained model/tokenizer in a dedicated subprocess, with serialized runtime access around model work.

| Location | Responsibility |
| --- | --- |
| `frontend/src/app/` | Overview, generation, lens, tracing, editing, evaluation, activity, and settings pages. |
| `frontend/src/components/` | Reusable workflow panels, model management, patch history, and visualizations. |
| `frontend/src/lib/` | API access, shared resources, browser persistence, and job progress helpers. |
| `backend/app/api/` | REST routes, job-wait endpoints, and event streams. |
| `backend/app/services/` | Model registry/cache/runtime, patches, experiments, and evaluation orchestration. |
| `backend/app/ml/editing/` | EasyEdit runtime integration and ROME, r-ROME, KN, MEMIT, FT, and LoRA wrappers. |
| `backend/app/ml/tracing/` | Activation corruption, restoration, and causal-effect computation. |
| `backend/app/ml/lens/` | Logit lens projections. |
| `backend/app/storage/` | Local artifact storage and the query catalog. |

Representative API groups—all prefixed with `/api`:

| Group | Routes |
| --- | --- |
| Models | `/models`, `/models/cache`, `/models/runtime`, `/models/local`, `/models/local/import/jobs` |
| Generation and analysis | `/generate`, `/logit-lens`, `/causal-trace` |
| Editing | `/edit/rome`, `/edit/r-rome`, `/edit/kn`, `/edit/memit`, `/edit/rome/sequential`, `/edit/r-rome/sequential` |
| Statistics | `/edit/mom2/status`, `/edit/mom2/prepare`, `/edit/mom2/import`, `/edit/mom2/cache` |
| Patches | `/patches`, `/patches/page`, `/patches/active`, `/patches/{patch_id}/load`, `/patches/{patch_id}/revert-to` |
| Evaluation | `/evaluate`, `/evaluate/reliability`, `/evaluate/active-model`, `/evaluate/active-facts/page`, `/evaluate/results/{experiment_id}` |
| Jobs and history | `/edit/jobs/{job_id}`, `/edit/jobs/{job_id}/events`, `/experiments`, `/experiments/page` |
| Exports | `/patches/{patch_id}/export`, `/models/export`, `/models/export/archive` |
| Configuration | `/settings`, `/models/hf-token` |

Use the running backend's [OpenAPI documentation](http://localhost:8000/docs) for HTTP methods, request schemas, and the complete endpoint list.

## Current scope and limitations

- **Compatibility is operation-specific.** A model's presence in the catalog does not imply support for every edit method or analysis tool. The current registry enables logit lens and causal tracing for GPT-2-family configurations; other families primarily support generation and configured EasyEdit methods.
- **Tracing computes the full grid.** Runtime grows with prompt length and layer count, and more corrupted samples use more memory. The UI exposes residual restoration; separate attention-output and MLP-output analysis are not exposed as independent tracing workflows.
- **Editing requires compatible EasyEdit code, hparams, and sufficient memory.** Patch persistence depends on capturing changed weights; a metrics-only result without tensor deltas cannot be reloaded as a weight patch.
- **Local imports are validated against supported architectures.** Arbitrary model formats and architectures are not universally supported.
- **Portability scoring is unfinished.** Its current placeholder should not be used as an evaluation result.
- **This is a local research/development application.** Production authentication and multi-user isolation are not implemented. Background work uses the app's current threads and model worker; the included Celery files are not required by the supplied Compose setup.

## Development checks and further reading

Run backend tests from an environment with the development dependencies installed:

```bash
cd backend
python -m pytest
```

Check frontend types and build from `frontend/`:

```bash
npx tsc --noEmit
npm run build
```

Additional documentation: [Architecture](docs/architecture.md), [Supported Models](docs/supported_models.md), [API](docs/api.md), and [Demo Script](docs/demo_script.md). For exact current model capabilities and API schemas, consult the in-app selectors and running API documentation.
