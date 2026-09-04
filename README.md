# Knowledge Editing Workbench

An interactive LLM knowledge-editing workbench for exploring how factual associations are represented and modified inside transformer language models. The app supports generation, logit lens inspection, causal tracing, ROME and Knowledge Neurons (KN) single-fact edits, MEMIT batch edits, reversible patch management, and before/after evaluation.

This repository ships a runnable MVP scaffold plus a real EasyEdit integration for ROME, KN, and MEMIT editing. The intended setup installs EasyEdit and uses real model weights for generation, inspection, editing, and evaluation.

## Features

- FastAPI backend with `/api/models`, `/api/generate`, `/api/logit-lens`, `/api/causal-trace`, `/api/edit/rome`, `/api/edit/kn`, `/api/edit/memit`, `/api/patches`, `/api/evaluate`, and `/api/experiments`.
- Next.js frontend with prompt playground, logit lens chart/table, causal tracing heatmap, edit forms, patch history, and evaluation dashboard.
- Local filesystem storage under `data/experiments` and `data/patches`.
- Built-in model catalog derived from EasyEdit ROME, KN, and MEMIT hparams, including GPT-2 XL, GPT-J, Baichuan 2, ChatGLM/GLM, InternLM, Llama, Mistral, and Qwen families.
- Method-specific edit selectors show only models with compatible EasyEdit hparams, while generation and the workbench show their combined catalog.
- Saved safetensors weight deltas when EasyEdit returns original changed weights. Base model weights are applied temporarily for generation and then rolled back in memory.
- Model cache management UI for Hugging Face downloads, browser-imported local models, cache size inspection, and deleting managed cached models.

## Local Setup

```bash
cp .env.example .env
INSTALL_EASYEDIT=true docker compose up --build
```

For backend-only development:

```bash
git clone https://github.com/zjunlp/EasyEdit.git ../EasyEdit
export EASYEDIT_ROOT="$(pwd)/../EasyEdit"
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../EasyEdit/requirements.txt
pip install -e ".[dev,ml]"
uvicorn app.main:app --reload
uvicorn app.main:app --reload --env-file ../.env
```

Then open:

- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:8000/api/health`
- API docs: `http://localhost:8000/docs`

For frontend-only development:

```bash
cd frontend
npm install
npm run dev
```

## EasyEdit Setup

```bash
git clone https://github.com/zjunlp/EasyEdit.git ../EasyEdit
export EASYEDIT_ROOT="$(pwd)/../EasyEdit"
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../EasyEdit/requirements.txt
pip install -e ".[dev,ml]"
uvicorn app.main:app --reload
```

ROME, KN, and MEMIT use EasyEdit via:

- `backend/app/ml/editing/easyedit_runtime.py`
- `backend/app/ml/editing/kn_wrapper.py`
- `backend/app/ml/editing/rome_wrapper.py`
- `backend/app/ml/editing/memit_wrapper.py`

Install EasyEdit into an existing backend virtualenv with:

```bash
export EASYEDIT_ROOT=/path/to/EasyEdit
cd backend
pip install -r "$EASYEDIT_ROOT/requirements.txt"
pip install -e ".[dev,ml]"
```

The EasyEdit routes require method hparams such as `hparams/ROME/gpt2-xl.yaml`, `hparams/KN/gpt2-xl.yaml`, and `hparams/MEMIT/gpt2-xl.yaml`. If EasyEdit or the hparams file is missing, the corresponding edit endpoint returns a 503 setup error instead of a fake patch.

Generation applies a saved EasyEdit patch when the selected patch matches the selected model and safetensors deltas were saved.

The causal tracing module uses real activation corruption/restoration for GPT-2 family models:

- `backend/app/ml/tracing/causal_trace.py`

For ROME, leaving `target_layer` as `null` runs causal tracing first, selects the layer with the strongest positive indirect effect over the subject tokens, and then calls EasyEdit ROME with that layer. Setting `target_layer` manually skips auto-location and uses the supplied override.

## Model Cache

The UI shows whether each supported Hugging Face model is cached locally. Use the `Download` action before generation, tracing, or editing if you want to avoid a long first-run wait. The cache manager can also delete a model's local Hugging Face cache directory when you no longer want to keep it.

Disk cache and backend memory are separate states. A model can be fully downloaded or imported but still take time to load weights into the running backend process. Use `Load` to preload weights before pressing Generate/Edit, and `Unload` to release the backend's model reference.

### Local model imports

The original **Register Local Model** form still accepts an existing local path. As an alternative, use **Choose Model Folder** to select a complete Hugging Face model export from the system file picker. Selecting the folder reads its root `config.json` in the browser and fills the browser-visible source-folder path; if that file is missing or incompatible, Register is disabled. The folder files are not sent to the backend until **Register** is clicked. Register then starts the upload/validation job and moves a compatible model into the managed cache.

After Register, managed imports live under `data/models/imported_models` (normally `backend/data/models/imported_models` when running backend-only development). The display name defaults to the inferred base model name (for example `GPT-2 XL`) and can be changed before registration. Removing a managed import deletes both its registration and copied model files after it has been unloaded.

Backend endpoints:

- `GET /api/models/cache`
- `POST /api/models/cache/download`
- `GET /api/models/cache/jobs/{job_id}`
- `DELETE /api/models/cache`
- `GET /api/models/runtime`
- `POST /api/models/runtime/load`
- `GET /api/models/runtime/jobs/{job_id}`
- `DELETE /api/models/runtime`

## Example Edit

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
  ]
}
```

Example files live in `examples/`.

## Documentation

- [Architecture](docs/architecture.md)
- [Supported Models](docs/supported_models.md)
- [API](docs/api.md)
- [Demo Script](docs/demo_script.md)

## Known Limitations

- ROME/KN/MEMIT require EasyEdit, compatible hparams, and enough GPU memory. EasyEdit’s README reports roughly 10GB VRAM for ROME on GPT-2 XL and 11GB for MEMIT on GPT-2 XL.
- If EasyEdit does not return original changed weights for a method/model combination, the backend can report metrics but may not be able to persist safetensors deltas.
- No arbitrary model uploads.
- No production authentication or multi-user collaboration.
- Background jobs are represented by job records; long-running work can be moved into Celery via `backend/app/workers`.
- SQLite schema files are included, while the MVP service layer uses local JSON records for fast iteration.

## Future Work

- Broaden real causal tracing beyond GPT-2-family block hooks.
- Add polling or WebSocket job progress for long runs.
- Add reproducible experiment replay from stored JSON inputs.
- Add screenshots and a short demo video once the local EasyEdit path is verified on GPU.
