# API

Base URL: `http://localhost:8000`

## Health

`GET /api/health`

```json
{ "status": "ok" }
```

## Models

`GET /api/models`

Returns the explicit supported-model registry.

## Model Cache

`GET /api/models/cache`

Lists supported models with local Hugging Face cache state and cache size.

`POST /api/models/cache/download`

```json
{ "model_id": "gpt2-xl" }
```

Starts a background download job.

`GET /api/models/cache/jobs/{job_id}`

Returns queued/downloading/completed/failed status for a model download.

`DELETE /api/models/cache`

```json
{ "model_id": "gpt2-xl" }
```

Deletes the local Hugging Face cache directory for that supported model.

## Model Runtime

`GET /api/models/runtime`

Lists whether each supported model is currently loaded in the running backend process.

`POST /api/models/runtime/load`

```json
{ "model_id": "gpt2-xl" }
```

Starts a background load job that reads cached weights into backend memory.

`GET /api/models/runtime/jobs/{job_id}`

Returns queued/loading/completed/failed status for a model load job.

`DELETE /api/models/runtime`

```json
{ "model_id": "gpt2-xl" }
```

Drops the backend's in-memory model reference and clears CUDA cache when available.

## Generation

`POST /api/generate`

```json
{
  "model_id": "gpt2",
  "prompt": "The Space Needle is located in the city of",
  "max_new_tokens": 20,
  "patch_id": null
}
```

## Logit Lens

`POST /api/logit-lens`

```json
{
  "model_id": "gpt2",
  "prompt": "The Space Needle is located in the city of",
  "target_token": " Seattle",
  "top_k": 10
}
```

## Causal Trace

`POST /api/causal-trace`

Returns a completed job object in the current local MVP. The result includes clean probability, corrupted probability, prompt tokens, and a layer-by-token heatmap.

## ROME Edit

`POST /api/edit/rome`

Runs EasyEdit ROME for a compatible model such as `gpt2-xl`, then saves returned metrics and safetensors weight deltas when EasyEdit exposes original changed weights.

When `target_layer` is `null`, the backend runs causal tracing first and uses the highest-scoring subject-token layer as the ROME layer. Supplying an integer skips auto-location.

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
  "sequential_edit": false
}
```

## MEMIT Edit

`POST /api/edit/memit`

Runs EasyEdit MEMIT for a compatible model such as `gpt2-xl`.

```json
{
  "model_id": "gpt2-xl",
  "edits": [
    {
      "subject": "Michael Jordan",
      "relation_prompt": "{} plays the sport of",
      "ground_truth": "basketball",
      "new_object": "baseball"
    }
  ],
  "layers": [3, 4, 5, 6, 7, 8],
  "sequential_edit": false
}
```

## Evaluation

`POST /api/evaluate`

Evaluates a saved patch against prompt lines and returns before/after sample generations plus metrics.
