# Supported Models

Initial explicit registry:

| Model ID | Display | Layers | Generation | Lens | Trace | ROME | KN | MEMIT |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `gpt2-xl` | GPT-2 XL | 48 | yes | yes | yes | yes | yes | yes |
| `gpt-j-6b` | GPT-J 6B | 28 | yes | no | no | yes | yes | yes |
| `baichuan-7b` | Baichuan 2 7B Chat | 32 | yes | no | no | yes | yes | yes |
| `chatglm2-6b` | ChatGLM2 6B | 28 | yes | no | no | yes | yes | yes |
| `chatglm4-9b` | GLM-4 9B Chat | 40 | yes | no | no | yes | no | no |
| `internlm-7b` | InternLM 7B | 32 | yes | no | no | yes | yes | yes |
| `llama-7b` | Llama 2 7B | 32 | yes | no | no | yes | yes | yes |
| `llama3-8b` | Llama 3 8B | 32 | yes | no | no | yes | yes | no |
| `llama3.2-3b` | Llama 3.2 3B | 28 | yes | no | no | yes | yes | yes |
| `mistral-7b` | Mistral 7B | 32 | yes | no | no | yes | yes | yes |
| `qwen2.5-7b` | Qwen2.5 7B | 28 | yes | no | no | yes | yes | yes |
| `qwen2-7b` | Qwen2 7B | 28 | yes | no | no | yes | yes | yes |
| `qwen-7b` | Qwen 7B | 32 | yes | no | no | yes | yes | yes |
| `qwen3.5-9b` | Qwen3.5 9B | 32 | yes | no | no | yes | no | no |

ROME/KN/MEMIT support is backed by EasyEdit hparams:

- `hparams/ROME/gpt2-xl.yaml`
- `hparams/KN/gpt2-xl.yaml`
- `hparams/MEMIT/gpt2-xl.yaml`

The registry is intentionally explicit. Arbitrary uploaded models are out of scope for the MVP.
