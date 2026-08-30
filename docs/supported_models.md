# Supported Models

Initial explicit registry:

| Model ID | Display | Layers | Generation | Lens | Trace | ROME | MEMIT |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `gpt2` | GPT-2 Small | 12 | yes | yes | yes | no | no |
| `gpt2-medium` | GPT-2 Medium | 24 | yes | yes | yes | no | no |
| `gpt2-xl` | GPT-2 XL | 48 | yes | no | no | yes | yes |
| `EleutherAI/pythia-70m` | Pythia 70M | 6 | yes | yes | yes | no | no |

ROME/MEMIT support is backed by EasyEdit hparams:

- `hparams/ROME/gpt2-xl.yaml`
- `hparams/MEMIT/gpt2-xl.yaml`

The registry is intentionally explicit. Arbitrary uploaded models are out of scope for the MVP.
