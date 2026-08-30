from __future__ import annotations

from app.ml.adapters.gpt2_adapter import GPT2Adapter


class PythiaAdapter(GPT2Adapter):
    model_type = "pythia"

    def get_residual_hook_name(self, layer: int) -> str:
        return f"gpt_neox.layers.{layer}"

    def get_mlp_input_hook_name(self, layer: int) -> str:
        return f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h"

    def get_mlp_output_hook_name(self, layer: int) -> str:
        return f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h"

    def get_attention_output_hook_name(self, layer: int) -> str:
        return f"gpt_neox.layers.{layer}.attention.dense"

