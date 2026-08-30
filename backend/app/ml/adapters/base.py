from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    model_id: str
    model_type: str
    n_layers: int

    @abstractmethod
    def load_model(self, local_files_only: bool = False) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load_tokenizer(self, local_files_only: bool = False) -> Any:
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int, local_files_only: bool = False) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_residual_hook_name(self, layer: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_mlp_input_hook_name(self, layer: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_mlp_output_hook_name(self, layer: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_attention_output_hook_name(self, layer: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_default_rome_layer(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_default_memit_layers(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def find_subject_token_indices(self, prompt: str, subject: str) -> list[int]:
        raise NotImplementedError
