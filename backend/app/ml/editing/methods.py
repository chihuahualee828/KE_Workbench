"""Shared capabilities for the workbench EasyEdit integrations."""
from collections.abc import Mapping
from typing import Any, Literal

EditMethod = Literal["rome", "r_rome", "kn", "memit", "ft", "lora"]
Mom2Method = Literal["rome", "r_rome", "memit"]
RANK_ONE_METHODS = ("rome", "r_rome")
BATCH_WEIGHT_METHODS = ("memit",)

METHODS = {
    "rome": {
        "hparams_class": "ROMEHyperParams",
        "hparams_key": "rome_hparams",
        "support_key": "supports_rome",
        "requires_subject": True,
    },
    "r_rome": {
        "hparams_class": "R_ROMEHyperParams",
        "hparams_key": "r_rome_hparams",
        "support_key": "supports_r_rome",
        "requires_subject": True,
    },
    "memit": {
        "hparams_class": "MEMITHyperParams",
        "hparams_key": "memit_hparams",
        "support_key": "supports_memit",
        "requires_subject": True,
    },
    "ft": {
        "hparams_class": "FTHyperParams",
        "hparams_key": "ft_hparams",
        "support_key": "supports_ft",
    },
    "lora": {
        "hparams_class": "LoRAHyperParams",
        "hparams_key": "lora_hparams",
        "support_key": "supports_lora",
    },
    "kn": {
        "hparams_class": "KNHyperParams",
        "hparams_key": "kn_hparams",
        "support_key": "supports_kn",
    },
}


def get_editing_support(metadata: Mapping[str, Any]) -> dict[str, bool]:
    """Copy every registered editing capability, defaulting absent flags to false."""
    return {spec["support_key"]: bool(metadata.get(spec["support_key"])) for spec in METHODS.values()}


def get_editing_hparams(metadata: Mapping[str, Any]) -> dict[str, str | None]:
    """Extract profile paths from flat registration fields or an easyedit mapping."""
    return {spec["hparams_key"]: metadata.get(spec["hparams_key"]) for spec in METHODS.values()}
