"""Shared capabilities for the workbench EasyEdit integrations."""
from collections.abc import Mapping
from typing import Any, Literal

EditMethod = Literal["rome", "r_rome", "kn", "memit", "ft", "lora", "emmet", "memit_fe", "pmet", "dpo", "eamet", "alphaedit", "sphere"]
Mom2Method = Literal["rome", "r_rome", "memit", "emmet", "memit_fe", "pmet", "eamet", "alphaedit", "sphere"]
RANK_ONE_METHODS = ("rome", "r_rome")
SCOPED_WEIGHT_METHODS = ("eamet", "alphaedit", "sphere")
BATCH_WEIGHT_METHODS = ("memit", "emmet", "memit_fe", "pmet", *SCOPED_WEIGHT_METHODS)
COVARIANCE_MODULES = {
    "memit": "easyeditor.models.memit.memit_main",
    "emmet": "easyeditor.models.emmet.emmet_main",
    "memit_fe": "easyeditor.models.memit_FE.memit_FE_main",
    "pmet": "easyeditor.models.pmet.pmet_main",
    "eamet": "easyeditor.models.eamet.eamet_main",
    "alphaedit": "easyeditor.models.alphaedit.AlphaEdit_main",
    "sphere": "easyeditor.models.SPHERE.SPHERE_main",
}

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
    "emmet": {
        "hparams_class": "EMMETHyperParams",
        "hparams_key": "emmet_hparams",
        "support_key": "supports_emmet",
        "requires_subject": True,
    },
    "memit_fe": {
        "hparams_class": "MEMITFEHyperParams",
        "hparams_key": "memit_fe_hparams",
        "support_key": "supports_memit_fe",
        "requires_subject": True,
    },
    "pmet": {
        "hparams_class": "PMETHyperParams",
        "hparams_key": "pmet_hparams",
        "support_key": "supports_pmet",
        "requires_subject": True,
    },
    "dpo": {
        "hparams_class": "DPOHyperParams",
        "hparams_key": "dpo_hparams",
        "support_key": "supports_dpo",
    },
    "eamet": {
        "hparams_class": "EAMETHyperParams",
        "hparams_key": "eamet_hparams",
        "support_key": "supports_eamet",
        "requires_subject": True,
    },
    "alphaedit": {
        "hparams_class": "AlphaEditHyperParams",
        "hparams_key": "alphaedit_hparams",
        "support_key": "supports_alphaedit",
        "requires_subject": True,
    },
    "sphere": {
        "hparams_class": "SPHEREHyperParams",
        "hparams_key": "sphere_hparams",
        "support_key": "supports_sphere",
        "requires_subject": True,
    },
}


def get_editing_support(metadata: Mapping[str, Any]) -> dict[str, bool]:
    """Copy every registered editing capability, defaulting absent flags to false."""
    return {spec["support_key"]: bool(metadata.get(spec["support_key"])) for spec in METHODS.values()}


def get_editing_hparams(metadata: Mapping[str, Any]) -> dict[str, str | None]:
    """Extract profile paths from flat registration fields or an easyedit mapping."""
    return {spec["hparams_key"]: metadata.get(spec["hparams_key"]) for spec in METHODS.values()}
