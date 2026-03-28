from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict

import yaml

from ..tasks.label_normalization import CANONICAL_LABELS


def load_config(config_path: Path) -> Dict[str, Any]:
    """Carrega, resolve e valida o arquivo YAML principal.

    Parametros:
        config_path: Caminho do arquivo de configuracao.

    Retorno:
        Dict[str, Any]: Dicionario resolvido e validado.
    """
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path.resolve())
    resolve_paths(config, config_path.parent)
    validate_config(config)
    return config


def resolve_paths(config: Dict[str, Any], base_dir: Path) -> None:
    """Resolve caminhos relativos da configuracao.

    Parametros:
        config: Dicionario de configuracao.
        base_dir: Pasta base do arquivo YAML.

    Retorno:
        None: Atualiza o dicionario em memoria.
    """
    for key in ("dataset_root", "source_experiment_dir", "output_root"):
        config[key] = str((base_dir / str(config[key])).resolve())


def validate_config(config: Dict[str, Any]) -> None:
    """Valida chaves obrigatorias e precondicoes basicas.

    Parametros:
        config: Dicionario ja carregado e com caminhos resolvidos.

    Retorno:
        None: Apenas valida e levanta excecao quando necessario.
    """
    required_keys = [
        "project_name",
        "base_model_id",
        "dataset_root",
        "source_experiment_dir",
        "output_root",
        "label_set",
        "backend",
        "training",
        "memory",
        "lora",
        "pilot_mode",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Configuracao ausente: {missing}")

    if list(config["label_set"]) != CANONICAL_LABELS:
        raise ValueError("label_set deve ser exatamente: " + json.dumps(CANONICAL_LABELS))

    backend_required = [
        "name",
        "runtime_family",
        "model_family",
        "quantization_mode",
        "enable_unsloth",
        "enable_transformers_fallback",
        "extra_requirements_profile",
    ]
    backend_missing = [key for key in backend_required if key not in config["backend"]]
    if backend_missing:
        raise ValueError(f"Campos obrigatorios ausentes em backend: {backend_missing}")

    training_required = [
        "seed",
        "num_folds",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "logging_steps",
        "max_new_tokens",
        "num_workers",
        "train_prompt",
    ]
    training_missing = [key for key in training_required if key not in config["training"]]
    if training_missing:
        raise ValueError(f"Campos obrigatorios ausentes em training: {training_missing}")

    pilot_required = [
        "enabled",
        "max_train_samples_per_fold",
        "max_val_samples_per_fold",
        "max_test_samples_per_fold",
    ]
    pilot_missing = [key for key in pilot_required if key not in config["pilot_mode"]]
    if pilot_missing:
        raise ValueError(f"Campos obrigatorios ausentes em pilot_mode: {pilot_missing}")

    for key in ("dataset_root", "source_experiment_dir"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"Caminho inexistente em config: {config[key]}")


def fold_manifest_path(config: Dict[str, Any], fold: int) -> Path:
    """Monta o caminho do manifesto de um fold.

    Parametros:
        config: Dicionario de configuracao carregado.
        fold: Indice do fold.

    Retorno:
        Path: Caminho absoluto para o fold_manifest.csv.
    """
    return Path(config["source_experiment_dir"]) / f"fold_{fold}" / "fold_manifest.csv"


def slugify_model_id(model_id: str) -> str:
    """Converte o identificador do modelo para um slug curto de pasta.

    Parametros:
        model_id: Identificador bruto do checkpoint.

    Retorno:
        str: Slug ASCII compativel com nome de pasta.
    """
    tail = model_id.split("/")[-1].strip().lower()
    tail = re.sub(r"[^a-z0-9]+", "_", tail)
    return tail.strip("_") or "modelo"


def run_mode_tag(config: Dict[str, Any], fold: int | None = None) -> str:
    """Monta a tag de modo para o nome da run.

    Parametros:
        config: Dicionario de configuracao carregado.
        fold: Fold isolado quando aplicavel.

    Retorno:
        str: Sufixo de modo para o nome da pasta.
    """
    if fold is None:
        return "pilot_cv" if bool(config["pilot_mode"]["enabled"]) else "cv_full"
    prefix = "pilot" if bool(config["pilot_mode"]["enabled"]) else "full"
    return f"{prefix}_fold{fold}"


def build_auto_run_dir(config: Dict[str, Any], fold: int | None = None) -> Path:
    """Cria um diretorio automatico de execucao baseado no modelo e no modo.

    Parametros:
        config: Dicionario de configuracao carregado.
        fold: Fold isolado quando aplicavel.

    Retorno:
        Path: Diretorio da nova execucao.
    """
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify_model_id(str(config["base_model_id"]))
    backend_name = str(config["backend"]["name"]).strip().lower()
    mode_tag = run_mode_tag(config, fold=fold)
    run_name = f"{model_slug}_{backend_name}_{ts}_{mode_tag}"
    run_dir = Path(config["output_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_run_dir(config: Dict[str, Any]) -> Path:
    """Cria o diretorio de saida de uma nova run completa de CV.

    Parametros:
        config: Dicionario de configuracao carregado.

    Retorno:
        Path: Diretorio da nova execucao.
    """
    return build_auto_run_dir(config, fold=None)
