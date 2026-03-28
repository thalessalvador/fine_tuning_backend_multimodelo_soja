from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..backends.backend_registry import create_backend
from ..backends.base_backend import FoldData
from ..tasks.label_normalization import CANONICAL_LABELS, RAW_LABEL_ORDER
from .config import fold_manifest_path
from .data import assert_label_idx_alignment, dataset_summary, load_and_validate_manifest, subset_records
from .metrics import compute_classification_metrics, save_confusion_matrix_csv, save_metrics_json


def _metrics_from_predictions(predictions, fold: int, model_name: str) -> Dict[str, object]:
    """Deriva metricas a partir do arquivo de predicoes.

    Parametros:
        predictions: DataFrame de predicoes normalizadas.
        fold: Fold avaliado.
        model_name: Nome do modelo avaliado.

    Retorno:
        Dict[str, object]: Dicionario serializavel de metricas.
    """
    y_true = predictions["true_label"].tolist()
    y_pred = [
        pred if pred in CANONICAL_LABELS else CANONICAL_LABELS[0]
        for pred in predictions["pred_label"].replace("", "__INVALID__").tolist()
    ]
    inference_failures = int((predictions["error"] != "").sum())
    return compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels_for_report=CANONICAL_LABELS,
        raw_labels_for_per_class=RAW_LABEL_ORDER,
        fold=fold,
        model_name=model_name,
        inference_failures=inference_failures,
    )


def log_training_identity(config: Dict[str, Any], fold: int, logger) -> None:
    """Registra a identidade solicitada para o treino no inicio do fold.

    Parametros:
        config: Configuracao resolvida do experimento.
        fold: Indice numerico do fold.
        logger: Logger ativo.

    Retorno:
        None: Apenas escreve no log de runtime.
    """
    logger.info(
        "training_identity fold=%s project=%s model_id=%s backend_requested=%s runtime_family=%s model_family=%s quantization_mode=%s extra_requirements_profile=%s",
        fold,
        config["project_name"],
        config["base_model_id"],
        config["backend"]["name"],
        config["backend"]["runtime_family"],
        config["backend"]["model_family"],
        config["backend"]["quantization_mode"],
        config["backend"]["extra_requirements_profile"],
    )


def log_runtime_identity(runtime_metadata: Dict[str, Any], logger) -> None:
    """Registra a identidade efetiva do runtime usado no treino.

    Parametros:
        runtime_metadata: Metadados retornados pelo backend.
        logger: Logger ativo.

    Retorno:
        None: Apenas escreve no log de runtime.
    """
    logger.info(
        "runtime_identity backend_requested=%s backend_used=%s model_id=%s model_family=%s runtime_family=%s quantization_mode=%s main_dtype=%s cuda_available=%s cuda_device_name=%s fallback_reason=%s",
        runtime_metadata.get("backend_requested"),
        runtime_metadata.get("backend_used"),
        runtime_metadata.get("model_id"),
        runtime_metadata.get("model_family"),
        runtime_metadata.get("runtime_family"),
        runtime_metadata.get("quantization_mode"),
        runtime_metadata.get("main_dtype"),
        runtime_metadata.get("cuda_available"),
        runtime_metadata.get("cuda_device_name"),
        runtime_metadata.get("fallback_reason", ""),
    )


def run_fold(config: Dict[str, Any], fold: int, fold_output_dir: Path, logger) -> None:
    """Executa o fluxo completo de um fold com backend modular.

    Parametros:
        config: Configuracao resolvida do experimento.
        fold: Indice numerico do fold.
        fold_output_dir: Pasta de saida do fold.
        logger: Logger ativo.

    Retorno:
        None: Persiste artefatos do fold em disco.
    """
    log_training_identity(config, fold, logger)

    manifest_path = fold_manifest_path(config, fold)
    manifest = load_and_validate_manifest(manifest_path, fold)
    assert_label_idx_alignment(manifest)

    train_records = subset_records(manifest, "train", config["pilot_mode"], int(config["training"]["seed"]))
    val_records = subset_records(manifest, "val", config["pilot_mode"], int(config["training"]["seed"]))
    test_records = subset_records(manifest, "test", config["pilot_mode"], int(config["training"]["seed"]))

    logger.info(
        "fold=%s manifest=%s train=%s val=%s test=%s",
        fold,
        manifest_path,
        len(train_records),
        len(val_records),
        len(test_records),
    )
    logger.info("train_label_distribution=%s", dataset_summary(train_records))
    logger.info("val_label_distribution=%s", dataset_summary(val_records))
    logger.info("test_label_distribution=%s", dataset_summary(test_records))

    fold_data = FoldData(
        fold=fold,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )
    backend = create_backend(str(config["backend"]["name"]))
    train_result = backend.train_fold(fold_data=fold_data, output_dir=fold_output_dir, config=config, logger=logger)
    log_runtime_identity(train_result.runtime_metadata, logger)

    save_metrics_json(train_result.val_metrics_best, fold_output_dir / "val_metrics_best.json")
    with (fold_output_dir / "train_history.json").open("w", encoding="utf-8") as handle:
        json.dump(train_result.history, handle, ensure_ascii=False, indent=2)

    predictions = backend.predict(
        records=test_records,
        adapter_dir=train_result.adapter_dir,
        output_dir=fold_output_dir,
        config=config,
        logger=logger,
    )
    predictions.to_csv(fold_output_dir / "predictions.csv", index=False)

    metrics = _metrics_from_predictions(predictions, fold, str(config["base_model_id"]))
    save_metrics_json(metrics, fold_output_dir / "metrics.json")
    save_confusion_matrix_csv(metrics, fold_output_dir / "confusion_matrix.csv")
    backend.save_backend_metadata(
        output_dir=fold_output_dir,
        config=config,
        runtime_metadata=train_result.runtime_metadata,
    )

    resolved_config = {
        **config,
        "resolved": {
            "fold": fold,
            "manifest_path": str(manifest_path),
            "fold_output_dir": str(fold_output_dir.resolve()),
            "best_epoch": train_result.best_epoch,
            "best_val_macro_f1": train_result.best_val_macro_f1,
        },
    }
    with (fold_output_dir / "train_config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, ensure_ascii=False, indent=2)
