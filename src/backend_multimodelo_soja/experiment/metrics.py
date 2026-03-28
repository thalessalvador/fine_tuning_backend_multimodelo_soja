from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


METRIC_COLUMNS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels_for_report: List[str],
    raw_labels_for_per_class: List[str],
    fold: int,
    model_name: str,
    inference_failures: int,
) -> Dict[str, object]:
    """Calcula metricas globais e por classe.

    Parametros:
        y_true: Rotulos reais.
        y_pred: Rotulos previstos normalizados.
        labels_for_report: Ordem canonica usada no relatorio.
        raw_labels_for_per_class: Ordem bruta usada em per_class.
        fold: Fold avaliado.
        model_name: Nome do modelo.
        inference_failures: Total de respostas invalidas.

    Retorno:
        Dict[str, object]: Estrutura serializavel de metricas.
    """
    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_for_report, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_for_report, average="micro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_for_report, average="weighted", zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_for_report, average=None, zero_division=0
    )

    confusion = confusion_matrix(y_true, y_pred, labels=labels_for_report)
    per_class_metrics = {}
    for index, raw_label in enumerate(raw_labels_for_per_class):
        label = labels_for_report[index]
        per_class_metrics[raw_label] = {
            "precision": float(per_class_precision[index]),
            "recall": float(per_class_recall[index]),
            "f1": float(per_class_f1[index]),
            "accuracy": float(
                np.mean([int((true == label) == (pred == label)) for true, pred in zip(y_true, y_pred)])
            ),
            "support": int(support[index]),
        }

    num_samples = len(y_true)
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
        "support_per_class": {
            raw_labels_for_per_class[idx]: int(value) for idx, value in enumerate(support)
        },
        "experiment_mode": "cv",
        "fold": fold,
        "model": model_name,
        "prompt_labels": labels_for_report,
        "inference_failures": int(inference_failures),
        "num_test_samples": int(num_samples),
        "inference_failures_rate": float(inference_failures / num_samples if num_samples else 0.0),
    }


def save_metrics_json(metrics: Dict[str, object], output_path: Path) -> None:
    """Salva metricas em JSON.

    Parametros:
        metrics: Dicionario de metricas.
        output_path: Caminho de destino.

    Retorno:
        None: Apenas persiste o arquivo.
    """
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def save_confusion_matrix_csv(metrics: Dict[str, object], output_path: Path) -> None:
    """Salva a matriz de confusao em CSV.

    Parametros:
        metrics: Dicionario contendo confusion_matrix.
        output_path: Caminho de destino.

    Retorno:
        None: Apenas persiste o arquivo.
    """
    matrix = pd.DataFrame(metrics["confusion_matrix"])
    matrix.to_csv(output_path, index=False, header=False)
