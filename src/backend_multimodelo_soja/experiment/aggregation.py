from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .metrics import METRIC_COLUMNS


def aggregate_run(run_dir: Path, logger) -> None:
    """Agrega os resultados dos folds concluidos.

    Parametros:
        run_dir: Pasta da run contendo fold_0 a fold_n.
        logger: Logger ativo da agregacao.

    Retorno:
        None: Persiste os artefatos agregados da CV.
    """
    fold_rows: List[Dict[str, object]] = []
    num_folds = 0
    total_failures = 0
    total_samples = 0
    model_name = None

    for fold_dir in sorted(run_dir.glob("fold_*")):
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json ausente em {fold_dir}")

        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        row = {"fold": metrics["fold"], "model": metrics["model"]}
        for key in ["inference_failures", "num_test_samples", "inference_failures_rate", *METRIC_COLUMNS]:
            row[key] = metrics[key]
        fold_rows.append(row)
        total_failures += int(metrics["inference_failures"])
        total_samples += int(metrics["num_test_samples"])
        num_folds += 1
        model_name = metrics["model"]

    if not fold_rows:
        raise RuntimeError(f"Nenhum fold concluido encontrado em {run_dir}")

    per_fold_df = pd.DataFrame(fold_rows).sort_values("fold")
    per_fold_df.to_csv(run_dir / "cv_summary_per_fold.csv", index=False)

    summary_rows = []
    summary_json = {
        "num_folds": num_folds,
        "folds": per_fold_df["fold"].tolist(),
        "per_fold": per_fold_df.to_dict(orient="records"),
        "inference_failures_total": int(total_failures),
        "num_test_samples_total": int(total_samples),
        "inference_failures_rate_total": float(total_failures / total_samples if total_samples else 0.0),
    }
    for metric in METRIC_COLUMNS:
        mean = float(per_fold_df[metric].mean())
        std = float(per_fold_df[metric].std(ddof=0))
        formatted = f"{mean:.4f} ({std:.4f})"
        summary_rows.append({"metric": metric, "mean": mean, "std": std, "formatted": formatted})
        summary_json[f"{metric}_mean"] = mean
        summary_json[f"{metric}_std"] = std
        summary_json[f"{metric}_formatted"] = formatted

    pd.DataFrame(summary_rows).to_csv(run_dir / "cv_summary.csv", index=False)
    with (run_dir / "cv_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, ensure_ascii=False, indent=2)

    thesis_summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "macro_f1": summary_json["macro_f1_formatted"],
                "micro_f1": summary_json["micro_f1_formatted"],
                "accuracy": summary_json["accuracy_formatted"],
                "precision": summary_json["macro_precision_formatted"],
                "recall": summary_json["macro_recall_formatted"],
                "inference_failures_total": summary_json["inference_failures_total"],
                "inference_failures_rate_total": summary_json["inference_failures_rate_total"],
            }
        ]
    )
    thesis_summary.to_csv(run_dir / "thesis_summary_table.csv", index=False)

    thesis_folds = pd.DataFrame(
        [
            {
                "model": model_name,
                "fold": int(row["fold"]),
                "macro_f1": row["macro_f1"],
                "micro_f1": row["micro_f1"],
                "accuracy": row["accuracy"],
                "precision": row["macro_precision"],
                "recall": row["macro_recall"],
                "inference_failures": int(row["inference_failures"]),
            }
            for row in per_fold_df.to_dict(orient="records")
        ]
    )
    thesis_folds.to_csv(run_dir / "thesis_folds_table.csv", index=False)
    logger.info("Agregacao concluida em %s", run_dir)
