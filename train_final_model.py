from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from src.backend_multimodelo_soja.backends.backend_registry import create_backend
from src.backend_multimodelo_soja.backends.base_backend import FoldData
from src.backend_multimodelo_soja.experiment.config import fold_manifest_path, load_config, slugify_model_id
from src.backend_multimodelo_soja.experiment.data import SampleRecord, assert_label_idx_alignment, load_and_validate_manifest
from src.backend_multimodelo_soja.experiment.fold_runner import log_runtime_identity, log_training_identity
from src.backend_multimodelo_soja.experiment.logging_utils import setup_logger
from src.backend_multimodelo_soja.experiment.metrics import save_metrics_json
from src.backend_multimodelo_soja.tasks.label_normalization import canonicalize_dataset_label


def parse_args() -> argparse.Namespace:
    """Processa os argumentos do treino final de modelo unico."""
    parser = argparse.ArgumentParser(
        description="Treina um unico modelo final a partir dos dados deduplicados dos manifests de CV."
    )
    parser.add_argument("--config", required=True, help="Arquivo de configuracao YAML.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Diretorio opcional de saida. Quando omitido, cria outputs/<modelo>_<backend>_<timestamp>_final_model.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fracao estratificada reservada para validacao interna do treino final.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para o split final. Quando omitida, usa training.seed do config.",
    )
    return parser.parse_args()


def build_final_run_dir(config: Dict[str, object]) -> Path:
    """Cria o diretorio da run final de modelo unico."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify_model_id(str(config["base_model_id"]))
    backend_name = str(config["backend"]["name"]).strip().lower()
    run_name = f"{model_slug}_{backend_name}_{ts}_final_model"
    run_dir = Path(str(config["output_root"])) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_unique_records_from_cv_manifests(config: Dict[str, object]) -> List[SampleRecord]:
    """Carrega os dados unicos a partir dos manifests de todos os folds da CV."""
    unique_by_path: Dict[str, SampleRecord] = {}
    num_folds = int(config["training"]["num_folds"])

    for fold in range(num_folds):
        manifest_path = fold_manifest_path(config, fold)
        manifest = load_and_validate_manifest(manifest_path, fold)
        assert_label_idx_alignment(manifest)

        for row in manifest.itertuples(index=False):
            path = str(Path(row.path).resolve())
            candidate = SampleRecord(
                fold=-1,
                subset="all",
                path=path,
                label_idx=int(row.label_idx),
                raw_label_name=str(row.label_name),
                canonical_label=canonicalize_dataset_label(str(row.label_name)),
            )
            if path in unique_by_path:
                existing = unique_by_path[path]
                if existing.label_idx != candidate.label_idx or existing.raw_label_name != candidate.raw_label_name:
                    raise ValueError(
                        f"Registro inconsistente entre manifests para {path}: "
                        f"{existing.raw_label_name}/{existing.label_idx} vs "
                        f"{candidate.raw_label_name}/{candidate.label_idx}"
                    )
                continue
            unique_by_path[path] = candidate

    return sorted(unique_by_path.values(), key=lambda item: item.path)


def build_final_train_val_split(
    records: List[SampleRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    """Gera um split final estratificado train/val."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"--val-ratio deve estar entre 0 e 1. Obtido: {val_ratio}")

    paths = [record.path for record in records]
    labels = [record.canonical_label for record in records]
    train_paths, val_paths = train_test_split(
        paths,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_set = set(train_paths)
    val_set = set(val_paths)

    train_records = [
        SampleRecord(**{**record.__dict__, "subset": "train"})
        for record in records
        if record.path in train_set
    ]
    val_records = [
        SampleRecord(**{**record.__dict__, "subset": "val"})
        for record in records
        if record.path in val_set
    ]
    return train_records, val_records


def summarize_records(records: List[SampleRecord]) -> Dict[str, int]:
    """Conta amostras por rotulo canonico."""
    summary: Dict[str, int] = {}
    for record in records:
        summary[record.canonical_label] = summary.get(record.canonical_label, 0) + 1
    return dict(sorted(summary.items()))


def save_final_manifest(train_records: List[SampleRecord], val_records: List[SampleRecord], output_path: Path) -> None:
    """Salva o manifesto do treino final para auditoria e reproducao."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "subset", "path", "label_idx", "label_name"])
        for subset_name, records in (("train", train_records), ("val", val_records)):
            for record in records:
                writer.writerow([-1, subset_name, record.path, record.label_idx, record.raw_label_name])


def write_final_model_manifest(
    run_dir: Path,
    config: Dict[str, object],
    train_records: List[SampleRecord],
    val_records: List[SampleRecord],
    split_seed: int,
    val_ratio: float,
    train_result,
) -> Path:
    """Escreve o manifesto principal da run final."""
    manifest = {
        "artifact_type": "final_model",
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "base_model_id": config["base_model_id"],
        "dataset_root": config["dataset_root"],
        "source_experiment_dir": config["source_experiment_dir"],
        "label_set": config["label_set"],
        "backend": config["backend"],
        "training": {
            "train_prompt": config["training"]["train_prompt"],
            "max_new_tokens": config["training"]["max_new_tokens"],
            "num_epochs": config["training"]["num_epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        },
        "final_split": {
            "strategy": "deduplicated_cv_manifests_stratified_train_val",
            "val_ratio": val_ratio,
            "seed": split_seed,
            "num_unique_samples": len(train_records) + len(val_records),
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "train_distribution": summarize_records(train_records),
            "val_distribution": summarize_records(val_records),
        },
        "artifacts": {
            "adapter_dir": str((run_dir / "adapter").resolve()),
            "runtime_log_path": str((run_dir / "runtime.log").resolve()),
            "run_config_path": str((run_dir / "run_config.json").resolve()),
            "train_history_path": str((run_dir / "train_history.json").resolve()),
            "val_metrics_best_path": str((run_dir / "val_metrics_best.json").resolve()),
            "backend_metadata_path": str((run_dir / "backend_metadata.json").resolve()),
            "final_manifest_csv_path": str((run_dir / "final_manifest.csv").resolve()),
        },
        "selection": {
            "best_epoch": train_result.best_epoch,
            "best_val_macro_f1": train_result.best_val_macro_f1,
        },
    }
    output_path = run_dir / "final_model_manifest.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_path


def write_inference_handoff(run_dir: Path, config: Dict[str, object]) -> Path:
    """Gera o manifesto de handoff do modelo final unico."""
    handoff = {
        "handoff_version": 1,
        "artifact_type": "final_model",
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "run_config_path": str((run_dir / "run_config.json").resolve()),
        "source_experiment_dir": config["source_experiment_dir"],
        "dataset_root": config["dataset_root"],
        "label_set": config["label_set"],
        "base_model_id": config["base_model_id"],
        "model_family": config["backend"]["model_family"],
        "runtime_family": config["backend"]["runtime_family"],
        "quantization_mode": config["backend"]["quantization_mode"],
        "train_prompt": config["training"]["train_prompt"],
        "max_new_tokens": config["training"]["max_new_tokens"],
        "recommended_runtime": "hf_transformers",
        "supported_model_sources": ["base", "finetuned"],
        "base_usage": {
            "base_model_id": config["base_model_id"],
            "adapter_dir": None,
        },
        "finetuned_usage": {
            "base_model_id": config["base_model_id"],
            "adapter_dir": str((run_dir / "adapter").resolve()),
        },
    }
    output_path = run_dir / "inference_handoff.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(handoff, handle, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    """Executa o treino final de um unico modelo a partir da base consolidada."""
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    split_seed = int(args.seed if args.seed is not None else config["training"]["seed"])
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if run_dir.exists():
            raise FileExistsError(f"--run-dir ja existe: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = build_final_run_dir(config)

    logger = setup_logger(run_dir / "runtime.log", "train_final_model")
    logger.info("Inicio do treino final de modelo unico")
    logger.info("Diretorio da run final: %s", run_dir)

    log_training_identity(config, fold=-1, logger=logger)
    logger.info("final_model_split strategy=deduplicated_cv_manifests_stratified_train_val val_ratio=%.4f seed=%s", args.val_ratio, split_seed)

    unique_records = load_unique_records_from_cv_manifests(config)
    train_records, val_records = build_final_train_val_split(unique_records, val_ratio=float(args.val_ratio), seed=split_seed)

    logger.info("final_model_unique_samples=%s", len(unique_records))
    logger.info("final_model_train_samples=%s distribution=%s", len(train_records), summarize_records(train_records))
    logger.info("final_model_val_samples=%s distribution=%s", len(val_records), summarize_records(val_records))

    save_final_manifest(train_records, val_records, run_dir / "final_manifest.csv")

    run_config = {
        "artifact_type": "final_model",
        "project_name": config["project_name"],
        "base_model_id": config["base_model_id"],
        "dataset_root": config["dataset_root"],
        "source_experiment_dir": config["source_experiment_dir"],
        "output_root": config["output_root"],
        "label_set": config["label_set"],
        "backend": config["backend"],
        "training": config["training"],
        "memory": config["memory"],
        "lora": config["lora"],
        "pilot_mode": config["pilot_mode"],
        "final_split": {
            "strategy": "deduplicated_cv_manifests_stratified_train_val",
            "val_ratio": float(args.val_ratio),
            "seed": split_seed,
        },
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    fold_data = FoldData(
        fold=-1,
        train_records=train_records,
        val_records=val_records,
        test_records=[],
    )
    backend = create_backend(str(config["backend"]["name"]))
    train_result = backend.train_fold(fold_data=fold_data, output_dir=run_dir, config=config, logger=logger)
    log_runtime_identity(train_result.runtime_metadata, logger)

    save_metrics_json(train_result.val_metrics_best, run_dir / "val_metrics_best.json")
    with (run_dir / "train_history.json").open("w", encoding="utf-8") as handle:
        json.dump(train_result.history, handle, ensure_ascii=False, indent=2)
    backend.save_backend_metadata(output_dir=run_dir, config=config, runtime_metadata=train_result.runtime_metadata)

    write_final_model_manifest(
        run_dir=run_dir,
        config=config,
        train_records=train_records,
        val_records=val_records,
        split_seed=split_seed,
        val_ratio=float(args.val_ratio),
        train_result=train_result,
    )
    write_inference_handoff(run_dir=run_dir, config=config)
    logger.info("Treino final concluido com sucesso em %s", run_dir)


if __name__ == "__main__":
    main()
