import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    """Processa os argumentos da geracao do manifesto de handoff para inferencia."""
    parser = argparse.ArgumentParser(
        description="Gera um manifesto de handoff da run de fine-tuning para consumo por runtimes externos."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Diretorio da run em outputs/ contendo run_config.json e as pastas fold_<i>.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Lista opcional de folds a incluir. Quando omitido, usa todos os folds encontrados em run-dir.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Caminho opcional do JSON de saida. Padrao: <run-dir>/inference_handoff.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    """Le um JSON e retorna dicionario vazio quando o arquivo nao existe."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def discover_folds(run_dir: Path) -> List[int]:
    """Descobre automaticamente os folds presentes na run."""
    folds: List[int] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("fold_"):
            continue
        try:
            folds.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(folds)


def detect_artifact_type(run_dir: Path, run_config: Dict[str, Any]) -> str:
    """Detecta se a run representa CV por fold ou um modelo final unico."""
    artifact_type = str(run_config.get("artifact_type", "")).strip().lower()
    if artifact_type:
        return artifact_type
    if (run_dir / "final_model_manifest.json").exists() or (run_dir / "adapter").exists():
        return "final_model"
    return "cv"


def resolve_manifest_path(run_config: Dict[str, Any], fold: int, fold_dir: Path) -> Optional[str]:
    """Resolve o manifesto do fold preferindo o config resolvido do proprio fold."""
    train_config = load_json(fold_dir / "train_config_resolved.json")
    resolved = train_config.get("resolved", {})
    manifest_path = resolved.get("manifest_path")
    if manifest_path:
        return str(Path(manifest_path).resolve())

    source_experiment_dir = run_config.get("source_experiment_dir")
    if not source_experiment_dir:
        return None

    candidate = Path(source_experiment_dir) / f"fold_{fold}" / "fold_manifest.csv"
    return str(candidate.resolve()) if candidate.exists() else str(candidate.resolve())


def build_fold_entry(run_config: Dict[str, Any], run_dir: Path, fold: int) -> Dict[str, Any]:
    """Constroi o bloco serializavel de um fold para o manifesto."""
    fold_dir = run_dir / f"fold_{fold}"
    adapter_dir = fold_dir / "adapter"
    backend_metadata_path = fold_dir / "backend_metadata.json"
    train_config_path = fold_dir / "train_config_resolved.json"
    metrics_path = fold_dir / "metrics.json"
    predictions_path = fold_dir / "predictions.csv"
    ollama_export_dir = fold_dir / "ollama_export"

    train_config = load_json(train_config_path)
    resolved = train_config.get("resolved", {})

    entry: Dict[str, Any] = {
        "fold": fold,
        "fold_dir": str(fold_dir.resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_exists": adapter_dir.exists(),
        "fold_manifest_path": resolve_manifest_path(run_config, fold, fold_dir),
        "backend_metadata_path": str(backend_metadata_path.resolve()) if backend_metadata_path.exists() else None,
        "train_config_resolved_path": str(train_config_path.resolve()) if train_config_path.exists() else None,
        "metrics_path": str(metrics_path.resolve()) if metrics_path.exists() else None,
        "predictions_path": str(predictions_path.resolve()) if predictions_path.exists() else None,
        "ollama_export_dir": str(ollama_export_dir.resolve()) if ollama_export_dir.exists() else None,
        "best_epoch": resolved.get("best_epoch"),
        "best_val_macro_f1": resolved.get("best_val_macro_f1"),
    }
    return entry


def build_cv_handoff(run_dir: Path, run_config: Dict[str, Any], folds: List[int]) -> Dict[str, Any]:
    """Monta o manifesto consolidado de handoff da run de CV."""
    backend_cfg = run_config.get("backend", {})
    training_cfg = run_config.get("training", {})
    run_config_path = run_dir / "run_config.json"

    fold_entries = [build_fold_entry(run_config, run_dir, fold) for fold in folds]
    label_set = list(run_config.get("label_set", []))
    if not label_set and folds:
        first_fold_config = load_json(run_dir / f"fold_{folds[0]}" / "train_config_resolved.json")
        label_set = list(first_fold_config.get("label_set", []))

    return {
        "handoff_version": 1,
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "run_config_path": str(run_config_path.resolve()),
        "source_experiment_dir": run_config.get("source_experiment_dir"),
        "dataset_root": run_config.get("dataset_root"),
        "label_set": label_set,
        "base_model_id": run_config.get("base_model_id"),
        "model_family": backend_cfg.get("model_family"),
        "runtime_family": backend_cfg.get("runtime_family"),
        "quantization_mode": backend_cfg.get("quantization_mode"),
        "num_folds_expected": training_cfg.get("num_folds"),
        "train_prompt": training_cfg.get("train_prompt"),
        "max_new_tokens": training_cfg.get("max_new_tokens"),
        "artifact_type": "cv",
        "inference_contract": {
            "recommended_primary_runtime": "hf_transformers",
            "supported_model_sources": ["base", "finetuned"],
            "base_usage": {
                "base_model_id": run_config.get("base_model_id"),
                "adapter_dir": None,
            },
            "finetuned_usage": {
                "base_model_id": run_config.get("base_model_id"),
                "adapter_dir_is_fold_specific": True,
                "adapter_dir_pattern": str((run_dir / "fold_{fold}" / "adapter").resolve()),
            },
        },
        "folds": fold_entries,
    }


def build_final_model_handoff(run_dir: Path, run_config: Dict[str, Any]) -> Dict[str, Any]:
    """Monta o manifesto de handoff de um modelo final unico."""
    backend_cfg = run_config.get("backend", {})
    training_cfg = run_config.get("training", {})
    run_config_path = run_dir / "run_config.json"
    adapter_dir = run_dir / "adapter"

    return {
        "handoff_version": 1,
        "artifact_type": "final_model",
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "run_config_path": str(run_config_path.resolve()),
        "source_experiment_dir": run_config.get("source_experiment_dir"),
        "dataset_root": run_config.get("dataset_root"),
        "label_set": list(run_config.get("label_set", [])),
        "base_model_id": run_config.get("base_model_id"),
        "model_family": backend_cfg.get("model_family"),
        "runtime_family": backend_cfg.get("runtime_family"),
        "quantization_mode": backend_cfg.get("quantization_mode"),
        "train_prompt": training_cfg.get("train_prompt"),
        "max_new_tokens": training_cfg.get("max_new_tokens"),
        "recommended_runtime": "hf_transformers",
        "supported_model_sources": ["base", "finetuned"],
        "base_usage": {
            "base_model_id": run_config.get("base_model_id"),
            "adapter_dir": None,
        },
        "finetuned_usage": {
            "base_model_id": run_config.get("base_model_id"),
            "adapter_dir": str(adapter_dir.resolve()),
            "adapter_exists": adapter_dir.exists(),
        },
        "artifacts": {
            "adapter_dir": str(adapter_dir.resolve()),
            "final_model_manifest_path": str((run_dir / "final_model_manifest.json").resolve())
            if (run_dir / "final_model_manifest.json").exists()
            else None,
            "backend_metadata_path": str((run_dir / "backend_metadata.json").resolve())
            if (run_dir / "backend_metadata.json").exists()
            else None,
            "val_metrics_best_path": str((run_dir / "val_metrics_best.json").resolve())
            if (run_dir / "val_metrics_best.json").exists()
            else None,
            "final_manifest_csv_path": str((run_dir / "final_manifest.csv").resolve())
            if (run_dir / "final_manifest.csv").exists()
            else None,
        },
    }


def main() -> None:
    """Orquestra a geracao do manifesto de handoff."""
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir nao encontrado: {run_dir}")

    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json nao encontrado em: {run_dir}")
    run_config = load_json(run_config_path)
    artifact_type = detect_artifact_type(run_dir, run_config)

    if artifact_type == "final_model":
        handoff = build_final_model_handoff(run_dir, run_config)
    else:
        folds = sorted(set(args.folds)) if args.folds is not None and len(args.folds) > 0 else discover_folds(run_dir)
        if not folds:
            raise ValueError(f"Nenhum fold encontrado em {run_dir}")
        handoff = build_cv_handoff(run_dir, run_config, folds)

    output_path = Path(args.output).resolve() if args.output else run_dir / "inference_handoff.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(handoff, handle, ensure_ascii=False, indent=2)

    print(f"Handoff salvo em: {output_path.resolve()}")


if __name__ == "__main__":
    main()
