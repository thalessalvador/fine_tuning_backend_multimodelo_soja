from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.backend_multimodelo_soja.experiment.config import build_auto_run_dir, load_config
from src.backend_multimodelo_soja.experiment.fold_runner import run_fold
from src.backend_multimodelo_soja.experiment.logging_utils import setup_logger


def parse_args():
    """Processa os argumentos de linha de comando do treino por fold.

    Parametros:
        Nenhum.

    Retorno:
        argparse.Namespace: Argumentos com config, fold e run-dir opcional.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-dir")
    return parser.parse_args()


def main():
    """Orquestra a execucao isolada de um fold.

    Parametros:
        Nenhum.

    Retorno:
        None: Cria a pasta do fold e executa o backend configurado.
    """
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    run_dir = Path(args.run_dir).resolve() if args.run_dir else build_auto_run_dir(config, fold=args.fold)
    fold_output_dir = run_dir / f"fold_{args.fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=False)

    logger = setup_logger(fold_output_dir / "runtime.log", f"train_fold_{args.fold}")
    logger.info("Inicio do fold %s", args.fold)
    logger.info("Diretorio da run: %s", run_dir)

    fold_run_config = {
        "project_name": config["project_name"],
        "base_model_id": config["base_model_id"],
        "fold": args.fold,
        "dataset_root": config["dataset_root"],
        "source_experiment_dir": config["source_experiment_dir"],
        "pilot_mode": config["pilot_mode"],
        "backend": config["backend"],
        "training": config["training"],
        "memory": config["memory"],
        "lora": config["lora"],
    }
    with (fold_output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_run_config, handle, ensure_ascii=False, indent=2)

    run_fold(config=config, fold=args.fold, fold_output_dir=fold_output_dir, logger=logger)
    logger.info("Fold %s concluido com sucesso", args.fold)


if __name__ == "__main__":
    main()
