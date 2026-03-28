from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.backend_multimodelo_soja.experiment.config import build_run_dir, load_config
from src.backend_multimodelo_soja.experiment.logging_utils import setup_logger


def parse_args():
    """Processa os argumentos da execucao completa de CV.

    Parametros:
        Nenhum.

    Retorno:
        argparse.Namespace: Argumentos com caminho do config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    """Executa todos os folds e a agregacao final da CV.

    Parametros:
        Nenhum.

    Retorno:
        None: Gera a run completa de validacao cruzada.
    """
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    run_dir = build_run_dir(config)
    logger = setup_logger(run_dir / "runtime.log", "run_cv")
    logger.info("Diretorio da run: %s", run_dir)

    run_config = {
        "project_name": config["project_name"],
        "base_model_id": config["base_model_id"],
        "dataset_root": config["dataset_root"],
        "source_experiment_dir": config["source_experiment_dir"],
        "output_root": config["output_root"],
        "backend": config["backend"],
        "training": config["training"],
        "memory": config["memory"],
        "lora": config["lora"],
        "pilot_mode": config["pilot_mode"],
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    for fold in range(int(config["training"]["num_folds"])):
        logger.info("Executando fold %s", fold)
        command = [
            sys.executable,
            "train_model_fold.py",
            "--config",
            str(Path(args.config).resolve()),
            "--fold",
            str(fold),
            "--run-dir",
            str(run_dir),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            logger.error("Fold %s falhou com exit code %s. Abortando CV.", fold, result.returncode)
            raise SystemExit(result.returncode)

    aggregate_command = [
        sys.executable,
        "aggregate_model_cv.py",
        "--run-dir",
        str(run_dir),
    ]
    result = subprocess.run(aggregate_command, check=False)
    if result.returncode != 0:
        logger.error("Agregacao final falhou com exit code %s", result.returncode)
        raise SystemExit(result.returncode)

    logger.info("CV completa concluida com sucesso em %s", run_dir)


if __name__ == "__main__":
    main()
