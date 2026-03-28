from __future__ import annotations

import argparse
from pathlib import Path

from src.backend_multimodelo_soja.experiment.aggregation import aggregate_run
from src.backend_multimodelo_soja.experiment.logging_utils import setup_logger


def parse_args():
    """Processa os argumentos da agregacao da CV.

    Parametros:
        Nenhum.

    Retorno:
        argparse.Namespace: Argumentos contendo o diretorio da run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main():
    """Executa a agregacao final de uma run completa.

    Parametros:
        Nenhum.

    Retorno:
        None: Produz tabelas e resumos agregados.
    """
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    logger = setup_logger(run_dir / "aggregate.log", "aggregate_cv")
    aggregate_run(run_dir, logger)


if __name__ == "__main__":
    main()
