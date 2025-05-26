import json
from pathlib import Path
import signal
from typing import Callable, Dict, List, Optional
import multiprocessing as mp


import pandas as pd

from ml_calories_expenditure.lib.models.HyperOptCombination import HyperOptCombination
from ml_calories_expenditure.lib.optymization.optimization_study import (
    CREATE_OBJECTIVE_TYPE,
    OPTUNA_DIRECTION_TYPE,
    optimize_model_and_save,
)
from ml_calories_expenditure.lib.logger import setup_logger

logger = setup_logger(__name__)


def init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_parallel_optimization(
    model_run: str,
    direction: OPTUNA_DIRECTION_TYPE,
    all_model_combinations: List[HyperOptCombination],
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    n_optimization_trials: int,
    optimization_timeout: Optional[int],
    n_patience: int,
    min_percentage_improvement: float,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    study_prefix: str,
    create_objective: CREATE_OBJECTIVE_TYPE,
    omit_names: List[str] = [],
    processes: Optional[int] = None,
    metadata: Optional[Dict] = None,
    evaluate_hyperopted_model_func: Optional[Callable] = None,
    force_all_sequential: bool = False,
) -> None:
    if processes is None:
        processes = mp.cpu_count()

    logger.info(f"Running optimization with {processes} processes")

    # IMPORTANT NOTE: paralelism depends on the classification categories amount
    # for binary outputs it is not worth to run parallel optimization
    parallel_model_prefixes = [
        "lgbm",
        "ridge",
        "sv",
        "kn",
        "logistic",
        "passiveaggressive",
        "sgd",
        "minibatchsgd",
        "logistic",
        "neuralnetworkcustommodel",
    ]
    parallel_3rd_model_prefixes = []
    omit_mulit_sufixes = ["top_0", "top_1", "top_2", ""]

    sequential_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if all(  # type: ignore
            not model_combination.name.lower().startswith(prefix.lower())
            for prefix in parallel_model_prefixes + parallel_3rd_model_prefixes
        )
        and all(
            f"{model_combination.name}{omit_sufix}" not in omit_names
            for omit_sufix in omit_mulit_sufixes
        )
    ]

    parallel_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if any(  # type: ignore
            model_combination.name.lower().startswith(prefix.lower())
            for prefix in parallel_model_prefixes
        )
        and all(
            f"{model_combination.name}{omit_sufix}" not in omit_names
            for omit_sufix in omit_mulit_sufixes
        )
    ]

    parallel_3rd_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if any(  # type: ignore
            model_combination.name.lower().startswith(prefix.lower())
            for prefix in parallel_3rd_model_prefixes
        )
        and all(
            f"{model_combination.name}{omit_sufix}" not in omit_names
            for omit_sufix in omit_mulit_sufixes
        )
    ]

    if force_all_sequential is True:
        sequential_model_combinations.extend(
            parallel_model_combinations + parallel_3rd_model_combinations
        )
        parallel_model_combinations = []
        parallel_3rd_model_combinations = []

    logger.info(
        "Will be running parallel optimization for models: "
        + json.dumps([model.name for model in parallel_model_combinations], indent=4)
    )

    logger.info(
        "Will be running parallel 1/3rd optimization for models: "
        + json.dumps(
            [model.name for model in parallel_3rd_model_combinations], indent=4
        )
    )
    logger.info(
        "Will be running sequential optimization for models: "
        + json.dumps([model.name for model in sequential_model_combinations], indent=4)
    )

    if len(parallel_model_combinations) == 0:
        logger.info("No parallel models to optimize")
    else:
        # Set up multiprocessing pool
        with mp.Pool(processes=processes, initializer=init_worker) as pool:
            # Map each iteration of the loop to a process
            _ = pool.starmap(
                optimize_model_and_save,
                [
                    (
                        model_run,
                        direction,
                        model_combination,
                        X,
                        y,
                        n_optimization_trials,
                        optimization_timeout,
                        n_patience,
                        min_percentage_improvement,
                        i,
                        output_dir_path,
                        hyper_opt_prefix,
                        study_prefix,
                        create_objective,
                        metadata,
                        evaluate_hyperopted_model_func,
                    )
                    for i, model_combination in enumerate(parallel_model_combinations)
                    if model_combination.name not in omit_names
                ],
            )

    if len(parallel_3rd_model_combinations) == 0:
        logger.info("No parallel 1/3rd models to optimize")
    else:
        with mp.Pool(processes=processes // 3, initializer=init_worker) as pool:
            # Map each iteration of the loop to a process
            _ = pool.starmap(
                optimize_model_and_save,
                [
                    (
                        model_run,
                        direction,
                        model_combination,
                        X,
                        y,
                        n_optimization_trials,
                        optimization_timeout,
                        n_patience,
                        min_percentage_improvement,
                        i,
                        output_dir_path,
                        hyper_opt_prefix,
                        study_prefix,
                        create_objective,
                        metadata,
                        evaluate_hyperopted_model_func,
                    )
                    for i, model_combination in enumerate(
                        parallel_3rd_model_combinations
                    )
                    if model_combination.name not in omit_names
                ],
            )

    if len(sequential_model_combinations) == 0:
        logger.info("No sequential models to optimize")
    else:
        for i, model_combination in enumerate(sequential_model_combinations):
            if model_combination.name in omit_names:
                continue

            optimize_model_and_save(
                model_run,
                direction,
                model_combination,
                X,
                y,
                n_optimization_trials,
                optimization_timeout,
                n_patience,
                min_percentage_improvement,
                i,
                output_dir_path,
                hyper_opt_prefix,
                study_prefix,
                create_objective,
                metadata,
                evaluate_hyperopted_model_func,
            )
