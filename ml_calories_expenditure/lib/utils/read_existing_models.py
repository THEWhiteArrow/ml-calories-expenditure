import os
from pathlib import Path
import pickle
from typing import List, Optional

from ml_calories_expenditure.lib.logger import setup_logger
from ml_calories_expenditure.lib.models.HyperOptResultDict import HyperOptResultDict


logger = setup_logger(__name__)


def read_existing_models(path: Path) -> List[str]:
    existing_models = []

    if not os.path.exists(path):
        return existing_models

    for file in os.listdir(path):
        if file.endswith(".pkl"):
            existing_models.append(file.split(".")[0])

    return existing_models


def read_hyper_results(
    path: Path, selection: Optional[List[str]] = None
) -> List[HyperOptResultDict]:

    hyper_opt_results: List[HyperOptResultDict] = []
    for model_file in os.listdir(path):
        if model_file.endswith(".pkl") and (
            selection is None or model_file.split(".")[0] in selection
        ):
            model_data = pickle.load(open(path / model_file, "rb"))
            model_data["name"] = model_file.split(".")[0]
            hyper_opt_results.append(model_data)

    if len(hyper_opt_results) == 0:
        raise ValueError("No models found in the specified path.")

    if selection is not None and len(hyper_opt_results) != len(selection):
        logger.error(
            f"Models not found: {list(set(selection) - set([model['name'] for model in hyper_opt_results]))}"
        )
        raise ValueError("Not all models were found in the specified path.")

    return hyper_opt_results
