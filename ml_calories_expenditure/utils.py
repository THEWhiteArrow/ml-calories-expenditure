import os
from enum import Enum
from typing import Tuple
from pathlib import Path

import pandas as pd


class PathManager(Enum):
    cwd = Path(os.getcwd())
    data = cwd / "data"
    output = cwd / "output"
    predictions = output / "predictions"
    trades = output / "trades"
    errors = output / "errors"


for path in PathManager:
    if not path.value.exists():
        path.value.mkdir(parents=True, exist_ok=True)


class PrefixManager(Enum):
    hyper = "hyper_opt_"
    ensemble = "ensemble_"
    study = "study_"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PathManager.data.value / "train.csv")
    test = pd.read_csv(PathManager.data.value / "test.csv")

    return train, test
