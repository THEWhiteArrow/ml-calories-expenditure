import pandas as pd
from typing import List, Optional

from ml_calories_expenditure.lib.features.FeatureManager import FeatureManager
from ml_calories_expenditure.lib.features.FeatureSet import FeatureSet
from ml_calories_expenditure.lib.logger import setup_logger
from ml_calories_expenditure.lib.models.HyperOptManager import HyperOptManager
from ml_calories_expenditure.lib.models.HyperOptCombination import HyperOptCombination
from ml_calories_expenditure.lib.models.ModelManager import ModelManager


logger = setup_logger(__name__)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["bmi"] = data["Weight"] / ((data["Height"] / 100) ** 2)

    return data.set_index("id")


def engineer_feature_selection_manual() -> List[FeatureSet]:
    logger.info("Engineering feature selection manual")
    ans: List[FeatureSet] = []
    ans = [
        FeatureSet(
            name="basic",
            features=[
                "Sex",
                "Age",
                "Height",
                "Weight",
                "Duration",
                "Body_Temp",
                "bmi",
            ],
            is_optional=False,
        )
    ]

    return ans


def engineer_combinations_wrapper(
    data: pd.DataFrame,
    processes: Optional[int] = None,
    use_models: Optional[List[str]] = None,
    gpu: bool = True,
    combos_limit: Optional[int] = None,
) -> List[HyperOptCombination]:

    feature_sets: List[FeatureSet] = []

    feature_sets.extend(engineer_feature_selection_manual())

    feature_manager = FeatureManager(feature_sets=feature_sets)

    model_manager = ModelManager()

    hyper_manager = HyperOptManager(
        feature_manager=feature_manager,
        models=model_manager.get_models(
            processes=processes,
            use_models=use_models,
            gpu=gpu,
        ),
    )
    hyper_combinations = hyper_manager.get_model_combinations()

    final_hyper_combinations = [
        combo for combo in hyper_combinations if is_valid_combo(combo)
    ]

    if combos_limit is not None:
        final_hyper_combinations = final_hyper_combinations[::-1][:combos_limit]

    return final_hyper_combinations


def is_valid_combo(combo: HyperOptCombination) -> bool:

    for tree_method in ["rfe_XGBClassifier", "rfe_LGBMClassifier"]:
        if tree_method in combo.name:
            if combo.name[:3] not in ["XGB", "LGB"]:
                return False

    for num_method in ["rfe_RidgeClassifier", "_f_classif", "random"]:
        if num_method in combo.name:
            if combo.name[:3] in ["XGB", "LGB"]:
                return False

    for xgb_delimitation in ["_225"]:
        if xgb_delimitation in combo.name:
            if combo.name[:3] in ["XGB"]:
                return False

    for ridge_delimitation in ["mutual_info"]:
        if ridge_delimitation in combo.name:
            if combo.name[:3] in ["Rid"]:
                return False

    return True
