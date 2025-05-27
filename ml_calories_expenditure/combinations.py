import pandas as pd
from typing import List, Optional

from ml_calories_expenditure.lib.features.FeatureManager import FeatureManager
from ml_calories_expenditure.lib.features.FeatureSet import FeatureSet
from ml_calories_expenditure.lib.logger import setup_logger
from ml_calories_expenditure.lib.models.HyperOptManager import HyperOptManager
from ml_calories_expenditure.lib.models.HyperOptCombination import HyperOptCombination
from ml_calories_expenditure.lib.models.ModelManager import ModelManager


logger = setup_logger(__name__)


def engineer_features(
    data: pd.DataFrame, manage_outliers: bool = False
) -> pd.DataFrame:
    data = data.copy()
    data["Bmi"] = data["Weight"] / ((data["Height"] / 100) ** 2)

    if manage_outliers is True:
        data.loc[:, "Weight"] = data["Weight"].clip(
            lower=data["Weight"].quantile(0.01),
            upper=data["Weight"].quantile(0.99),
        )
        data.loc[:, "Height"] = data["Height"].clip(
            lower=data["Height"].quantile(0.01),
            upper=data["Height"].quantile(0.99),
        )
        data.loc[:, "Heart_Rate"] = data["Heart_Rate"].clip(
            lower=data["Heart_Rate"].quantile(0.01),
            upper=data["Heart_Rate"].quantile(0.99),
        )

    data["Age_19_30"] = (data["Age"] >= 19) & (data["Age"] <= 30)
    data["Age_31_50"] = (data["Age"] >= 31) & (data["Age"] <= 50)
    data["Age_51_70"] = (data["Age"] >= 51) & (data["Age"] <= 70)
    data["Age_71_plus"] = data["Age"] >= 71

    multiplication_pairs = [
        ("Height", "Body_Temp"),
        ("Weight", "Body_Temp"),
        # heart rate
        ("Heart_Rate", "Duration"),
        ("Heart_Rate", "Duration", "Age"),
        ("Heart_Rate", "Body_Temp", "Duration"),
    ]

    for pair in multiplication_pairs:
        feature_name = "_".join(pair)
        data[feature_name] = data[list(pair)].prod(axis=1)

    return data.set_index("id")


def engineer_feature_selection_manual() -> List[FeatureSet]:
    logger.info("Engineering feature selection manual")
    ans: List[FeatureSet] = []
    ans = [
        FeatureSet(
            name="exmandatory",
            features=[
                "Sex",
                "Height",
                "Weight",
                "Duration",
                "Body_Temp",
                "Heart_Rate",
            ],
            is_optional=False,
            is_standalone=False,
        ),
        FeatureSet(
            name="age_groups",
            features=[
                "Age_19_30",
                "Age_31_50",
                "Age_51_70",
                "Age_71_plus",
            ],
            is_optional=True,
            is_standalone=False,
            bans=["age_numeric"],
        ),
        FeatureSet(
            name="age_numeric",
            features=["Age"],
            is_optional=True,
            is_standalone=False,
            bans=["age_groups"],
        ),
        FeatureSet(
            name="multi_weight",
            features=[
                "Bmi",
                "Height_Body_Temp",
                "Weight_Body_Temp",
            ],
            is_optional=True,
            is_standalone=False,
        ),
        FeatureSet(
            name="multi_heart_rate",
            features=[
                "Heart_Rate_Duration",
                "Heart_Rate_Duration_Age",
                "Heart_Rate_Body_Temp_Duration",
            ],
            is_optional=True,
            is_standalone=False,
        ),
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
