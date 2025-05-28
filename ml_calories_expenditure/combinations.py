import pandas as pd
import pickle as pkl
from typing import List, Optional, cast

from autofeat import AutoFeatRegressor

from ml_calories_expenditure.lib.features.FeatureManager import FeatureManager
from ml_calories_expenditure.lib.features.FeatureSet import FeatureSet
from ml_calories_expenditure.lib.logger import setup_logger
from ml_calories_expenditure.lib.models.HyperOptManager import HyperOptManager
from ml_calories_expenditure.lib.models.HyperOptCombination import HyperOptCombination
from ml_calories_expenditure.lib.models.ModelManager import ModelManager
from ml_calories_expenditure.utils import PathManager


logger = setup_logger(__name__)


def engineer_features(
    data: pd.DataFrame, manage_outliers: bool = False
) -> pd.DataFrame:
    data = data.copy().set_index("id")

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

    data.loc[:, "Sex_male"] = data["Sex"].eq("male").astype(int)
    data = data.drop(columns=["Sex"])

    # NOTE: autofeat features
    autofeat2 = cast(
        AutoFeatRegressor,
        pkl.load(open(PathManager.cwd.value / "autofeat_model_2.pkl", "rb")),
    )
    autofeat3 = cast(
        AutoFeatRegressor,
        pkl.load(open(PathManager.cwd.value / "autofeat_model_3.pkl", "rb")),
    )

    datafeat2 = autofeat2.transform(data.drop(columns=["Calories"]))
    datafeat2.index = data.index  # type: ignore

    datafeat3 = autofeat3.transform(data.drop(columns=["Calories"]))
    datafeat3.index = data.index  # type: ignore

    # NOTE: merge autofeat features with the original data but drop the duplicated columns
    merged_data = pd.concat(
        [data, datafeat2, datafeat3],  # type: ignore
        axis=1,
        ignore_index=False,
    )
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

    merged_data["Bmi"] = merged_data["Weight"] / ((merged_data["Height"] / 100) ** 2)
    multiplication_pairs = [
        ("Height", "Body_Temp"),
        ("Weight", "Body_Temp"),
        ("Heart_Rate", "Duration"),
        ("Heart_Rate", "Duration", "Age"),
        ("Heart_Rate", "Body_Temp", "Duration"),
    ]

    for pair in multiplication_pairs:
        feature_name = "_".join(pair)
        merged_data[feature_name] = merged_data[list(pair)].prod(axis=1)

    return merged_data


def engineer_feature_selection_manual() -> List[FeatureSet]:
    logger.info("Engineering feature selection manual")
    ans: List[FeatureSet] = []
    ans = [
        FeatureSet(
            name="mandatory",
            features=[
                "Age",
                "Sex_male",
                "Height",
                "Weight",
                "Duration",
                "Body_Temp",
                "Heart_Rate",
            ],
            is_optional=False,
        ),
        FeatureSet(
            name="multi_heart_rate",
            features=[
                "Height_Body_Temp",
                "Weight_Body_Temp",
                "Heart_Rate_Duration",
                "Heart_Rate_Duration_Age",
                "Heart_Rate_Body_Temp_Duration",
            ],
            is_optional=True,
            is_standalone=False,
        ),
        FeatureSet(
            name="auto_feat_selection",
            features=["Age", "Weight", "Duration", "Sex_male", "Heart_Rate"],
            is_exclusive=True,
        ),
        FeatureSet(
            name="auto_feat_creation_2",
            features=[
                "Age",
                "Height",
                "Weight",
                "Duration",
                "Heart_Rate",
                "Body_Temp",
                "Sex_male",
                "1/Duration",
                "Weight/Age",
                "Age/Duration",
                "Duration/Age",
                "Age*Sex_male",
                "Age*Weight**3",
                "1/(Age*Weight)",
                "Age**3*Duration",
                "Age*Duration**3",
                "Age**3*Sex_male",
                "Heart_Rate/Weight",
                "Height/Heart_Rate",
                "Sex_male/Duration",
                "Duration*Sex_male",
                "Weight**3/Duration",
                "Duration*Weight**2",
                "Body_Temp**3/Height",
                "Heart_Rate**3*Weight",
                "Heart_Rate*Height**2",
                "Body_Temp**3*log(Age)",
                "Heart_Rate**2*log(Age)",
                "log(Duration)/Duration",
                "sqrt(Age)*log(Duration)",
                "Body_Temp**3/Heart_Rate",
                "log(Heart_Rate)/Body_Temp",
                "Sex_male/Age",
                "Height**2/Weight",
                "Height**3/Weight",
                "log(Age)/Weight",
            ],
            is_exclusive=True,
        ),
        FeatureSet(
            name="auto_feat_creation_3",
            features=[
                "Age",
                "Height",
                "Weight",
                "Duration",
                "Heart_Rate",
                "Body_Temp",
                "Sex_male",
                "Duration**3*Weight**(3/2)",
                "Abs(Duration - sqrt(Weight))",
                "Sex_male/Age",
                "Age**4*Sex_male**2",
                "Heart_Rate**3*Weight",
                "Heart_Rate**3/Weight",
                "log(Duration)/Duration",
                "sqrt(Age)*Heart_Rate**3",
                "exp(sqrt(Heart_Rate))/Age",
                "Duration**(3/2)*log(Age)**3",
                "exp(sqrt(Age) - sqrt(Duration))",
                "(Duration**3 - Heart_Rate**2)**2",
                "1/(Duration**3 - sqrt(Heart_Rate))",
                "exp(-sqrt(Duration) + sqrt(Heart_Rate))",
                "1/(Age**2 + Weight**2)",
                "Duration**6*log(Heart_Rate)**3",
                "1/(-sqrt(Duration) + sqrt(Weight))",
                "1/(Duration**2 - sqrt(Height))",
                "(-sqrt(Age) + sqrt(Duration))**3",
                "(-sqrt(Duration) + sqrt(Weight))**3",
                "Body_Temp**3/Height",
                "exp(-Duration + Sex_male)",
                "1/(-Duration**3 + Height)",
                "1/(-Duration**3 + Heart_Rate)",
            ],
            is_exclusive=True,
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
