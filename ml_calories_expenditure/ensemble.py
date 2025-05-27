from sklearn.linear_model import Ridge
from ml_calories_expenditure.combinations import engineer_features
from ml_calories_expenditure.lib.logger import setup_logger
from ml_calories_expenditure.lib.models.EnsembleModel2 import EnsembleModel2
from ml_calories_expenditure.lib.utils.results import load_hyper_opt_results
from ml_calories_expenditure.utils import PathManager, PrefixManager, load_data

logger = setup_logger(__name__)

results = load_hyper_opt_results(
    model_run="demo",
    output_dir_path=PathManager.output.value,
    hyper_opt_prefix=PrefixManager.hyper.value,
)


logger.info(f"Loaded {len(results)} hyperopt results")

results = [
    res
    for res in results
    if res["name"] is not None
    and res["model"] is not None
    and res["features"] is not None
]

ens = EnsembleModel2(
    models=[res["model"] for res in results],  # type: ignore
    combination_features=[res["features"] for res in results],  # type: ignore
    combination_names=[res["name"] for res in results],  # type: ignore
    task="regression",
    metamodel=Ridge(),
)

train, test = load_data()
eng_train = engineer_features(train, manage_outliers=True)
eng_test = engineer_features(test)

target_column = "Calories"
X_train, y_train = eng_train.drop(columns=[target_column]), eng_train[target_column]
X_test = eng_test

ens.fit(X_train, y_train)
predictions = ens.predict(X_test)
predictions.to_csv(
    PathManager.predictions.value / "ensemble_predictions.csv", index=True
)
