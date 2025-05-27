import optuna
import pandas as pd
from sklearn import clone
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_log_error

from ml_calories_expenditure.combinations import engineer_features
from ml_calories_expenditure.lib.logger import setup_logger
from ml_calories_expenditure.lib.optymization.TrialParamWrapper import TrialParamWrapper
from ml_calories_expenditure.lib.optymization.optimization_study import (
    OBJECTIVE_RETURN_TYPE,
    OBJECTIVE_FUNC_TYPE,
)
from ml_calories_expenditure.lib.models.HyperOptCombination import HyperOptCombination
from ml_calories_expenditure.lib.pipelines.ProcessingPipelineWrapper import (
    create_pipeline,
)
from ml_calories_expenditure.lib.utils.garbage_collector import garbage_manager
from ml_calories_expenditure.lib.utils.features_utils import correlation_simplification

logger = setup_logger(__name__)


def create_objective(
    data: pd.DataFrame,
    model_combination: HyperOptCombination,
) -> OBJECTIVE_FUNC_TYPE:

    target_name = "Calories"
    engineered_data = engineer_features(data, manage_outliers=True)

    model = model_combination.model
    if model.__class__.__name__ in ("passive", "logistic", "sgd"):

        uncorrelated_features, correlated_features = correlation_simplification(
            engineered_data=engineered_data,
            features_in=model_combination.feature_combination.features,
            threshold=0.85,
        )
        model_combination.feature_combination.features = uncorrelated_features

    def objective(trial: optuna.Trial) -> OBJECTIVE_RETURN_TYPE:
        params = TrialParamWrapper().get_params(
            model_name=model_combination.name, trial=trial
        )

        pipeline = create_pipeline(
            model=clone(model).set_params(**params),
            features_in=model_combination.feature_combination.features,
        )

        try:
            kfold = KFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )

            oof_predictions = pd.Series(index=engineered_data.index, name=target_name)
            for train_index, test_index in kfold.split(engineered_data):
                X_train, X_test = (
                    engineered_data.iloc[train_index],
                    engineered_data.iloc[test_index],
                )
                y_train, _ = (
                    engineered_data[target_name].iloc[train_index],
                    engineered_data[target_name].iloc[test_index],
                )
                pipeline.fit(X_train, y_train)
                fold_predictions = pipeline.predict(X_test)

                oof_predictions.iloc[test_index] = fold_predictions

            score_dict = {
                "rmsle": float(
                    root_mean_squared_log_error(
                        y_true=engineered_data[target_name],
                        y_pred=oof_predictions.clip(lower=0.0),
                    )
                )
            }

            garbage_manager.clean()
            score = score_dict["rmsle"]
            return -score

        except optuna.exceptions.TrialPruned as e:
            logger.warning(f"Trial {trial.number} pruned: {e}")
            raise e

        except Exception as e:
            logger.error(f"Error in objective: {e}")
            raise e

    return objective
