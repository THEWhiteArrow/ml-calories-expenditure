from ml_calories_expenditure.combinations import (
    engineer_combinations_wrapper,
    engineer_features,
)
from ml_calories_expenditure.lib.optymization.analysis_setup import setup_analysis
from ml_calories_expenditure.lib.optymization.hyper_setup import setup_hyper
from ml_calories_expenditure.lib.optymization.optimization_study import (
    aggregate_studies,
)
from ml_calories_expenditure.lib.optymization.parrarel_optimization import (
    HyperFunctionDto,
    HyperSetupDto,
)
from ml_calories_expenditure.objectives import create_objective
from ml_calories_expenditure.utils import PathManager, PrefixManager, load_data


processes = 30
model_run = "demo"
use_models = [
    "RidgeReg",
    "KNeighborsReg",
    "LGBMReg",
    # "RandomForestReg",
    # "SGDReg",
    # "CatBoostReg",
    "XGBReg",
    "HistGradientBoostingReg",
]

train, test = load_data()

# eng = engineer_features(train)
# import matplotlib.pyplot as plt
# # Select features to plot (excluding 'Calories' if present)
# features = [col for col in eng.columns if col != "Calories" and "Age_" not in col]
# n_features = len(features)
# n_cols = 3
# n_rows = (n_features + n_cols - 1) // n_cols
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
# axes = axes.flatten()
# for idx, feature in enumerate(features):
#     axes[idx].scatter(eng[feature], eng["Calories"], alpha=0.5, s=5)
#     axes[idx].set_xlabel(feature)
#     axes[idx].set_ylabel("Calories")
#     axes[idx].set_title(f"Calories vs {feature}")
# # Hide any unused subplots
# for ax in axes[n_features:]:
#     ax.set_visible(False)
# plt.tight_layout()
# plt.show()


setup_dto = HyperSetupDto(
    n_optimization_trials=60,
    optimization_timeout=60 * 60,
    n_patience=20,
    min_percentage_improvement=0.05,
    model_run=model_run,
    processes=processes,
    output_dir_path=PathManager.output.value,
    hyper_opt_prefix=PrefixManager.hyper.value,
    study_prefix=PrefixManager.study.value,
    hyper_direction="maximize",
    omit_names=None,
    force_all_sequential=False,
    metadata=None,
    data=train,
    limit_data_percentage=None,
    combinations=engineer_combinations_wrapper(
        data=train,
        processes=processes,
        use_models=use_models,
        gpu=True,
        combos_limit=None,
    ),
)
func_dto = HyperFunctionDto(
    create_objective_func=create_objective, evaluate_hyperopted_model_func=None
)

n = setup_hyper(
    setup_dto=setup_dto,
    function_dto=func_dto,
)


if n >= 0:
    df = setup_analysis(
        model_run=model_run,
        output_dir_path=PathManager.output.value,
        hyper_opt_prefix=PrefixManager.hyper.value,
        study_prefix=PrefixManager.study.value,
        display_plots=False,
    )

    studies_storage_path = aggregate_studies(
        study_dir_path=PathManager.output.value
        / f"{PrefixManager.study.value}{model_run}"
    )
