from ml_calories_expenditure.combinations import engineer_combinations_wrapper
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
    "LGBMReg",
    "SGDReg",
    "RidgeReg",
    "KNeighborsReg",
    "CatBoostReg",
    "RandomForestReg",
    "XGBReg",
    "HistGradientBoostingReg",
]

train, test = load_data()
setup_dto = HyperSetupDto(
    n_optimization_trials=100,
    optimization_timeout=60 * 60,
    n_patience=30,
    min_percentage_improvement=0.01,
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
    limit_data_percentage=0.5,
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
