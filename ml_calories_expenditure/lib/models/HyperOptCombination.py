from dataclasses import dataclass

from sklearn.base import BaseEstimator

from ml_calories_expenditure.lib.features.FeatureCombination import FeatureCombination


@dataclass
class HyperOptCombination:
    name: str
    model: BaseEstimator
    feature_combination: FeatureCombination
