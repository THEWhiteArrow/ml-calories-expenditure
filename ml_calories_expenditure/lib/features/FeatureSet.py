from dataclasses import dataclass

from ml_calories_expenditure.lib.features.FeatureCombination import FeatureCombination


@dataclass
class FeatureSet(FeatureCombination):
    is_optional: bool = True
    is_exclusive: bool = False
    is_exclusive_mandatory: bool = False
