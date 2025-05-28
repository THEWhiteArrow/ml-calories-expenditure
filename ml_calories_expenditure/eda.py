from autofeat import FeatureSelector, AutoFeatRegressor
import pandas as pd
import pickle as pkl

from ml_calories_expenditure.utils import PathManager, load_data


train, test = load_data()


train = train.sample(frac=0.20, random_state=42).set_index(
    "id"
)  # Reduce dataset size for testing

X_raw = train.drop(columns=["Calories"])
X = pd.get_dummies(X_raw, drop_first=True)
y = train["Calories"]

selector = FeatureSelector()
feateng_steps = 2
model = AutoFeatRegressor(n_jobs=1, feateng_steps=feateng_steps)


# X_selected = selector.fit_transform(X, y)  # type: ignore
# print(f"Selected features: {X_selected.columns.tolist()} | features left out: {list(set(X.columns.tolist()) - set(X_selected.columns.tolist()))}")  # type: ignore

X_created = model.fit_transform(X, y)  # type: ignore
print("Created features:", X_created.columns.tolist())  # type: ignore

pkl.dump(
    model,
    open(PathManager.output.value / f"autofeat_model_{feateng_steps}.pkl", "wb"),
)


print("done")
