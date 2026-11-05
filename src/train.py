"""Train a Ridge model predicting log solubility from molecular descriptors."""

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["mol_weight", "logp", "h_donors", "h_acceptors", "rings"]
TARGET = "log_solubility"


def cv_rmse(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def main():
    df = pd.read_csv("data/molecules.csv")
    # The test part is held out until the final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=SEED
    )

    baseline = cv_rmse(DummyRegressor(strategy="mean"), X_train, y_train)
    ridge = cv_rmse(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X_train, y_train)

    print(f"Baseline CV RMSE: {baseline:.3f}")
    print(f"Ridge CV RMSE:    {ridge:.3f}")


if __name__ == "__main__":
    main()
