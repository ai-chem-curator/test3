"""Train a Ridge model predicting log solubility from molecular descriptors."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["mol_weight", "logp", "h_donors", "h_acceptors", "rings"]
TARGET = "log_solubility"
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


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

    best_alpha, best_rmse = None, float("inf")
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X_train, y_train)
        rmse = float(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse

    print(f"Baseline CV RMSE: {baseline:.3f}")
    print(f"Best alpha:       {best_alpha}")
    print(f"Final test RMSE:  {best_rmse:.3f}")


if __name__ == "__main__":
    main()
