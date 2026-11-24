"""
Linear Regression: predicting diabetes disease progression

Dataset: scikit-learn's built-in Diabetes dataset (442 patients, 10 features).
Target: a quantitative measure of disease progression one year after baseline.
No download needed - the data ships with scikit-learn.

Run:
    python linear_regression.py

Outputs: outputs/metrics.json, outputs/coefficients.csv, outputs/figures/*.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Load data ---------------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame.copy()          # 10 features + 'target'
print(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")
print("Missing values:", int(df.isna().sum().sum()))
print(df.describe().T[["mean", "std", "min", "max"]].round(3))

X = df.drop(columns="target")
y = df["target"]

# 2. EDA ---------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(y, bins=25, kde=True)
plt.title("Distribution of disease progression (target)")
plt.tight_layout(); plt.savefig(FIG / "01_target_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
plt.tight_layout(); plt.savefig(FIG / "02_correlation_heatmap.png", dpi=150); plt.close()

corr_target = df.corr()["target"].drop("target").sort_values(ascending=False)
print("\nCorrelation with target:\n", corr_target.round(3))

# 3. Train / test split and model -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae = mean_absolute_error(y_test, pred)
cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")

metrics = {
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "test_r2": round(float(r2), 3),
    "test_rmse": round(rmse, 2),
    "test_mae": round(float(mae), 2),
    "cv5_r2_mean": round(float(cv.mean()), 3),
    "cv5_r2_std": round(float(cv.std()), 3),
    "baseline_rmse_predict_mean": round(float(np.sqrt(mean_squared_error(
        y_test, np.full_like(y_test, y_train.mean(), dtype=float)))), 2),
}
print("\n=== MODEL RESULTS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# 4. Coefficients (features are already standardised in this dataset) --------
coef = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
coef.rename("coefficient").to_csv(OUT / "coefficients.csv")
print("\nCoefficients (largest effect first):\n", coef.round(1))

plt.figure(figsize=(7, 4.5))
sns.barplot(x=coef.values, y=coef.index, color="#2a6fb0")
plt.title("Linear regression coefficients")
plt.xlabel("Coefficient (effect on progression score)")
plt.tight_layout(); plt.savefig(FIG / "03_coefficients.png", dpi=150); plt.close()

# 5. Evaluation plots --------------------------------------------------------
plt.figure(figsize=(5.5, 5))
plt.scatter(y_test, pred, alpha=0.7)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title(f"Actual vs predicted (R2 = {r2:.2f})")
plt.tight_layout(); plt.savefig(FIG / "04_actual_vs_predicted.png", dpi=150); plt.close()

resid = y_test - pred
plt.figure(figsize=(6, 4))
plt.scatter(pred, resid, alpha=0.7)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Predicted"); plt.ylabel("Residual")
plt.title("Residual plot")
plt.tight_layout(); plt.savefig(FIG / "05_residuals.png", dpi=150); plt.close()

print("\nDone. Figures saved in", FIG)
