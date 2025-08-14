import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import joblib

# ---------- 1) Load Data ----------
df = pd.read_csv("data/heart.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------- 2) Quick EDA ----------
print("Shape:", df.shape)
print(df["output"].value_counts(normalize=True))

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.show()

# ---------- 3) Split ----------
X = df.drop("output", axis=1)
y = df["output"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 4) Define Models & Params ----------
models_params = {
    "LogisticRegression": {
        "model": LogisticRegression(solver="liblinear", max_iter=1000),
        "params": {"clf__C": [0.01, 0.1, 1, 10, 100]}
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear","rbf"]}
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {"clf__max_depth": [None,3,5,7,10]}
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"clf__n_neighbors": [3,5,7,9]}
    }
}

results = []

# ---------- 5) Loop over models ----------
for name, mp in models_params.items():
    print(f"\n===== {name} =====")
    pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", mp["model"])] )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid=mp["params"], cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    joblib.dump(best_model, f"models/{name}_pipeline.joblib")
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    acc = (y_pred==y_test).mean()
    
    print("Best Params:", grid.best_params_)
    print(f"Test Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}, PR-AUC: {ap:.3f}")
    
    results.append({"Model": name, "Accuracy": acc, "ROC-AUC": auc, "PR-AUC": ap})
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"plots/{name}_confusion_matrix.png")
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.2f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"plots/{name}_roc_curve.png")
    plt.show()
    
    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(5,5))
    plt.plot(rec, prec, label=f"AP={ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{name}_pr_curve.png")
    plt.show()

# ---------- 6) Compare Models ----------
df_results = pd.DataFrame(results)
print("\nModel Comparison:\n", df_results)

df_results.set_index("Model")[["ROC-AUC","PR-AUC"]].plot(kind="bar", figsize=(8,5), color=["skyblue","orange"])
plt.title("Model Comparison on Test Set")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/model_comparison.png")
plt.show()
