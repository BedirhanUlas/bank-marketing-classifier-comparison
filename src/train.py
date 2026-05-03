"""
Bank Marketing Classifier — Training Pipeline

Compares Logistic Regression, Decision Tree, KNN, and SVM classifiers
on the UCI Bank Marketing dataset to predict term deposit subscription.
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "bank-additional-full.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> pd.DataFrame:
    """Load bank marketing dataset from CSV.

    Args:
        path: Path to the semicolon-delimited CSV file.

    Returns:
        Raw DataFrame with all 21 columns.

    Raises:
        FileNotFoundError: If the dataset is not found at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path, sep=";")
    logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    assert "y" in df.columns, "Target column 'y' missing from dataset."
    return df


def explore(df: pd.DataFrame) -> None:
    """Log key dataset statistics."""
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Class distribution:\n{df['y'].value_counts()}")
    logger.info(f"Positive rate: {(df['y'] == 'yes').mean():.1%}")


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode categorical features and scale numerics.

    Strategy:
        - Binary encode target (yes → 1, no → 0)
        - One-hot encode all categorical columns
        - StandardScaler on numeric columns

    Returns:
        X (features array), y (target array), feature_names (list).
    """
    df = df.copy()
    y = (df["y"] == "yes").astype(int).values
    df = df.drop(columns=["y"])

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    feature_names = df_encoded.columns.tolist()

    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    X = df_encoded.values
    logger.info(f"Features after encoding: {X.shape[1]}")
    return X, y, feature_names


def build_pipelines() -> Dict[str, Pipeline]:
    """Return sklearn Pipelines for each classifier."""
    return {
        "Logistic Regression": Pipeline([
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ]),
        "SVM": Pipeline([
            ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Train a pipeline and return evaluation metrics.

    Args:
        name: Human-readable model name.
        pipeline: Untrained sklearn Pipeline.
        X_train, X_test, y_train, y_test: Split arrays.

    Returns:
        Dict with accuracy, precision, recall, f1, train_time_s.
    """
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = round(time.perf_counter() - t0, 2)

    y_pred = pipeline.predict(X_test)

    return {
        "Model": name,
        "Train Accuracy": round(pipeline.score(X_train, y_train), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "Train Time (s)": train_time,
    }


def cross_validate_models(
    pipelines: Dict[str, Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> pd.DataFrame:
    """Run stratified k-fold cross-validation on all models.

    Args:
        pipelines: Dict of model name → Pipeline.
        X, y: Full feature/target arrays (before train/test split).
        cv: Number of folds.

    Returns:
        DataFrame with mean and std CV accuracy per model.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        rows.append({
            "Model": name,
            f"CV Accuracy (mean ± std, {cv}-fold)": f"{scores.mean():.4f} ± {scores.std():.4f}",
        })
        logger.info(f"[CV] {name}: {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows)


def plot_results(results_df: pd.DataFrame, save_dir: Path) -> None:
    """Save a bar chart of test accuracy by model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        results_df["Model"],
        results_df["Test Accuracy"],
        color=sns.color_palette("muted"),
        edgecolor="black",
    )
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0.80, 0.95)
    ax.set_title("Test Accuracy by Classifier — Bank Marketing Dataset", fontsize=13)
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Model")
    ax.axhline(y=results_df["Test Accuracy"].iloc[0], color="gray", linestyle="--", label="Baseline")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = save_dir / "accuracy_comparison.png"
    fig.savefig(out, dpi=150)
    logger.info(f"Chart saved → {out}")
    plt.close(fig)


def main() -> None:
    logger.info("=== Bank Marketing Classifier Pipeline ===")

    df = load_data(DATA_PATH)
    explore(df)

    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_acc = dummy.score(X_test, y_test)
    logger.info(f"Baseline (majority class): {baseline_acc:.4f}")

    pipelines = build_pipelines()
    results = []
    for name, pipe in pipelines.items():
        logger.info(f"Training: {name}...")
        metrics = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results.append(metrics)
        logger.info(f"  Test Accuracy: {metrics['Test Accuracy']} | F1: {metrics['F1']}")

    results_df = pd.DataFrame(results)
    results_df = pd.concat(
        [
            pd.DataFrame([{"Model": "Dummy Baseline", "Test Accuracy": round(baseline_acc, 4)}]),
            results_df,
        ],
        ignore_index=True,
    )

    logger.info("\n=== Results ===\n" + results_df.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    cv_df = cross_validate_models(pipelines, X, y)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

    plot_results(results_df.iloc[1:], RESULTS_DIR)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
