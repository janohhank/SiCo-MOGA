from __future__ import annotations

import os
from typing import Any

import numpy
import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from training_utils import ensure_directory


# ---------------------------------------------------------------------------
# Column type detection
# ---------------------------------------------------------------------------

def get_continuous_columns(df: pandas.DataFrame) -> list[str]:
    """Return columns with more than 2 unique numeric values (continuous features)."""
    return [col for col in df.select_dtypes(include=[numpy.number]).columns
            if df[col].nunique() > 2]


def get_dummy_columns(df: pandas.DataFrame) -> list[str]:
    """Return binary columns whose values are a subset of {0, 1}."""
    dummy_cols: list[str] = []
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        if len(unique_vals) <= 2 and unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            dummy_cols.append(col)
    return dummy_cols


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def apply_proportional_noise(
        X_test: pandas.DataFrame,
        train_std: pandas.Series,
        noise_fraction: float,
        mean_shift_fraction: float,
        continuous_cols: list[str]) -> pandas.DataFrame:
    """
    Adds Gaussian noise and/or systematic mean shift to continuous variables,
    proportional to the training-set standard deviation.

    noise_fraction:      0.0-1.0  (scale of additive Gaussian noise)
    mean_shift_fraction: -1.0-1.0 (direction and magnitude of covariate shift)
    """
    no_noise: bool = numpy.isclose(noise_fraction, 0.0, atol=1e-09)
    no_shift: bool = numpy.isclose(mean_shift_fraction, 0.0, atol=1e-09)

    if no_noise and no_shift:
        return X_test.copy()

    X_out: pandas.DataFrame = X_test.copy()

    for col in continuous_cols:
        if col not in X_out.columns or col not in train_std.index:
            continue
        std_val: float = train_std[col]

        if not no_shift:
            X_out[col] += mean_shift_fraction * std_val

        if not no_noise:
            noise: numpy.ndarray = numpy.random.normal(
                loc=0.0, scale=noise_fraction * std_val, size=len(X_out))
            X_out[col] += noise

    return X_out


def apply_dummy_noise(
        X_test: pandas.DataFrame,
        noise_fraction: float,
        dummy_cols: list[str]) -> pandas.DataFrame:
    """
    Bit-flip noise on dummy (binary) variables.
    noise_fraction: probability [0.0-1.0] that any single bit is flipped.
    """
    if numpy.isclose(noise_fraction, 0.0, atol=1e-09):
        return X_test.copy()

    X_out: pandas.DataFrame = X_test.copy()

    for col in dummy_cols:
        if col not in X_out.columns:
            continue
        flip_mask: numpy.ndarray = numpy.random.rand(len(X_out)) < noise_fraction

        if pandas.api.types.is_bool_dtype(X_out[col]):
            X_out.loc[flip_mask, col] = ~X_out.loc[flip_mask, col]
        else:
            X_out.loc[flip_mask, col] = 1 - X_out.loc[flip_mask, col]

    return X_out


# ---------------------------------------------------------------------------
# Model building and evaluation
# ---------------------------------------------------------------------------

def build_model_package(
        individual: list[int],
        feature_names: list[str],
        X_train: pandas.DataFrame,
        y_train: pandas.Series,
        seed: int) -> dict[str, Any]:
    """Retrain a LogisticRegression on the full training set for the features
    selected by *individual* and return a ready-to-evaluate model package."""
    selected_features: list[str] = [
        f for f, bit in zip(feature_names, individual) if bit == 1
    ]

    scaler: StandardScaler = StandardScaler()
    X_scaled: numpy.ndarray = scaler.fit_transform(X_train[selected_features].to_numpy())

    model: LogisticRegression = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(X_scaled, y_train)

    return {"model": model, "scaler": scaler, "features": selected_features}


def evaluate_model(
        model_pkg: dict[str, Any],
        X_test: pandas.DataFrame,
        y_test: pandas.Series,
        use_roc_auc: bool = True) -> float:
    """Score a model package on (possibly noisy) test data."""
    features: list[str] = model_pkg["features"]
    X_scaled: numpy.ndarray = model_pkg["scaler"].transform(
        X_test[features].to_numpy())
    y_prob: numpy.ndarray = model_pkg["model"].predict_proba(X_scaled)[:, 1]

    if use_roc_auc:
        return float(roc_auc_score(y_test, y_prob))
    return float(average_precision_score(y_test, y_prob))


# ---------------------------------------------------------------------------
# Full noise sweep
# ---------------------------------------------------------------------------

def run_noise_evaluation(
        model_pkg_multi: dict[str, Any],
        model_pkg_single: dict[str, Any],
        X_test: pandas.DataFrame,
        y_test: pandas.Series,
        train_std: pandas.Series,
        continuous_cols: list[str],
        dummy_cols: list[str],
        use_roc_auc: bool = True) -> pandas.DataFrame:
    """Run Gaussian-noise + mean-shift sweep and dummy bit-flip sweep,
    returning a tidy DataFrame with all results."""

    noise_levels: numpy.ndarray = numpy.arange(0.0, 1.1, 0.1)
    shift_levels: numpy.ndarray = numpy.arange(-1.0, 1.1, 0.2)
    flip_levels: numpy.ndarray = numpy.arange(0.0, 0.55, 0.05)

    results: list[dict[str, Any]] = []

    # --- Gaussian noise + mean-shift on continuous features ---
    for shift in shift_levels:
        for noise in noise_levels:
            clean_shift: float = round(float(shift), 2)
            clean_noise: float = round(float(noise), 2)

            X_noisy = apply_proportional_noise(
                X_test, train_std, clean_noise, clean_shift, continuous_cols)

            results.append({
                "noise_type": "gaussian",
                "noise_level": clean_noise,
                "mean_shift": clean_shift,
                "flip_rate": 0.0,
                "auc_multi": evaluate_model(model_pkg_multi, X_noisy, y_test, use_roc_auc),
                "auc_single": evaluate_model(model_pkg_single, X_noisy, y_test, use_roc_auc),
            })

    # --- Bit-flip on dummy features ---
    for flip in flip_levels:
        clean_flip: float = round(float(flip), 2)

        X_noisy = apply_dummy_noise(X_test, clean_flip, dummy_cols)

        results.append({
            "noise_type": "dummy_flip",
            "noise_level": 0.0,
            "mean_shift": 0.0,
            "flip_rate": clean_flip,
            "auc_multi": evaluate_model(model_pkg_multi, X_noisy, y_test, use_roc_auc),
            "auc_single": evaluate_model(model_pkg_single, X_noisy, y_test, use_roc_auc),
        })

    return pandas.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_robustness_heatmaps(results_df: pandas.DataFrame, filepath: str) -> None:
    """Side-by-side heatmaps: multi vs single under Gaussian noise + shift."""
    ensure_directory(os.path.dirname(filepath))

    df_gauss = results_df[results_df["noise_type"] == "gaussian"]

    pivot_multi = df_gauss.pivot_table(
        index="mean_shift", columns="noise_level", values="auc_multi")
    pivot_multi = pivot_multi.sort_index(ascending=False)

    pivot_single = df_gauss.pivot_table(
        index="mean_shift", columns="noise_level", values="auc_single")
    pivot_single = pivot_single.sort_index(ascending=False)

    vmin = min(df_gauss["auc_multi"].min(), df_gauss["auc_single"].min())
    vmax = max(df_gauss["auc_multi"].max(), df_gauss["auc_single"].max())

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(pivot_multi, cmap="viridis", vmin=vmin, vmax=vmax, annot=False,
                cbar_kws={"label": "Test AUC"}, ax=axes[0])
    axes[0].set_title("Multi-Objective Model Robustness", fontweight="bold", pad=15)
    axes[0].set_xlabel("Noise Level (Fraction of Train StDev)", fontweight="bold")
    axes[0].set_ylabel("Mean Shift (Fraction of Train StDev)", fontweight="bold")

    sns.heatmap(pivot_single, cmap="viridis", vmin=vmin, vmax=vmax, annot=False,
                cbar_kws={"label": "Test AUC"}, ax=axes[1])
    axes[1].set_title("Single-Objective Model Robustness", fontweight="bold", pad=15)
    axes[1].set_xlabel("Noise Level (Fraction of Train StDev)", fontweight="bold")
    axes[1].set_ylabel("Mean Shift (Fraction of Train StDev)", fontweight="bold")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_dummy_flip_comparison(results_df: pandas.DataFrame, filepath: str) -> None:
    """Line plot: multi vs single AUC under increasing bit-flip rate."""
    ensure_directory(os.path.dirname(filepath))

    df_flip = results_df[results_df["noise_type"] == "dummy_flip"].sort_values("flip_rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_flip["flip_rate"], df_flip["auc_multi"],
            label="Multi-Objective", linewidth=2, marker="o")
    ax.plot(df_flip["flip_rate"], df_flip["auc_single"],
            label="Single-Objective", linewidth=2, marker="s", linestyle="--")
    ax.set_xlabel("Bit-Flip Rate")
    ax.set_ylabel("AUC")
    ax.set_title("Dummy Variable Noise Robustness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_gaussian_noise_comparison(results_df: pandas.DataFrame, filepath: str) -> None:
    """Line plot: multi vs single AUC under Gaussian noise (zero mean-shift)."""
    ensure_directory(os.path.dirname(filepath))

    df_gauss = results_df[
        (results_df["noise_type"] == "gaussian") &
        (numpy.isclose(results_df["mean_shift"], 0.0))
    ].sort_values("noise_level")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_gauss["noise_level"], df_gauss["auc_multi"],
            label="Multi-Objective", linewidth=2, marker="o")
    ax.plot(df_gauss["noise_level"], df_gauss["auc_single"],
            label="Single-Objective", linewidth=2, marker="s", linestyle="--")
    ax.set_xlabel("Noise Level (Fraction of Train StDev)")
    ax.set_ylabel("AUC")
    ax.set_title("Gaussian Noise Robustness (No Mean Shift)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator: evaluate one seed and save everything
# ---------------------------------------------------------------------------

def evaluate_and_save(
        seed: int,
        model_pkg_multi: dict[str, Any],
        model_pkg_single: dict[str, Any],
        X_test: pandas.DataFrame,
        y_test: pandas.Series,
        train_std: pandas.Series,
        continuous_cols: list[str],
        dummy_cols: list[str],
        result_directory: str,
        use_roc_auc: bool = True) -> pandas.DataFrame:
    """Run the full noise evaluation for one seed and persist CSV + plots."""
    results_df: pandas.DataFrame = run_noise_evaluation(
        model_pkg_multi, model_pkg_single,
        X_test, y_test, train_std,
        continuous_cols, dummy_cols, use_roc_auc)

    seed_dir: str = os.path.join(result_directory, f"seed_{seed}")
    ensure_directory(seed_dir)

    results_df.to_csv(os.path.join(seed_dir, "noise_evaluation.csv"), index=False)
    plot_robustness_heatmaps(results_df, os.path.join(seed_dir, "robustness_heatmaps.png"))
    plot_gaussian_noise_comparison(results_df, os.path.join(seed_dir, "gaussian_noise_comparison.png"))
    plot_dummy_flip_comparison(results_df, os.path.join(seed_dir, "dummy_flip_comparison.png"))

    # Print summary for this seed
    clean = results_df[
        (results_df["noise_type"] == "gaussian") &
        (numpy.isclose(results_df["noise_level"], 0.0)) &
        (numpy.isclose(results_df["mean_shift"], 0.0))
    ]
    if not clean.empty:
        row = clean.iloc[0]
        print(f"  Seed {seed} | Multi features: {len(model_pkg_multi['features']):3d} | "
              f"Single features: {len(model_pkg_single['features']):3d} | "
              f"Clean AUC  Multi: {row['auc_multi']:.4f}  Single: {row['auc_single']:.4f}")

    return results_df
