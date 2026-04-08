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
        unique_vals: set[Any] = set(df[col].dropna().unique())
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

            X_noisy: pandas.DataFrame = apply_proportional_noise(
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

    df_gauss: pandas.DataFrame = results_df[results_df["noise_type"] == "gaussian"]

    pivot_multi: pandas.DataFrame = df_gauss.pivot_table(
        index="mean_shift", columns="noise_level", values="auc_multi")
    pivot_multi = pivot_multi.sort_index(ascending=False)

    pivot_single: pandas.DataFrame = df_gauss.pivot_table(
        index="mean_shift", columns="noise_level", values="auc_single")
    pivot_single = pivot_single.sort_index(ascending=False)

    vmin: float = min(df_gauss["auc_multi"].min(), df_gauss["auc_single"].min())
    vmax: float = max(df_gauss["auc_multi"].max(), df_gauss["auc_single"].max())

    clean_multi: float = pivot_multi.loc[0.0, 0.0] if 0.0 in pivot_multi.index and 0.0 in pivot_multi.columns else float("nan")
    clean_single: float = pivot_single.loc[0.0, 0.0] if 0.0 in pivot_single.index and 0.0 in pivot_single.columns else float("nan")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(pivot_multi, cmap="viridis", vmin=vmin, vmax=vmax, annot=False,
                cbar_kws={"label": "Test ROC-AUC"}, ax=axes[0])
    axes[0].set_title(f"Multi-Objective (SiCo-MOGA) Robustness\n"
                      f"Clean AUC = {clean_multi:.4f} | AUC range [{vmin:.4f}, {vmax:.4f}]",
                      fontweight="bold", pad=15)
    axes[0].set_xlabel("Gaussian Noise Level (fraction of training std)", fontweight="bold")
    axes[0].set_ylabel("Systematic Mean Shift (fraction of training std)", fontweight="bold")

    sns.heatmap(pivot_single, cmap="viridis", vmin=vmin, vmax=vmax, annot=False,
                cbar_kws={"label": "Test ROC-AUC"}, ax=axes[1])
    axes[1].set_title(f"Single-Objective (AUC-only GA) Robustness\n"
                      f"Clean AUC = {clean_single:.4f} | AUC range [{vmin:.4f}, {vmax:.4f}]",
                      fontweight="bold", pad=15)
    axes[1].set_xlabel("Gaussian Noise Level (fraction of training std)", fontweight="bold")
    axes[1].set_ylabel("Systematic Mean Shift (fraction of training std)", fontweight="bold")

    fig.suptitle("Out-of-Distribution Robustness: Gaussian Noise + Covariate Shift on Continuous Features",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dummy_flip_comparison(results_df: pandas.DataFrame, filepath: str) -> None:
    """Line plot: multi vs single AUC under increasing bit-flip rate."""
    ensure_directory(os.path.dirname(filepath))

    df_flip: pandas.DataFrame = results_df[results_df["noise_type"] == "dummy_flip"].sort_values("flip_rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_flip["flip_rate"], df_flip["auc_multi"],
            label="Multi-Objective (SiCo-MOGA)", linewidth=2, marker="o", color="tab:blue")
    ax.plot(df_flip["flip_rate"], df_flip["auc_single"],
            label="Single-Objective (AUC-only GA)", linewidth=2, marker="s", linestyle="--", color="tab:orange")
    ax.set_xlabel("Bit-Flip Probability (fraction of dummy features flipped)", fontweight="bold")
    ax.set_ylabel("Test ROC-AUC", fontweight="bold")
    ax.set_title("Robustness to Random Bit-Flip Noise on Binary (Dummy) Features",
                 fontweight="bold", pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    if not df_flip.empty:
        ax.annotate(f"Clean AUC: Multi={df_flip['auc_multi'].iloc[0]:.4f}, "
                    f"Single={df_flip['auc_single'].iloc[0]:.4f}",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9,
                    fontstyle="italic", color="gray")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_gaussian_noise_comparison(results_df: pandas.DataFrame, filepath: str) -> None:
    """Line plot: multi vs single AUC under Gaussian noise (zero mean-shift)."""
    ensure_directory(os.path.dirname(filepath))

    df_gauss: pandas.DataFrame = results_df[
        (results_df["noise_type"] == "gaussian") &
        (numpy.isclose(results_df["mean_shift"], 0.0))
    ].sort_values("noise_level")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_gauss["noise_level"], df_gauss["auc_multi"],
            label="Multi-Objective (SiCo-MOGA)", linewidth=2, marker="o", color="tab:blue")
    ax.plot(df_gauss["noise_level"], df_gauss["auc_single"],
            label="Single-Objective (AUC-only GA)", linewidth=2, marker="s", linestyle="--", color="tab:orange")
    ax.set_xlabel("Gaussian Noise Level (fraction of training std, no mean shift)",
                  fontweight="bold")
    ax.set_ylabel("Test ROC-AUC", fontweight="bold")
    ax.set_title("Robustness to Additive Gaussian Noise on Continuous Features\n"
                 "(zero systematic mean shift)",
                 fontweight="bold", pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    if not df_gauss.empty:
        ax.annotate(f"Clean AUC: Multi={df_gauss['auc_multi'].iloc[0]:.4f}, "
                    f"Single={df_gauss['auc_single'].iloc[0]:.4f}",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9,
                    fontstyle="italic", color="gray")
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
    clean: pandas.DataFrame = results_df[
        (results_df["noise_type"] == "gaussian") &
        (numpy.isclose(results_df["noise_level"], 0.0)) &
        (numpy.isclose(results_df["mean_shift"], 0.0))
    ]
    if not clean.empty:
        row: pandas.Series = clean.iloc[0]
        print(f"  Seed {seed} | Multi features: {len(model_pkg_multi['features']):3d} | "
              f"Single features: {len(model_pkg_single['features']):3d} | "
              f"Clean AUC  Multi: {row['auc_multi']:.4f}  Single: {row['auc_single']:.4f}")

    return results_df


# ---------------------------------------------------------------------------
# Feature selection stability (Jaccard similarity)
# ---------------------------------------------------------------------------

def _individual_to_feature_set(individual: list[int], feature_names: list[str]) -> set[str]:
    """Convert a binary individual to a set of selected feature names."""
    return {f for f, bit in zip(feature_names, individual) if bit == 1}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity: |A intersect B| / |A union B|.  Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_stability_matrix(
        individuals_by_seed: dict[int, list[int]],
        feature_names: list[str]) -> tuple[list[int], numpy.ndarray]:
    """Compute pairwise Jaccard similarity matrix across seeds.
    Returns (seed_list, similarity_matrix)."""
    seeds: list[int] = sorted(individuals_by_seed.keys())
    n: int = len(seeds)
    feature_sets: list[set[str]] = [
        _individual_to_feature_set(individuals_by_seed[s], feature_names)
        for s in seeds
    ]

    matrix: numpy.ndarray = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = jaccard_similarity(feature_sets[i], feature_sets[j])

    return seeds, matrix


def plot_stability_heatmap(
        seeds_multi: list[int],
        matrix_multi: numpy.ndarray,
        seeds_single: list[int],
        matrix_single: numpy.ndarray,
        filepath: str) -> None:
    """Side-by-side Jaccard similarity heatmaps for multi and single objective."""
    ensure_directory(os.path.dirname(filepath))

    mean_multi: float = float(matrix_multi[numpy.triu_indices_from(matrix_multi, k=1)].mean()) \
        if len(seeds_multi) > 1 else 1.0
    mean_single: float = float(matrix_single[numpy.triu_indices_from(matrix_single, k=1)].mean()) \
        if len(seeds_single) > 1 else 1.0

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(matrix_multi, annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0.0, vmax=1.0, square=True, ax=axes[0],
                xticklabels=[f"Seed {s}" for s in seeds_multi],
                yticklabels=[f"Seed {s}" for s in seeds_multi],
                cbar_kws={"label": "Jaccard Similarity"})
    axes[0].set_title(f"Multi-Objective (SiCo-MOGA)\n"
                      f"Mean pairwise Jaccard = {mean_multi:.3f}",
                      fontweight="bold", pad=15)
    axes[0].set_xlabel("Seed", fontweight="bold")
    axes[0].set_ylabel("Seed", fontweight="bold")

    sns.heatmap(matrix_single, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0.0, vmax=1.0, square=True, ax=axes[1],
                xticklabels=[f"Seed {s}" for s in seeds_single],
                yticklabels=[f"Seed {s}" for s in seeds_single],
                cbar_kws={"label": "Jaccard Similarity"})
    axes[1].set_title(f"Single-Objective (AUC-only GA)\n"
                      f"Mean pairwise Jaccard = {mean_single:.3f}",
                      fontweight="bold", pad=15)
    axes[1].set_xlabel("Seed", fontweight="bold")
    axes[1].set_ylabel("Seed", fontweight="bold")

    fig.suptitle("Feature Selection Stability Across Seeds (Jaccard Similarity)",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_frequency(
        individuals_by_seed: dict[int, list[int]],
        feature_names: list[str],
        label: str,
        filepath: str) -> None:
    """Bar chart showing how often each feature is selected across seeds."""
    ensure_directory(os.path.dirname(filepath))

    n_seeds: int = len(individuals_by_seed)
    counts: numpy.ndarray = numpy.zeros(len(feature_names))
    for ind in individuals_by_seed.values():
        counts += numpy.array(ind)

    order: numpy.ndarray = numpy.argsort(-counts)
    # Keep only features selected at least once
    mask: numpy.ndarray = counts[order] > 0
    order: numpy.ndarray = order[mask]

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.4), 6))
    x_pos: numpy.ndarray = numpy.arange(len(order))
    bars = ax.bar(x_pos, counts[order], color="steelblue", edgecolor="black", alpha=0.8)

    # Highlight features selected by all seeds
    for i, idx in enumerate(order):
        if counts[idx] == n_seeds:
            bars[i].set_color("darkgreen")
            bars[i].set_alpha(1.0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([feature_names[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Feature Name", fontweight="bold")
    ax.set_ylabel(f"Selection Count (out of {n_seeds} seeds)", fontweight="bold")
    ax.set_title(f"Feature Selection Frequency - {label}\n"
                 f"(green = selected by all {n_seeds} seeds, "
                 f"blue = selected by some seeds)",
                 fontweight="bold", pad=10)
    ax.set_ylim(0, n_seeds + 0.5)
    ax.axhline(y=n_seeds, color="darkgreen", linestyle="--", alpha=0.5, label=f"All {n_seeds} seeds")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
