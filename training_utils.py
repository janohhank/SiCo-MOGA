import csv
import os

import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_directory(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def _apply_plot_theme() -> None:
    """Set a clean white-background, black-text theme for all plots."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
        "grid.color": "#cccccc",
    })


def save_stats_csv(stats: list[dict], filepath: str) -> None:
    if not stats:
        return
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "w", newline="") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)


def plot_single_objective_convergence(stats: list[dict], filepath: str) -> None:
    ensure_directory(os.path.dirname(filepath))
    gens: list[int] = [s["gen"] for s in stats]
    max_vals: list[float] = [s["max"] for s in stats]
    avg_vals: list[float] = [s["avg"] for s in stats]

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, max_vals, label="Max", linewidth=2)
    ax.plot(gens, avg_vals, label="Avg", linewidth=2, linestyle="--")
    if "min" in stats[0]:
        ax.fill_between(gens,
                         [s["min"] for s in stats],
                         max_vals,
                         alpha=0.15, label="Min-Max range")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (AUC)")
    ax.set_title("Single-Objective GA Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_multi_objective_convergence(stats: list[dict], filepath: str) -> None:
    ensure_directory(os.path.dirname(filepath))
    gens: list[int] = [s["gen"] for s in stats]

    _apply_plot_theme()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # AUC convergence
    axes[0].plot(gens, [s["auc_max"] for s in stats], label="Max", linewidth=2)
    axes[0].plot(gens, [s["auc_mean"] for s in stats], label="Mean", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sign consistency convergence
    axes[1].plot(gens, [s["sign_max"] for s in stats], label="Max", linewidth=2)
    axes[1].plot(gens, [s["sign_mean"] for s in stats], label="Mean", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Sign Consistency")
    axes[1].set_title("Sign Consistency Convergence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Pareto front size
    axes[2].plot(gens, [s["pareto_size"] for s in stats], linewidth=2, color="green")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Pareto Front Size")
    axes[2].set_title("Pareto Front Size")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_pareto_front(pareto_individuals: list, filepath: str) -> None:
    ensure_directory(os.path.dirname(filepath))
    auc_vals: list[float] = [ind.fitness.values[0] for ind in pareto_individuals]
    sign_vals: list[float] = [ind.fitness.values[1] for ind in pareto_individuals]

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(auc_vals, sign_vals, c="royalblue", alpha=0.7, edgecolors="black", s=50)
    ax.set_xlabel("AUC")
    ax.set_ylabel("Sign Consistency")
    ax.set_title("Pareto Front (AUC vs Sign Consistency)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pareto front model selection
# Fitness convention (multi-objective GA):
#   ind.fitness.values[0] = AUC              (higher is better)
#   ind.fitness.values[1] = Sign consistency (higher is better)
# ---------------------------------------------------------------------------

def best_auc_index(pareto_front: list) -> int:
    """Return the index of the Pareto individual with the highest AUC."""
    if not pareto_front:
        raise ValueError("Pareto front is empty")
    return max(range(len(pareto_front)),
               key=lambda i: pareto_front[i].fitness.values[0])


def best_sign_consistency_index(pareto_front: list) -> int:
    """Return the index of the Pareto individual with the highest sign-consistency score."""
    if not pareto_front:
        raise ValueError("Pareto front is empty")
    return max(range(len(pareto_front)),
               key=lambda i: pareto_front[i].fitness.values[1])


def knee_point_index(pareto_front: list) -> int:
    """Return the index of the knee-point individual on the Pareto front.

    The knee point is the Pareto solution with the largest perpendicular
    distance from the straight line connecting the two extreme points
    (best-AUC and best-sign-consistency). Both objectives are min-max
    normalized to [0, 1] first so they are comparable. For a single-point
    front, index 0 is returned; if the two extremes coincide, the best-AUC
    index is returned.
    """
    n: int = len(pareto_front)
    if n == 0:
        raise ValueError("Pareto front is empty")
    if n == 1:
        return 0

    auc: numpy.ndarray = numpy.array(
        [ind.fitness.values[0] for ind in pareto_front], dtype=float)
    sign: numpy.ndarray = numpy.array(
        [ind.fitness.values[1] for ind in pareto_front], dtype=float)

    def _norm(v: numpy.ndarray) -> numpy.ndarray:
        rng: float = float(v.max() - v.min())
        if rng == 0.0:
            return numpy.zeros_like(v)
        return (v - v.min()) / rng

    auc_n: numpy.ndarray = _norm(auc)
    sign_n: numpy.ndarray = _norm(sign)

    p_auc_idx: int = int(numpy.argmax(auc_n))
    p_sign_idx: int = int(numpy.argmax(sign_n))
    if p_auc_idx == p_sign_idx:
        return p_auc_idx

    p1: numpy.ndarray = numpy.array([auc_n[p_auc_idx], sign_n[p_auc_idx]])
    p2: numpy.ndarray = numpy.array([auc_n[p_sign_idx], sign_n[p_sign_idx]])
    line_vec: numpy.ndarray = p2 - p1
    line_len: float = float(numpy.linalg.norm(line_vec))
    if line_len == 0.0:
        return p_auc_idx

    # Perpendicular distance from each (auc_n, sign_n) point to the line p1-p2
    points: numpy.ndarray = numpy.column_stack([auc_n, sign_n])
    rel: numpy.ndarray = points - p1
    cross: numpy.ndarray = rel[:, 0] * line_vec[1] - rel[:, 1] * line_vec[0]
    distances: numpy.ndarray = numpy.abs(cross) / line_len

    return int(numpy.argmax(distances))


def select_pareto_individual(pareto_front: list, use_knee_point: bool = True):
    """Pick one individual from a Pareto front using a consistent strategy.

    use_knee_point=True  -> knee-point (balanced trade-off via knee_point_index).
    use_knee_point=False -> best-sign-consistency (max fitness.values[1]).

    Centralising the choice here guarantees that every call site in the
    notebook (evaluation, stability, all-models comparison, ...) picks the
    *same* individual for a given Pareto front.
    """
    if use_knee_point:
        return pareto_front[knee_point_index(pareto_front)]
    return pareto_front[best_sign_consistency_index(pareto_front)]
