import csv
import os

import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_directory(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(auc_vals, sign_vals, c="royalblue", alpha=0.7, edgecolors="black", s=50)
    ax.set_xlabel("AUC")
    ax.set_ylabel("Sign Consistency")
    ax.set_title("Pareto Front (AUC vs Sign Consistency)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
