from __future__ import annotations
import os
import random
from typing import Sequence, Any

import numpy
from deap import creator, base, tools, algorithms
from numpy import floating
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from training_config import TrainingConfig
from training_utils import save_stats_csv, plot_single_objective_convergence

class SingleObjectiveTraining:
    def __init__(self,
                 config: TrainingConfig,
                 feature_names: list[str],
                 fold_indices: list[tuple[numpy.ndarray, numpy.ndarray]],
                 X_train_scaled_folds: list[numpy.ndarray],
                 X_val_scaled_folds: list[numpy.ndarray],
                 y_train_folds: list[numpy.ndarray],
                 y_val_folds: list[numpy.ndarray]) -> None:
        # Training configuration
        self._config = config

        # Pre-compute folds and scaling
        self._feature_names: list[str] = feature_names
        self._fold_indices: list[tuple[numpy.ndarray, numpy.ndarray]] = fold_indices
        self._X_train_scaled_folds: list[numpy.ndarray] = X_train_scaled_folds
        self._X_val_scaled_folds: list[numpy.ndarray] = X_val_scaled_folds
        self._y_train_folds: list[numpy.ndarray] = y_train_folds
        self._y_val_folds: list[numpy.ndarray] = y_val_folds

        # Per-instance evaluation cache
        self._cache: dict[tuple[int, ...], tuple[float]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def evaluate_single(self, individual: Sequence[int]) -> tuple[float]:
        key: tuple[int, ...] = tuple(individual)
        if key in self._cache:
            return self._cache[key]
        result = self._evaluate_single(key)
        self._cache[key] = result
        return result

    def _evaluate_single(self, individual: Sequence[int]) -> tuple[float] | tuple[floating[Any]]:
        if sum(individual) == 0:
            return (0.0,)

        cols: numpy.ndarray = numpy.where(numpy.array(individual) == 1)[0]

        auc_scores: list[float] = []

        for fold_idx in range(len(self._fold_indices)):
            X_fold_train_scaled: numpy.ndarray = self._X_train_scaled_folds[fold_idx][:, cols]
            X_fold_val_scaled: numpy.ndarray = self._X_val_scaled_folds[fold_idx][:, cols]
            y_fold_train: numpy.ndarray = self._y_train_folds[fold_idx]
            y_fold_val: numpy.ndarray = self._y_val_folds[fold_idx]

            model: LogisticRegression = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=self._config.seed)
            model.fit(X_fold_train_scaled, y_fold_train)

            probs: numpy.ndarray = model.predict_proba(X_fold_val_scaled)[:, 1]

            if self._config.use_roc_auc:
                fold_auc = roc_auc_score(y_fold_val, probs)
            else:
                fold_auc = average_precision_score(y_fold_val, probs)
            auc_scores.append(fold_auc)

        return (numpy.mean(auc_scores),)

    def run(self) -> creator.Individual:
        if "FitnessSingle" not in creator.__dict__:
            creator.create("FitnessSingle", base.Fitness, weights=(1.0,))

        if "IndividualSingle" not in creator.__dict__:
            creator.create("IndividualSingle", list, fitness=creator.FitnessSingle)

        toolbox: base.Toolbox = base.Toolbox()

        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.IndividualSingle,
            toolbox.attr_bool,
            n=len(self._feature_names),
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate_single)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(self._feature_names))
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop: list[creator.IndividualSingle] = toolbox.population(n=self._config.pop_size)
        hof: tools.HallOfFame = tools.HallOfFame(1)

        stats: tools.Statistics = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", numpy.max)
        stats.register("avg", numpy.mean)
        stats.register("min", numpy.min)
        stats.register("std", numpy.std)

        _, logbook = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=self._config.pop_size,
            lambda_=self._config.pop_size,
            cxpb=self._config.cxpb,
            mutpb=self._config.mutpb,
            ngen=self._config.ngen,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # Save statistics and plots
        if self._config.result_directory:
            gen_stats: list[dict] = [
                {"gen": rec["gen"], "nevals": rec["nevals"],
                 "max": rec["max"], "avg": rec["avg"],
                 "min": rec["min"], "std": rec["std"]}
                for rec in logbook
            ]
            seed_dir: str = os.path.join(self._config.result_directory, f"seed_{self._config.seed}")
            save_stats_csv(gen_stats, os.path.join(seed_dir, "convergence.csv"))
            plot_single_objective_convergence(gen_stats, os.path.join(seed_dir, "convergence.png"))

        best_individual: creator.IndividualSingle = hof[0]
        return best_individual
