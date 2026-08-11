from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .stacking import _validate_probabilities, soft_vote


FUSION_OPERATORS = ("arithmetic", "geometric", "confidence")


@dataclass(slots=True)
class EvolutionConfig:
    generations: int = 30
    population_size: int = 36
    elite_count: int = 4
    tournament_size: int = 3
    meta_validation_fraction: float = 0.20
    correction_reward: float = 0.15
    harm_penalty: float = 0.35
    mutation_scale: float = 0.08

    def validate(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 1 and population_size - 1")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be between 1 and population_size")
        if not 0.05 <= self.meta_validation_fraction <= 0.50:
            raise ValueError("meta_validation_fraction must be between 0.05 and 0.50")
        if self.correction_reward < 0.0 or self.harm_penalty < 0.0:
            raise ValueError("correction_reward and harm_penalty must be non-negative")
        if self.mutation_scale <= 0.0:
            raise ValueError("mutation_scale must be positive")


@dataclass(slots=True)
class FusionGenome:
    operator: str
    weights: np.ndarray
    temperatures: np.ndarray
    gate_threshold: float
    require_disagreement: bool

    def clone(self) -> FusionGenome:
        return FusionGenome(
            operator=self.operator,
            weights=self.weights.copy(),
            temperatures=self.temperatures.copy(),
            gate_threshold=float(self.gate_threshold),
            require_disagreement=bool(self.require_disagreement),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "weights": self.weights.tolist(),
            "temperatures": self.temperatures.tolist(),
            "gate_threshold": float(self.gate_threshold),
            "require_disagreement": bool(self.require_disagreement),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FusionGenome:
        return cls(
            operator=str(values["operator"]),
            weights=np.asarray(values["weights"], dtype=np.float64),
            temperatures=np.asarray(values["temperatures"], dtype=np.float64),
            gate_threshold=float(values["gate_threshold"]),
            require_disagreement=bool(values["require_disagreement"]),
        )


@dataclass(slots=True)
class FusionScore:
    fitness: float
    accuracy: float
    unique_correction_rate: float
    harm_rate: float
    override_rate: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class EvolutionResult:
    genome: FusionGenome
    search_score: FusionScore
    validation_score: FusionScore
    history: list[dict[str, Any]]
    search_indices: np.ndarray
    validation_indices: np.ndarray


def stratified_meta_split(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    indices = np.arange(len(labels))
    search, validation = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return np.sort(search), np.sort(validation)


def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)
    return values / values.sum()


def _temperature_scale(
    probabilities: np.ndarray, temperatures: np.ndarray
) -> np.ndarray:
    values = _validate_probabilities(probabilities).astype(np.float64)
    temperatures = np.clip(np.asarray(temperatures, dtype=np.float64), 0.45, 2.50)
    if temperatures.shape != (values.shape[0],):
        raise ValueError("temperatures must contain one value per model")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temperatures[:, None, None]
    logits -= logits.max(axis=2, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=2, keepdims=True)


def fusion_probabilities(
    probabilities: np.ndarray, genome: FusionGenome
) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    if genome.operator not in FUSION_OPERATORS:
        raise ValueError(f"Unknown fusion operator: {genome.operator}")
    if genome.weights.shape != (values.shape[0],):
        raise ValueError("weights must contain one value per model")
    weights = _normalise_weights(genome.weights)
    scaled = _temperature_scale(values, genome.temperatures)

    if genome.operator == "arithmetic":
        fused = np.tensordot(weights, scaled, axes=(0, 0))
    elif genome.operator == "geometric":
        log_fused = np.tensordot(
            weights, np.log(np.clip(scaled, 1e-12, 1.0)), axes=(0, 0)
        )
        log_fused -= log_fused.max(axis=1, keepdims=True)
        fused = np.exp(log_fused)
    else:
        adjusted_confidence = scaled.max(axis=2) * weights[:, None]
        selected_models = adjusted_confidence.argmax(axis=0)
        sample_indices = np.arange(values.shape[1])
        fused = scaled[selected_models, sample_indices]

    return (fused / fused.sum(axis=1, keepdims=True)).astype(np.float32)


def predict_genome(
    probabilities: np.ndarray, genome: FusionGenome
) -> tuple[np.ndarray, np.ndarray]:
    values = _validate_probabilities(probabilities)
    fallback = soft_vote(values)
    fused = fusion_probabilities(values, genome)
    fused_predictions = fused.argmax(axis=1)
    sample_indices = np.arange(values.shape[1])
    advantage = fused[sample_indices, fused_predictions] - fused[sample_indices, fallback]
    override = (fused_predictions != fallback) & (advantage >= genome.gate_threshold)
    if genome.require_disagreement:
        base_predictions = values.argmax(axis=2)
        disagreement = np.any(base_predictions != base_predictions[0], axis=0)
        override &= disagreement
    predictions = fallback.copy()
    predictions[override] = fused_predictions[override]
    return predictions.astype(np.int64), override


def score_genome(
    probabilities: np.ndarray,
    labels: np.ndarray,
    genome: FusionGenome,
    config: EvolutionConfig,
) -> FusionScore:
    predictions, override = predict_genome(probabilities, genome)
    fallback = soft_vote(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    candidate_correct = predictions == labels
    fallback_correct = fallback == labels
    unique = float(np.mean(candidate_correct & ~fallback_correct))
    harm = float(np.mean(~candidate_correct & fallback_correct))
    accuracy = float(np.mean(candidate_correct))
    fitness = accuracy + config.correction_reward * unique - config.harm_penalty * harm
    return FusionScore(
        fitness=fitness,
        accuracy=accuracy,
        unique_correction_rate=unique,
        harm_rate=harm,
        override_rate=float(np.mean(override)),
    )


def _initial_population(
    model_count: int,
    validation_accuracies: np.ndarray,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> list[FusionGenome]:
    equal = np.full(model_count, 1.0 / model_count)
    validation_weights = _normalise_weights(validation_accuracies)
    population = [
        FusionGenome("arithmetic", equal, np.ones(model_count), 0.0, False),
        FusionGenome("arithmetic", validation_weights, np.ones(model_count), 0.0, False),
        FusionGenome("geometric", equal, np.ones(model_count), 0.0, True),
        FusionGenome("geometric", validation_weights, np.ones(model_count), 0.03, True),
        FusionGenome("confidence", equal, np.ones(model_count), 0.05, True),
    ]
    while len(population) < config.population_size:
        population.append(
            FusionGenome(
                operator=str(rng.choice(FUSION_OPERATORS)),
                weights=rng.dirichlet(np.ones(model_count)),
                temperatures=rng.uniform(0.70, 1.40, size=model_count),
                gate_threshold=float(rng.uniform(0.0, 0.25)),
                require_disagreement=bool(rng.random() < 0.80),
            )
        )
    return population[: config.population_size]


def _mutate(
    first: FusionGenome,
    second: FusionGenome,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> FusionGenome:
    mix = float(rng.uniform(0.25, 0.75))
    weights = mix * first.weights + (1.0 - mix) * second.weights
    weights += rng.normal(0.0, config.mutation_scale, size=weights.shape)
    temperatures = mix * first.temperatures + (1.0 - mix) * second.temperatures
    temperatures += rng.normal(0.0, config.mutation_scale, size=temperatures.shape)
    threshold = mix * first.gate_threshold + (1.0 - mix) * second.gate_threshold
    threshold += float(rng.normal(0.0, config.mutation_scale / 3.0))
    operator = first.operator if rng.random() < 0.5 else second.operator
    if rng.random() < 0.15:
        operator = str(rng.choice(FUSION_OPERATORS))
    require_disagreement = (
        first.require_disagreement if rng.random() < 0.5 else second.require_disagreement
    )
    if rng.random() < 0.10:
        require_disagreement = not require_disagreement
    return FusionGenome(
        operator=operator,
        weights=_normalise_weights(weights),
        temperatures=np.clip(temperatures, 0.50, 2.00),
        gate_threshold=float(np.clip(threshold, 0.0, 0.50)),
        require_disagreement=require_disagreement,
    )


def _score_key(item: tuple[FusionGenome, FusionScore]) -> tuple[float, float, float]:
    _, score = item
    return score.fitness, score.accuracy, -score.harm_rate


def evolve_fusion(
    meta_probabilities: np.ndarray,
    meta_labels: np.ndarray,
    validation_accuracies: np.ndarray,
    config: EvolutionConfig,
    seed: int,
) -> EvolutionResult:
    config.validate()
    values = _validate_probabilities(meta_probabilities)
    labels = np.asarray(meta_labels, dtype=np.int64)
    if values.shape[1] != len(labels):
        raise ValueError("Meta probabilities and labels have different sample counts")
    validation_accuracies = np.asarray(validation_accuracies, dtype=np.float64)
    if validation_accuracies.shape != (values.shape[0],):
        raise ValueError("validation_accuracies must contain one value per model")

    search_indices, validation_indices = stratified_meta_split(
        labels, config.meta_validation_fraction, seed + 7000
    )
    search_probabilities = values[:, search_indices]
    search_labels = labels[search_indices]
    validation_probabilities = values[:, validation_indices]
    validation_labels = labels[validation_indices]
    rng = np.random.default_rng(seed + 8000)
    population = _initial_population(
        values.shape[0], validation_accuracies, config, rng
    )
    # Keep the fixed baselines in the final validation pool even if evolution
    # later prefers more aggressive candidates on the search partition.
    candidate_pool: list[tuple[FusionGenome, FusionScore]] = [
        (
            genome.clone(),
            score_genome(search_probabilities, search_labels, genome, config),
        )
        for genome in population[:5]
    ]
    history: list[dict[str, Any]] = []

    for generation in range(1, config.generations + 1):
        scored = [
            (genome, score_genome(search_probabilities, search_labels, genome, config))
            for genome in population
        ]
        scored.sort(key=_score_key, reverse=True)
        elites = [(genome.clone(), score) for genome, score in scored[: config.elite_count]]
        candidate_pool.extend(elites)
        best_genome, best_score = elites[0]
        history.append(
            {
                "generation": generation,
                **best_score.as_dict(),
                **{f"genome_{key}": value for key, value in best_genome.as_dict().items()},
            }
        )

        def tournament() -> FusionGenome:
            positions = rng.choice(
                len(scored), size=config.tournament_size, replace=False
            )
            winner = max((scored[int(index)] for index in positions), key=_score_key)
            return winner[0]

        population = [genome.clone() for genome, _ in elites]
        while len(population) < config.population_size:
            population.append(_mutate(tournament(), tournament(), config, rng))

    validation_candidates: list[tuple[FusionGenome, FusionScore, FusionScore]] = []
    for genome, search_score in candidate_pool:
        validation_score = score_genome(
            validation_probabilities, validation_labels, genome, config
        )
        validation_candidates.append((genome, search_score, validation_score))
    validation_candidates.sort(
        key=lambda item: (
            item[2].accuracy,
            -item[2].harm_rate,
            item[2].unique_correction_rate,
            item[1].fitness,
        ),
        reverse=True,
    )
    selected, search_score, validation_score = validation_candidates[0]
    return EvolutionResult(
        genome=selected.clone(),
        search_score=search_score,
        validation_score=validation_score,
        history=history,
        search_indices=search_indices,
        validation_indices=validation_indices,
    )
