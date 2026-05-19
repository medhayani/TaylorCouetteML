"""Population-Based Training (PBT) — exploit + explore on hyperparameters.

Reference: Jaderberg et al. 2017.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class PBTMember:
    member_id: int
    hyperparams: Dict[str, float]
    score: float = 0.0
    state: Any = None


class PopulationBasedTrainer:
    def __init__(
        self,
        population_size: int = 8,
        exploit_top_frac: float = 0.25,
        explore_perturb_factor: float = 1.2,
        rng_seed: int = 0,
    ):
        self.population_size = population_size
        self.exploit_top_frac = exploit_top_frac
        self.explore_perturb_factor = explore_perturb_factor
        self.rng = random.Random(rng_seed)
        self.members: List[PBTMember] = []

    def init_population(self, hp_sampler: Callable[[], Dict[str, float]]) -> None:
        self.members = [
            PBTMember(member_id=i, hyperparams=hp_sampler())
            for i in range(self.population_size)
        ]

    def step(self) -> None:
        if not self.members:
            return
        ranked = sorted(self.members, key=lambda m: m.score, reverse=True)
        n_top = max(1, int(self.exploit_top_frac * self.population_size))
        top = ranked[:n_top]
        bottom = ranked[-n_top:]
        for loser in bottom:
            winner = self.rng.choice(top)
            loser.hyperparams = self._explore(winner.hyperparams)
            loser.state = copy.deepcopy(winner.state)

    def _explore(self, hp: Dict[str, float]) -> Dict[str, float]:
        new_hp = dict(hp)
        for k, v in new_hp.items():
            if self.rng.random() < 0.5:
                factor = self.explore_perturb_factor
            else:
                factor = 1.0 / self.explore_perturb_factor
            new_hp[k] = float(v) * factor
        return new_hp
