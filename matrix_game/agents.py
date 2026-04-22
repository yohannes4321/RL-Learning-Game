from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import random

from .core import Action, GameState, MatrixGame
from .q_learning import QTablePolicy


class Agent(Protocol):
    name: str

    def choose_action(self, game: MatrixGame, state: GameState) -> Action:
        ...


@dataclass
class RandomAgent:
    name: str = "random"

    def choose_action(self, game: MatrixGame, state: GameState) -> Action:
        actions = game.available_actions(state)
        if not actions:
            raise RuntimeError("No action available")
        return random.choice(actions)


@dataclass
class HeuristicAgent:
    name: str = "heuristic"

    def choose_action(self, game: MatrixGame, state: GameState) -> Action:
        actions = game.available_actions(state)
        if not actions:
            raise RuntimeError("No action available")

        best = None
        best_score = -float("inf")

        for action in actions:
            det_abs, cond, row_zero = game.preview_metrics_after_move(state, action)
            if row_zero:
                score = -1e9
            else:
                # Prefer smaller |det| and larger condition number.
                cond_term = 0.0 if cond == float("inf") else min(cond, 1e6)
                score = -det_abs + 0.15 * cond_term

            if score > best_score:
                best_score = score
                best = action

        if best is None:
            return random.choice(actions)
        return best


@dataclass
class RLAgent:
    policy: QTablePolicy
    epsilon: float = 0.0
    name: str = "rl"

    def choose_action(self, game: MatrixGame, state: GameState) -> Action:
        if self.epsilon > 0:
            return self.policy.epsilon_greedy_action(state, self.epsilon)
        return self.policy.best_action(state)
