from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import math
import random

from .core import Action, GameState, MatrixGame


StateKey = str
ActionKey = str


@dataclass
class QLearningConfig:
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 3000
    win_bonus: float = 10.0
    draw_bonus: float = 2.0


class QTablePolicy:
    def __init__(self, game: MatrixGame, q_table: Optional[Dict[StateKey, Dict[ActionKey, float]]] = None):
        self.game = game
        self.q_table = q_table or {}

    @staticmethod
    def action_to_key(action: Action) -> ActionKey:
        row, col, value = action
        return f"{row},{col},{value}"

    @staticmethod
    def key_to_action(key: ActionKey) -> Action:
        row_s, col_s, value_s = key.split(",")
        return int(row_s), int(col_s), int(value_s)

    def state_to_key(self, state: GameState) -> StateKey:
        filled = self.game.filled_count(state.board)
        fill_bin = filled // 2

        det_bucket = self._bucket_det(state.det_abs)
        cond_bucket = self._bucket_cond(state.condition_number)

        return f"p{state.current_player}|f{fill_bin}|d{det_bucket}|c{cond_bucket}"

    def best_action(self, state: GameState) -> Action:
        actions = self.game.available_actions(state)
        if not actions:
            raise RuntimeError("No actions available")

        state_key = self.state_to_key(state)
        row = self.q_table.get(state_key, {})

        best_val = -float("inf")
        best_actions: List[Action] = []

        for action in actions:
            q = row.get(self.action_to_key(action), 0.0)
            if q > best_val:
                best_val = q
                best_actions = [action]
            elif q == best_val:
                best_actions.append(action)

        if not best_actions:
            return random.choice(actions)
        return random.choice(best_actions)

    def epsilon_greedy_action(self, state: GameState, epsilon: float) -> Action:
        actions = self.game.available_actions(state)
        if not actions:
            raise RuntimeError("No actions available")
        if random.random() < epsilon:
            return random.choice(actions)
        return self.best_action(state)

    def q_value(self, state_key: StateKey, action: Action) -> float:
        return self.q_table.get(state_key, {}).get(self.action_to_key(action), 0.0)

    def set_q_value(self, state_key: StateKey, action: Action, value: float) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][self.action_to_key(action)] = value

    def max_q(self, state: GameState) -> float:
        actions = self.game.available_actions(state)
        if not actions:
            return 0.0
        state_key = self.state_to_key(state)
        row = self.q_table.get(state_key, {})
        return max((row.get(self.action_to_key(a), 0.0) for a in actions), default=0.0)

    def _bucket_det(self, det_abs: float) -> int:
        # Bucketize by order of magnitude. Smaller determinant => lower bucket.
        if det_abs <= 1e-12:
            return 0
        if det_abs <= 1e-8:
            return 1
        if det_abs <= 1e-5:
            return 2
        if det_abs <= 1e-3:
            return 3
        if det_abs <= 1e-1:
            return 4
        if det_abs <= 1:
            return 5
        if det_abs <= 10:
            return 6
        if det_abs <= 100:
            return 7
        return 8

    def _bucket_cond(self, cond: float) -> int:
        if not math.isfinite(cond):
            return 8
        if cond < 2:
            return 0
        if cond < 5:
            return 1
        if cond < 10:
            return 2
        if cond < 30:
            return 3
        if cond < 100:
            return 4
        if cond < 300:
            return 5
        if cond < 1000:
            return 6
        if cond < 3000:
            return 7
        return 8


class QLearner:
    def __init__(self, game: MatrixGame, cfg: Optional[QLearningConfig] = None):
        self.game = game
        self.cfg = cfg or QLearningConfig()
        self.policy = QTablePolicy(game)

    def train_self_play(self, episodes: int, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None:
            random.seed(seed)

        wins = [0, 0]
        draws = 0

        for ep in range(1, episodes + 1):
            epsilon = self._epsilon(ep)
            state = self.game.initial_state()

            while not state.done:
                player = state.current_player
                s_key = self.policy.state_to_key(state)
                action = self.policy.epsilon_greedy_action(state, epsilon)

                next_state, reward = self.game.apply_action(state, action)
                max_next = 0.0 if next_state.done else self.policy.max_q(next_state)
                old_q = self.policy.q_value(s_key, action)

                new_q = old_q + self.cfg.alpha * (reward + self.cfg.gamma * max_next - old_q)
                self.policy.set_q_value(s_key, action, new_q)

                state = next_state

            # Terminal shaping by winner
            if state.winner is None:
                draws += 1
            else:
                wins[state.winner] += 1
                self._apply_terminal_bonus(state)

        return {
            "episodes": float(episodes),
            "p0_wins": float(wins[0]),
            "p1_wins": float(wins[1]),
            "draws": float(draws),
            "q_states": float(len(self.policy.q_table)),
        }

    def _apply_terminal_bonus(self, terminal_state: GameState) -> None:
        if terminal_state.winner is None:
            return

        winner = terminal_state.winner
        loser = 1 - winner

        winner_bonus = self.cfg.win_bonus
        loser_penalty = -self.cfg.win_bonus

        # Lightly attribute terminal outcome to each player's most recent move.
        for move in reversed(terminal_state.history):
            if move.player_id == winner:
                s = terminal_state.clone()
                s.current_player = winner
                key = self.policy.state_to_key(s)
                a = (move.row, move.col, move.value)
                old = self.policy.q_value(key, a)
                self.policy.set_q_value(key, a, old + self.cfg.alpha * (winner_bonus - old))
                break

        for move in reversed(terminal_state.history):
            if move.player_id == loser:
                s = terminal_state.clone()
                s.current_player = loser
                key = self.policy.state_to_key(s)
                a = (move.row, move.col, move.value)
                old = self.policy.q_value(key, a)
                self.policy.set_q_value(key, a, old + self.cfg.alpha * (loser_penalty - old))
                break

    def _epsilon(self, episode: int) -> float:
        if episode >= self.cfg.epsilon_decay_episodes:
            return self.cfg.epsilon_end
        span = self.cfg.epsilon_start - self.cfg.epsilon_end
        frac = episode / max(1, self.cfg.epsilon_decay_episodes)
        return self.cfg.epsilon_start - span * frac

    def save(self, path: str) -> None:
        payload = {
            "config": {
                "alpha": self.cfg.alpha,
                "gamma": self.cfg.gamma,
                "epsilon_start": self.cfg.epsilon_start,
                "epsilon_end": self.cfg.epsilon_end,
                "epsilon_decay_episodes": self.cfg.epsilon_decay_episodes,
                "win_bonus": self.cfg.win_bonus,
                "draw_bonus": self.cfg.draw_bonus,
            },
            "q_table": self.policy.q_table,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def load(path: str, game: MatrixGame) -> QTablePolicy:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        q_table = payload.get("q_table", {})
        return QTablePolicy(game=game, q_table=q_table)
