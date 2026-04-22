from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import copy
import math
import random


Cell = Optional[int]
Position = Tuple[int, int]
Action = Tuple[int, int, int]  # (row, col, value)


@dataclass
class Move:
    player_id: int
    row: int
    col: int
    value: int


@dataclass
class GameConfig:
    size: int = 4
    init_fill_ratio: float = 0.10
    number_pool: Sequence[int] = (-4, -3, -2, -1, 1, 2, 3, 4)
    singularity_check_min_filled: int = 8
    singularity_tol: float = 1e-12
    det_weight: float = 1.0
    cond_weight: float = 0.30
    singular_bonus: float = 15.0
    full_board_bonus: float = 3.0
    row_zero_penalty: float = 8.0
    invalid_move_penalty: float = 3.0
    undo_tokens_per_player: int = 1


@dataclass
class GameState:
    board: List[List[Cell]]
    current_player: int = 0
    rewards: List[float] = field(default_factory=lambda: [0.0, 0.0])
    undo_tokens: List[int] = field(default_factory=lambda: [1, 1])
    done: bool = False
    winner: Optional[int] = None
    history: List[Move] = field(default_factory=list)
    det_abs: float = 0.0
    condition_number: float = 1.0

    def clone(self) -> "GameState":
        return copy.deepcopy(self)


class MatrixGame:
    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        self.config = config or GameConfig()
        self.random = random.Random(seed)

    def initial_state(self) -> GameState:
        size = self.config.size
        board: List[List[Cell]] = [[None for _ in range(size)] for _ in range(size)]
        filled = max(1, int(size * size * self.config.init_fill_ratio))

        all_positions = [(r, c) for r in range(size) for c in range(size)]
        self.random.shuffle(all_positions)

        for row, col in all_positions[:filled]:
            board[row][col] = self.random.choice(self.config.number_pool)

        state = GameState(
            board=board,
            undo_tokens=[self.config.undo_tokens_per_player, self.config.undo_tokens_per_player],
        )
        det_abs, cond = self.compute_metrics(board)
        state.det_abs = det_abs
        state.condition_number = cond
        return state

    def available_actions(self, state: GameState) -> List[Action]:
        actions: List[Action] = []
        for row in range(self.config.size):
            for col in range(self.config.size):
                if state.board[row][col] is None:
                    for value in self.config.number_pool:
                        actions.append((row, col, value))
        return actions

    def apply_action(self, state: GameState, action: Action) -> Tuple[GameState, float]:
        if state.done:
            return state, 0.0

        player = state.current_player
        row, col, value = action

        if value == 0 or value not in self.config.number_pool:
            reward = -self.config.invalid_move_penalty
            state.rewards[player] += reward
            return self._advance_turn(state), reward

        if not (0 <= row < self.config.size and 0 <= col < self.config.size):
            reward = -self.config.invalid_move_penalty
            state.rewards[player] += reward
            return self._advance_turn(state), reward

        if state.board[row][col] is not None:
            reward = -self.config.invalid_move_penalty
            state.rewards[player] += reward
            return self._advance_turn(state), reward

        prev_det_abs = state.det_abs
        prev_cond = state.condition_number

        state.board[row][col] = value
        state.history.append(Move(player_id=player, row=row, col=col, value=value))

        det_abs, cond = self.compute_metrics(state.board)
        state.det_abs = det_abs
        state.condition_number = cond

        reward = self._shaped_reward(prev_det_abs, det_abs, prev_cond, cond)

        if self._has_zero_sum_row(state.board):
            reward -= self.config.row_zero_penalty
            # Traceback capability: if a player creates a zero-sum row,
            # their latest move can be reverted once.
            if state.undo_tokens[player] > 0:
                state.undo_tokens[player] -= 1
                self._undo_last_move_internal(state, expected_player=player)
                det_abs, cond = self.compute_metrics(state.board)
                state.det_abs = det_abs
                state.condition_number = cond

        filled = self.filled_count(state.board)
        singular_now = self.is_singular_exact(state.board)

        if singular_now and filled >= self.config.singularity_check_min_filled:
            reward += self.config.singular_bonus
            state.done = True
        elif self.is_full(state.board):
            reward += self.config.full_board_bonus
            state.done = True

        state.rewards[player] += reward

        if state.done:
            state.winner = self._winner_from_rewards(state.rewards)
            return state, reward

        state = self._advance_turn(state)
        return state, reward

    def undo_last_move(self, state: GameState, player_id: int) -> bool:
        if state.undo_tokens[player_id] <= 0:
            return False
        ok = self._undo_last_move_internal(state, expected_player=player_id)
        if ok:
            state.undo_tokens[player_id] -= 1
            det_abs, cond = self.compute_metrics(state.board)
            state.det_abs = det_abs
            state.condition_number = cond
        return ok

    def preview_metrics_after_move(self, state: GameState, action: Action) -> Tuple[float, float, bool]:
        row, col, value = action
        if state.board[row][col] is not None or value == 0:
            return float("inf"), float("inf"), False
        temp = [r[:] for r in state.board]
        temp[row][col] = value
        det_abs, cond = self.compute_metrics(temp)
        row_zero = self._has_zero_sum_row(temp)
        return det_abs, cond, row_zero

    def is_full(self, board: List[List[Cell]]) -> bool:
        return all(cell is not None for row in board for cell in row)

    def filled_count(self, board: List[List[Cell]]) -> int:
        return sum(1 for row in board for cell in row if cell is not None)

    def is_singular_exact(self, board: List[List[Cell]]) -> bool:
        matrix = self._to_numeric(board)
        det = determinant(matrix)
        return abs(det) <= self.config.singularity_tol

    def compute_metrics(self, board: List[List[Cell]]) -> Tuple[float, float]:
        matrix = self._to_numeric(board)
        det = determinant(matrix)
        cond = condition_number_1_norm(matrix)
        return abs(det), cond

    def board_to_string(self, board: List[List[Cell]]) -> str:
        rows = []
        for row in board:
            cells = []
            for v in row:
                cells.append(" . " if v is None else f"{v:>2d}")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def _to_numeric(self, board: List[List[Cell]]) -> List[List[float]]:
        return [[float(v if v is not None else 0) for v in row] for row in board]

    def _winner_from_rewards(self, rewards: List[float]) -> Optional[int]:
        if rewards[0] > rewards[1]:
            return 0
        if rewards[1] > rewards[0]:
            return 1
        return None

    def _advance_turn(self, state: GameState) -> GameState:
        state.current_player = 1 - state.current_player
        return state

    def _shaped_reward(self, prev_det_abs: float, det_abs: float, prev_cond: float, cond: float) -> float:
        det_component = (prev_det_abs - det_abs) * self.config.det_weight
        prev_l = math.log1p(prev_cond if math.isfinite(prev_cond) else 1e6)
        curr_l = math.log1p(cond if math.isfinite(cond) else 1e6)
        cond_component = (curr_l - prev_l) * self.config.cond_weight
        return det_component + cond_component

    def _has_zero_sum_row(self, board: List[List[Cell]]) -> bool:
        for row in board:
            # Only consider fully-filled rows to avoid placeholder artifacts.
            if all(v is not None for v in row):
                if sum(row) == 0:
                    return True
        return False

    def _undo_last_move_internal(self, state: GameState, expected_player: int) -> bool:
        for idx in range(len(state.history) - 1, -1, -1):
            move = state.history[idx]
            if move.player_id == expected_player:
                if state.board[move.row][move.col] == move.value:
                    state.board[move.row][move.col] = None
                    del state.history[idx]
                    state.current_player = expected_player
                    state.done = False
                    state.winner = None
                    return True
        return False


def determinant(matrix: List[List[float]]) -> float:
    n = len(matrix)
    a = [row[:] for row in matrix]
    sign = 1.0

    for i in range(n):
        pivot_row = i
        pivot_abs = abs(a[i][i])
        for r in range(i + 1, n):
            if abs(a[r][i]) > pivot_abs:
                pivot_abs = abs(a[r][i])
                pivot_row = r

        if pivot_abs < 1e-15:
            return 0.0

        if pivot_row != i:
            a[i], a[pivot_row] = a[pivot_row], a[i]
            sign *= -1.0

        pivot = a[i][i]
        for r in range(i + 1, n):
            factor = a[r][i] / pivot
            if factor == 0.0:
                continue
            for c in range(i, n):
                a[r][c] -= factor * a[i][c]

    det = sign
    for i in range(n):
        det *= a[i][i]
    return det


def condition_number_1_norm(matrix: List[List[float]]) -> float:
    norm_a = matrix_norm_1(matrix)
    inv = invert_matrix(matrix)
    if inv is None:
        return float("inf")
    norm_inv = matrix_norm_1(inv)
    return norm_a * norm_inv


def matrix_norm_1(matrix: List[List[float]]) -> float:
    n = len(matrix)
    m = len(matrix[0])
    best = 0.0
    for col in range(m):
        s = 0.0
        for row in range(n):
            s += abs(matrix[row][col])
        if s > best:
            best = s
    return best


def invert_matrix(matrix: List[List[float]]) -> Optional[List[List[float]]]:
    n = len(matrix)
    a = [row[:] for row in matrix]
    inv = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        inv[i][i] = 1.0

    for i in range(n):
        pivot_row = i
        pivot_abs = abs(a[i][i])
        for r in range(i + 1, n):
            if abs(a[r][i]) > pivot_abs:
                pivot_abs = abs(a[r][i])
                pivot_row = r

        if pivot_abs < 1e-15:
            return None

        if pivot_row != i:
            a[i], a[pivot_row] = a[pivot_row], a[i]
            inv[i], inv[pivot_row] = inv[pivot_row], inv[i]

        pivot = a[i][i]
        for c in range(n):
            a[i][c] /= pivot
            inv[i][c] /= pivot

        for r in range(n):
            if r == i:
                continue
            factor = a[r][i]
            if factor == 0.0:
                continue
            for c in range(n):
                a[r][c] -= factor * a[i][c]
                inv[r][c] -= factor * inv[i][c]

    return inv
