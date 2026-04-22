# 4x4 Near-Singularity Matrix Game (RL vs Heuristic)

This project implements a two-player 4x4 matrix game where each move places a non-zero value into an empty cell. The game objective is to make the matrix close to singular (determinant near zero), while avoiding invalid/penalized structure.

It includes:
- A custom game environment with constraints from your task.
- A Q-learning RL agent trained by self-play.
- A non-ML heuristic agent.
- Evaluation and match-play CLI commands.
- A simple browser frontend to watch matches step-by-step.

No external libraries are required (Python standard library only).

## Rules and constraints implemented

1. Board is `4 x 4`.
2. Initialization fills about `10%` of cells (at least one cell); others are placeholders (`None`).
3. Number pool is generated and used for all entries.
4. Zero is forbidden in initialization and gameplay.
5. Metrics for singularity are both:
- `|det(A)|` (primary optimization target, lower is better)
- `cond(A)` in 1-norm (higher tends toward ill-conditioning)
6. Rank-deficiency style penalty:
- If a move creates a fully-filled row with row-sum exactly `0`, it gets penalized.
7. Traceback capability:
- Each player has undo tokens (default `1`), used to revert a harmful move.
8. End conditions:
- Board is full, or
- Matrix becomes singular exactly (`|det(A)| <= tol`) after minimum filled threshold.
9. Winner:
- Player with highest total reward.

## Reward design

At each move, reward is shaped by:

- Determinant improvement: `prev|det| - new|det|` (positive if determinant gets closer to zero)
- Condition-number shift: `log(1+cond_new) - log(1+cond_prev)`
- Penalties/bonuses:
- Invalid move penalty
- Zero-row penalty
- Singularity bonus on terminal singularity
- Small full-board terminal bonus

So the RL agent learns to reduce determinant magnitude while managing penalties.

## Q-Learning summary

Q-learning is an off-policy temporal-difference algorithm that estimates a state-action value function:

`Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`

In this implementation:
- `s` is a compact bucketized game state key (player turn, fill-level, determinant bucket, condition bucket).
- `a` is `(row, col, value)`.
- Exploration uses epsilon-greedy with linear decay.
- Training is self-play: the same learner controls both players during episodes.

### Elegant method used

Because full board states are large, we use a compact **feature-bucket state encoding** instead of storing exact board tensors in the Q-table. This keeps memory small and learning stable while still reflecting near-singularity dynamics via determinant/condition buckets.

## Project structure

- `main.py`: CLI entry point (`train`, `eval`, `play`)
- `matrix_game/core.py`: game logic, matrix metrics, reward shaping, undo traceback
- `matrix_game/q_learning.py`: Q-table policy and Q-learning trainer
- `matrix_game/agents.py`: heuristic, RL, random agents

## Requirements

- Python 3.9+ recommended

## How to run

From project root:

```bash
python main.py train --episodes 5000 --model-path models/q_table.json
```

Evaluate RL vs heuristic:

```bash
python main.py eval --games 200 --model-path models/q_table.json
```

Play one verbose game:

```bash
python main.py play --model-path models/q_table.json
```

## Simple frontend (browser view)

Run the web app:

```bash
python web_app.py --model-path models/q_table.json
```

Then open:

```text
http://127.0.0.1:8000
```

Buttons in the UI:
- `New Game`: reset the board.
- `Step`: play exactly one move (RL then heuristic by turns).
- `Auto Play`: finish the game automatically.

The UI shows:
- Current board and turn.
- `|det(A)|` and condition number.
- Player rewards and winner.

If you changed server code and still see old behavior, stop old server processes and restart `web_app.py`, then hard refresh the browser.

## Useful options

- `--pool -4 -3 -2 -1 1 2 3 4`
- `--seed 7`
- `--singularity-min-filled 8`

Example:

```bash
python main.py train --episodes 8000 --pool -6 -4 -2 2 4 6 --model-path models/q_table_alt.json
```

## Notes

- Determinant uses Gaussian-elimination based calculation.
- Condition number is computed as `||A||_1 * ||A^{-1}||_1` using Gauss-Jordan inversion.
- Empty cells are treated as `0` for matrix metric computation during intermediate states.
- Exact singularity is controlled by tolerance in `GameConfig.singularity_tol`.

## What to tune next

- Increase training episodes for stronger policy.
- Adjust reward weights (`det_weight`, `cond_weight`, penalties).
- Improve heuristic with deeper lookahead/backtracking beyond one-step preview.
# RL-Learning-Game
