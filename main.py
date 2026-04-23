from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

from matrix_game.agents import HeuristicAgent, RLAgent
from matrix_game.core import GameConfig, MatrixGame
from matrix_game.q_learning import QLearner


def play_match(game: MatrixGame, a0, a1, verbose: bool = False) -> Tuple[int | None, list[float], float, float]:
    state = game.initial_state()

    if verbose:
        print("Initial board:")
        print(game.board_to_string(state.board))
        print(f"det_abs={state.det_abs:.6g}, cond={state.condition_number:.6g}\n")

    agents = [a0, a1]

    while not state.done:
        player = state.current_player
        action = agents[player].choose_action(game, state)
        state, reward = game.apply_action(state, action)

        if verbose:
            r, c, v = action
            print(f"P{player} ({agents[player].name}) -> row={r}, col={c}, value={v}, reward={reward:.4f}")
            print(game.board_to_string(state.board))
            print(f"det_abs={state.det_abs:.6g}, cond={state.condition_number:.6g}")
            print(f"scores={state.rewards}, undo_tokens={state.undo_tokens}\n")

    return state.winner, state.rewards, state.det_abs, state.condition_number


def train_rl_vs_heuristic(learner: QLearner, episodes: int, seed: int | None = None) -> dict[str, float]:
    # Keep RL fixed as P0 and heuristic as P1.
    if seed is not None:
        random.seed(seed)

    heuristic = HeuristicAgent()
    wins = [0, 0]
    draws = 0

    for ep in range(1, episodes + 1):
        epsilon = learner._epsilon(ep)
        state = learner.game.initial_state()

        while not state.done:
            if state.current_player == 0:
                s_key = learner.policy.state_to_key(state)
                action = learner.policy.epsilon_greedy_action(state, epsilon)

                next_state, reward = learner.game.apply_action(state, action)

                if next_state.done:
                    target = reward
                    state = next_state
                else:
                    opp_action = heuristic.choose_action(learner.game, next_state)
                    post_opp_state, _ = learner.game.apply_action(next_state, opp_action)
                    max_next = 0.0 if post_opp_state.done else learner.policy.max_q(post_opp_state)
                    target = reward + learner.cfg.gamma * max_next
                    state = post_opp_state

                old_q = learner.policy.q_value(s_key, action)
                new_q = old_q + learner.cfg.alpha * (target - old_q)
                learner.policy.set_q_value(s_key, action, new_q)
            else:
                # Safety branch in case turn order is altered in future variants.
                opp_action = heuristic.choose_action(learner.game, state)
                state, _ = learner.game.apply_action(state, opp_action)

        if state.winner is None:
            draws += 1
        else:
            wins[state.winner] += 1

    return {
        "episodes": float(episodes),
        "p0_wins": float(wins[0]),
        "p1_wins": float(wins[1]),
        "draws": float(draws),
        "q_states": float(len(learner.policy.q_table)),
    }


def run_train(args):
    cfg = GameConfig(
        size=4,
        init_fill_ratio=0.10,
        number_pool=tuple(args.pool),
        singularity_check_min_filled=args.singularity_min_filled,
    )
    game = MatrixGame(config=cfg, seed=args.seed)
    learner = QLearner(game)

    metrics = train_rl_vs_heuristic(learner, args.episodes, seed=args.seed)

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    learner.save(args.model_path)

    print("Training completed (RL as P0 vs Heuristic as P1)")
    print(f"Episodes: {int(metrics['episodes'])}")
    print(f"P0 wins:  {int(metrics['p0_wins'])}")
    print(f"P1 wins:  {int(metrics['p1_wins'])}")
    print(f"Draws:    {int(metrics['draws'])}")
    print(f"Q states: {int(metrics['q_states'])}")
    print(f"Model saved to: {args.model_path}")


def run_eval(args):
    cfg = GameConfig(
        size=4,
        init_fill_ratio=0.10,
        number_pool=tuple(args.pool),
        singularity_check_min_filled=args.singularity_min_filled,
    )
    game = MatrixGame(config=cfg, seed=args.seed)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    policy = QLearner.load(args.model_path, game)
    rl = RLAgent(policy=policy, epsilon=0.0)
    heuristic = HeuristicAgent()

    rl_wins = 0
    heuristic_wins = 0
    draws = 0

    for _ in range(args.games):
        winner, _, _, _ = play_match(game, rl, heuristic, verbose=False)
        if winner is None:
            draws += 1
        elif winner == 0:
            rl_wins += 1
        else:
            heuristic_wins += 1

    print("Evaluation (RL as P0 vs Heuristic as P1)")
    print(f"Games:           {args.games}")
    print(f"RL wins:         {rl_wins}")
    print(f"Heuristic wins:  {heuristic_wins}")
    print(f"Draws:           {draws}")


def run_play(args):
    cfg = GameConfig(
        size=4,
        init_fill_ratio=0.10,
        number_pool=tuple(args.pool),
        singularity_check_min_filled=args.singularity_min_filled,
    )
    game = MatrixGame(config=cfg, seed=args.seed)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    policy = QLearner.load(args.model_path, game)
    rl = RLAgent(policy=policy, epsilon=0.0)
    heuristic = HeuristicAgent()

    winner, rewards, det_abs, cond = play_match(game, rl, heuristic, verbose=True)

    print("Final result")
    print(f"Winner: {'Draw' if winner is None else ('RL (P0)' if winner == 0 else 'Heuristic (P1)')}")
    print(f"Rewards: {rewards}")
    print(f"Final |det|: {det_abs:.6g}")
    print(f"Final cond: {cond:.6g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4x4 near-singularity matrix game with Q-learning")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-path", default="models/q_table.json", help="Path to save/load Q-table")
    common.add_argument("--seed", type=int, default=7, help="Random seed")
    common.add_argument(
        "--pool",
        nargs="+",
        type=int,
        default=[-4, -3, -2, -1, 1, 2, 3, 4],
        help="Non-zero number pool for entries",
    )
    common.add_argument(
        "--singularity-min-filled",
        type=int,
        default=8,
        help="Only allow exact singular termination after this many cells are filled",
    )

    train_p = sub.add_parser("train", parents=[common], help="Train RL vs heuristic")
    train_p.add_argument("--episodes", type=int, default=5000)

    eval_p = sub.add_parser("eval", parents=[common], help="Evaluate RL vs heuristic")
    eval_p.add_argument("--games", type=int, default=200)

    sub.add_parser("play", parents=[common], help="Play one verbose RL vs heuristic match")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if any(v == 0 for v in args.pool):
        raise ValueError("Zero is not allowed in the number pool.")

    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "eval":
        run_eval(args)
    elif args.cmd == "play":
        run_play(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
