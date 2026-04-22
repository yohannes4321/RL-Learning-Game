from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from matrix_game.agents import HeuristicAgent, RLAgent
from matrix_game.core import GameConfig, MatrixGame
from matrix_game.q_learning import QLearner


ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"


class GameSession:
    def __init__(self, model_path: str, seed: int = 7):
        self.cfg = GameConfig(
            size=4,
            init_fill_ratio=0.10,
            number_pool=(-4, -3, -2, -1, 1, 2, 3, 4),
            singularity_check_min_filled=8,
        )
        self.game = MatrixGame(config=self.cfg, seed=seed)
        self.model_path = model_path
        self.rl = self._load_rl_agent(model_path)
        self.heuristic = HeuristicAgent()
        self.state = self.game.initial_state()

    def _load_rl_agent(self, model_path: str) -> RLAgent:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train first with: "
                "python main.py train --episodes 8000 --model-path models/q_table.json"
            )
        policy = QLearner.load(model_path, self.game)
        return RLAgent(policy=policy, epsilon=0.0)

    def reset(self) -> None:
        self.state = self.game.initial_state()

    def step(self) -> None:
        if self.state.done:
            return
        player = self.state.current_player
        agent = self.rl if player == 0 else self.heuristic
        action = agent.choose_action(self.game, self.state)
        self.state, _ = self.game.apply_action(self.state, action)

    def snapshot(self) -> dict:
        board = [[cell if cell is not None else "." for cell in row] for row in self.state.board]
        return {
            "board": board,
            "current_player": self.state.current_player,
            "done": self.state.done,
            "winner": self.state.winner,
            "rewards": self.state.rewards,
            "det_abs": self.state.det_abs,
            "condition_number": self.state.condition_number,
            "undo_tokens": self.state.undo_tokens,
        }


class App:
    def __init__(self, model_path: str, seed: int):
        self.session = GameSession(model_path=model_path, seed=seed)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, data: dict, status: int = 200) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
            payload = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _serve_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}
            raw = self.rfile.read(content_length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
                return
            if path == "/style.css":
                self._serve_file(WEB_DIR / "style.css", "text/css; charset=utf-8")
                return
            if path == "/app.js":
                self._serve_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
                return
            if path == "/api/state":
                self._json(app.session.snapshot())
                return

            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/new":
                app.session.reset()
                self._json(app.session.snapshot())
                return

            if path == "/api/step":
                app.session.step()
                self._json(app.session.snapshot())
                return

            if path == "/api/auto":
                body = self._read_json_body()
                max_steps = int(body.get("max_steps", 100))
                steps = 0
                while not app.session.state.done and steps < max_steps:
                    app.session.step()
                    steps += 1
                state = app.session.snapshot()
                state["steps_ran"] = steps
                self._json(state)
                return

            self.send_error(404)

        def log_message(self, fmt: str, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple web UI for the matrix game")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", default="models/q_table.json")
    args = parser.parse_args()

    app = App(model_path=args.model_path, seed=args.seed)
    handler = make_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Web UI running at http://{args.host}:{args.port}")
    print("Open this URL in your browser. Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
