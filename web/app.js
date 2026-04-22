async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function render(state) {
  const boardEl = document.getElementById("board");
  const statsEl = document.getElementById("stats");
  const statusEl = document.getElementById("status");

  boardEl.innerHTML = "";
  for (const row of state.board) {
    for (const val of row) {
      const d = document.createElement("div");
      d.className = "cell " + (val === "." ? "empty" : "filled");
      d.textContent = val;
      boardEl.appendChild(d);
    }
  }

  const winnerText =
    state.winner === null
      ? "None"
      : state.winner === 0
      ? "RL (P0)"
      : "Heuristic (P1)";

  const formatMetric = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) {
      return "inf";
    }
    return n.toExponential(4);
  };

  statsEl.innerHTML = [
    `<div class="card">Current Player: P${state.current_player}</div>`,
    `<div class="card">det(A) abs: ${formatMetric(state.det_abs)}</div>`,
    `<div class="card">cond(A): ${formatMetric(state.condition_number)}</div>`,
    `<div class="card">Rewards: [${state.rewards.map(v => v.toFixed(2)).join(", ")}]</div>`,
    `<div class="card">Undo tokens: [${state.undo_tokens.join(", ")}]</div>`,
    `<div class="card">Winner: ${winnerText}</div>`,
  ].join("");

  if (state.done) {
    statusEl.textContent = "Game over";
  } else {
    statusEl.textContent = "Game running";
  }
}

async function refresh() {
  const state = await api("/api/state");
  render(state);
}

async function newGame() {
  const state = await api("/api/new", "POST");
  render(state);
}

async function stepGame() {
  const state = await api("/api/step", "POST");
  render(state);
}

async function autoGame() {
  const state = await api("/api/auto", "POST", { max_steps: 100 });
  render(state);
}

document.getElementById("newGame").addEventListener("click", () => {
  newGame().catch((e) => alert(e.message));
});

document.getElementById("step").addEventListener("click", () => {
  stepGame().catch((e) => alert(e.message));
});

document.getElementById("auto").addEventListener("click", () => {
  autoGame().catch((e) => alert(e.message));
});

refresh().catch((e) => alert(e.message));
