import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# CONFIGURAÇÃO DE PLOT
# =========================

OUTPUT_DIR_NAME = "gridworld_value-iteration"

def setup_matplotlib_for_spyder():
    """
    Para Spyder:
    - não use matplotlib.use('Agg')
    - esta função apenas ajusta estilo e comportamento
    """
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def get_output_dir():
    """
    Cria a pasta de saída abaixo da raiz onde está este script.
    """
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def finalize_figure(fig, filename=None):
    """
    Salva a figura na pasta gridworld_value-iteration.
    """
    plt.tight_layout()

    if filename is not None:
        output_dir = get_output_dir()
        full_path = output_dir / filename
        fig.savefig(full_path, dpi=200, bbox_inches="tight")
        print(f"Figura salva em: {full_path}")

    plt.close(fig)


# =========================
# CONFIGURAÇÃO DO GRIDWORLD
# =========================

N_ROWS = 5
N_COLS = 5

START_STATE = (4, 0)
TERMINAL_STATES = {(0, 4)}
WALLS = {(1, 1), (2, 1), (3, 1)}  # pode deixar vazio: set()

ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ACTION_SYMBOLS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}

GAMMA = 0.95
THETA = 1e-6
MAX_ITERS = 10000

STEP_REWARD = -1.0
TERMINAL_REWARD = 0.0
WALL_REWARD = -100.0
BOUNDARY_REWARD = -100.0


# =========================
# FUNÇÕES DO AMBIENTE
# =========================

def is_inside(state):
    r, c = state
    return 0 <= r < N_ROWS and 0 <= c < N_COLS


def is_terminal(state):
    return state in TERMINAL_STATES


def is_wall(state):
    return state in WALLS


def get_all_states():
    states = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = (r, c)
            if not is_wall(s):
                states.append(s)
    return states


def transition(state, action):
    """
    Ambiente determinístico.
    Se bater em parede/borda, permanece no estado.
    """
    if is_terminal(state):
        return state, TERMINAL_REWARD

    dr, dc = ACTION_DELTAS[action]
    candidate = (state[0] + dr, state[1] + dc)

    if not is_inside(candidate):
        return state, BOUNDARY_REWARD

    if is_wall(candidate):
        return state, WALL_REWARD

    if is_terminal(candidate):
        return candidate, TERMINAL_REWARD

    return candidate, STEP_REWARD


# =========================
# VALUE ITERATION
# =========================

def value_iteration():
    states = get_all_states()
    V = {s: 0.0 for s in states}

    for s in TERMINAL_STATES:
        if s in V:
            V[s] = 0.0

    for iteration in range(MAX_ITERS):
        delta = 0.0
        new_V = V.copy()

        for s in states:
            if is_terminal(s):
                new_V[s] = 0.0
                continue

            q_values = []
            for a in ACTIONS:
                s_next, r = transition(s, a)
                q = r + GAMMA * V[s_next]
                q_values.append(q)

            best_value = max(q_values)
            new_V[s] = best_value
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if delta < THETA:
            print(f"Value Iteration convergiu em {iteration + 1} iterações.")
            break

    return V


def get_q_values(V, state):
    q = {}
    if is_terminal(state):
        for a in ACTIONS:
            q[a] = 0.0
        return q

    for a in ACTIONS:
        s_next, r = transition(state, a)
        q[a] = r + GAMMA * V[s_next]
    return q


def greedy_policy_from_values(V):
    policy = {}
    for s in get_all_states():
        if is_terminal(s):
            policy[s] = []
            continue

        q = get_q_values(V, s)
        max_q = max(q.values())
        best_actions = [a for a, val in q.items() if np.isclose(val, max_q)]
        policy[s] = best_actions
    return policy


# =========================
# RUN / TRAJETÓRIA GREEDY
# =========================

def greedy_run(V, start_state=START_STATE, max_steps=100):
    """
    Executa uma trajetória usando política greedy baseada em V.
    Em empates, escolhe a primeira ação na ordem ACTIONS.
    """
    if is_wall(start_state):
        raise ValueError("O estado inicial não pode ser parede.")

    trajectory = [start_state]
    actions_taken = []
    rewards = []

    current = start_state

    for _ in range(max_steps):
        if is_terminal(current):
            break

        q = get_q_values(V, current)
        max_q = max(q.values())
        best_actions = [a for a in ACTIONS if np.isclose(q[a], max_q)]
        action = best_actions[0]

        next_state, reward = transition(current, action)

        actions_taken.append(action)
        rewards.append(reward)
        trajectory.append(next_state)

        current = next_state

        if is_terminal(current):
            break

    return trajectory, actions_taken, rewards


# =========================
# PLOTS
# =========================

def values_to_grid(V):
    grid = np.full((N_ROWS, N_COLS), np.nan)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = (r, c)
            if is_wall(s):
                grid[r, c] = np.nan
            elif is_terminal(s):
                grid[r, c] = 0.0
            else:
                grid[r, c] = V[s]
    return grid


def plot_value_heatmap(V, title="Valores dos estados"):
    grid = values_to_grid(V)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="viridis")

    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = (r, c)
            if is_wall(s):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, alpha=0.7, color="gray"))
                ax.text(c, r, "W", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
            elif is_terminal(s):
                ax.text(c, r, "T\n0.00", ha="center", va="center", fontsize=11, fontweight="bold")
            else:
                ax.text(c, r, f"{V[s]:.2f}", ha="center", va="center", fontsize=9, color="white")

    ax.set_title(title)
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(N_ROWS - 0.5, -0.5)
    ax.grid(True)
    fig.colorbar(im, ax=ax, shrink=0.8)

    finalize_figure(fig, "heatmap_valores.png")
    return fig, ax


def plot_greedy_policy(V, title="Política greedy derivada de V"):
    policy = greedy_policy_from_values(V)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(N_ROWS - 0.5, -0.5)
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.grid(True)
    ax.set_title(title)

    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = (r, c)

            if is_wall(s):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, alpha=0.7, color="gray"))
                ax.text(c, r, "W", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
            elif is_terminal(s):
                ax.text(c, r, "T", ha="center", va="center", fontsize=13, fontweight="bold")
            else:
                arrows = "".join(ACTION_SYMBOLS[a] for a in policy[s])
                ax.text(c, r, arrows, ha="center", va="center", fontsize=14)

    finalize_figure(fig, "politica_greedy.png")
    return fig, ax


def plot_trajectory(V, trajectory, title="Run com política greedy"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(N_ROWS - 0.5, -0.5)
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.grid(True)
    ax.set_title(title)

    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = (r, c)
            if is_wall(s):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, alpha=0.7, color="gray"))
                ax.text(c, r, "W", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
            elif is_terminal(s):
                ax.text(c, r, "T", ha="center", va="center", fontsize=13, fontweight="bold")
            else:
                ax.text(c, r, f"{V[s]:.1f}", ha="center", va="center", fontsize=9)

    xs = [s[1] for s in trajectory]
    ys = [s[0] for s in trajectory]
    ax.plot(xs, ys, marker="o", linewidth=2)

    sr, sc = trajectory[0]
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold")

    for t, s in enumerate(trajectory):
        ax.text(s[1] + 0.18, s[0] - 0.18, str(t), fontsize=8)

    finalize_figure(fig, "trajetoria_greedy.png")
    return fig, ax


def plot_action_preferences(V, state=START_STATE, title=None):
    q = get_q_values(V, state)
    actions = list(q.keys())
    values = [q[a] for a in actions]

    if title is None:
        title = f"Preferência por ação no estado {state}"

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(actions, values)
    ax.set_title(title)
    ax.set_ylabel("Q(s,a) = r + γV(s')")

    max_val = max(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top"
        )

    for bar, val in zip(bars, values):
        if np.isclose(val, max_val):
            bar.set_linewidth(2)
            bar.set_edgecolor("black")

    finalize_figure(fig, "preferencia_acoes.png")
    return fig, ax


# =========================
# IMPRESSÃO AUXILIAR
# =========================

def print_run_summary(trajectory, actions_taken, rewards):
    print("\n=== RESUMO DA RUN GREEDY ===")
    for t in range(len(actions_taken)):
        print(
            f"Passo {t}: estado={trajectory[t]} | ação={actions_taken[t]} | "
            f"recompensa={rewards[t]} | próximo={trajectory[t+1]}"
        )
    print(f"Estado final: {trajectory[-1]}")
    print(f"Total de passos: {len(actions_taken)}")
    print(f"Retorno acumulado simples: {sum(rewards):.2f}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    setup_matplotlib_for_spyder()

    output_dir = get_output_dir()
    print(f"Pasta de saída: {output_dir}")

    V = value_iteration()
    policy = greedy_policy_from_values(V)

    trajectory, actions_taken, rewards = greedy_run(
        V,
        start_state=START_STATE,
        max_steps=100
    )

    plot_value_heatmap(V, title="Heatmap dos valores V*(s)")
    plot_greedy_policy(V, title="Política greedy extraída dos valores")
    plot_trajectory(V, trajectory, title="Trajetória da run com política greedy")
    plot_action_preferences(V, state=START_STATE, title=f"Preferência por ação em {START_STATE}")

    print_run_summary(trajectory, actions_taken, rewards)