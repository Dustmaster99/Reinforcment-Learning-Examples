from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# CONFIGURAÇÃO DE PLOT
# =========================

OUTPUT_DIR_NAME = "policy_iteration"


def setup_matplotlib_for_spyder():
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def get_output_dir():
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def finalize_figure(fig, filename=None):
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
WALLS = {(1, 1), (2, 1), (3, 1)}

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
MAX_POLICY_EVAL_ITERS = 10000
MAX_POLICY_ITER_ITERS = 1000

STEP_REWARD = -1.0
TERMINAL_REWARD = 0.0
WALL_REWARD = -100.0
BOUNDARY_REWARD = -1.0


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




def print_policy_probabilities(policy):
    print("\n=== PROBABILIDADES DA POLÍTICA POR ESTADO ===")

    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = (r, c)

            if is_wall(s):
                print(f"Estado {s}: WALL")
                continue

            if is_terminal(s):
                print(f"Estado {s}: TERMINAL")
                continue

            probs = policy[s]
            print(
                f"Estado {s}: "
                f"up={probs['up']:.2f}, "
                f"down={probs['down']:.2f}, "
                f"left={probs['left']:.2f}, "
                f"right={probs['right']:.2f}"
            )


# =========================
# POLÍTICA INICIAL EQUIPROVÁVEL
# =========================

def create_equiprobable_policy():
    policy = {}
    prob = 1.0 / len(ACTIONS)

    for s in get_all_states():
        if is_terminal(s):
            policy[s] = {a: 0.0 for a in ACTIONS}
        else:
            policy[s] = {a: prob for a in ACTIONS}

    return policy


def get_action_values(V, state):
    q = {}

    if is_terminal(state):
        for a in ACTIONS:
            q[a] = 0.0
        return q

    for a in ACTIONS:
        next_state, reward = transition(state, a)
        q[a] = reward + GAMMA * V[next_state]

    return q


def greedy_actions_from_values(V, state):
    q = get_action_values(V, state)
    max_q = max(q.values())
    best_actions = [a for a, val in q.items() if np.isclose(val, max_q)]
    return best_actions, q


# =========================
# POLICY EVALUATION
# =========================

def policy_evaluation(policy, V=None):
    states = get_all_states()

    if V is None:
        V = {s: 0.0 for s in states}

    for s in TERMINAL_STATES:
        if s in V:
            V[s] = 0.0

    for iteration in range(MAX_POLICY_EVAL_ITERS):
        delta = 0.0
        new_V = V.copy()

        for s in states:
            if is_terminal(s):
                new_V[s] = 0.0
                continue

            value = 0.0
            for a, prob in policy[s].items():
                next_state, reward = transition(s, a)
                value += prob * (reward + GAMMA * V[next_state])

            new_V[s] = value
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if delta < THETA:
            return V, iteration + 1

    return V, MAX_POLICY_EVAL_ITERS


# =========================
# POLICY IMPROVEMENT
# =========================

def policy_improvement(V, policy):
    policy_stable = True
    new_policy = {}

    for s in get_all_states():
        if is_terminal(s):
            new_policy[s] = {a: 0.0 for a in ACTIONS}
            continue

        old_policy = policy[s]
        best_actions, _ = greedy_actions_from_values(V, s)

        prob = 1.0 / len(best_actions)
        greedy_policy = {
            a: (prob if a in best_actions else 0.0)
            for a in ACTIONS
        }

        new_policy[s] = greedy_policy

        if greedy_policy != old_policy:
            policy_stable = False

    return new_policy, policy_stable


# =========================
# POLICY ITERATION
# =========================

def policy_iteration():
    policy = create_equiprobable_policy()

    V = {s: 0.0 for s in get_all_states()}
    for s in TERMINAL_STATES:
        if s in V:
            V[s] = 0.0

    V_initial = V.copy()
    policy_eval_counts = []

    for iteration in range(MAX_POLICY_ITER_ITERS):
        print(f"\n=== Policy Iteration {iteration + 1} ===")

        V, eval_iters = policy_evaluation(policy, V=V)
        policy_eval_counts.append(eval_iters)
        print(f"Policy evaluation convergiu em {eval_iters} iterações internas.")

        policy, policy_stable = policy_improvement(V, policy)
        print(f"Policy stable? {policy_stable}")

        if policy_stable:
            print("\nPolicy Iteration convergiu: política ótima encontrada.")
            return V_initial, V, policy, iteration + 1, policy_eval_counts

    print("\nAtingido número máximo de iterações de Policy Iteration.")
    return V_initial, V, policy, MAX_POLICY_ITER_ITERS, policy_eval_counts


# =========================
# RUN COM POLÍTICA FINAL
# =========================

def sample_action_from_policy(policy, state):
    probs = [policy[state][a] for a in ACTIONS]
    return np.random.choice(ACTIONS, p=probs)


def policy_run(policy, start_state=START_STATE, max_steps=100, seed=42):
    rng_state = np.random.get_state()
    np.random.seed(seed)

    if is_wall(start_state):
        raise ValueError("O estado inicial não pode ser parede.")

    trajectory = [start_state]
    actions_taken = []
    rewards = []

    current = start_state

    for _ in range(max_steps):
        if is_terminal(current):
            break

        action = sample_action_from_policy(policy, current)
        next_state, reward = transition(current, action)

        actions_taken.append(action)
        rewards.append(reward)
        trajectory.append(next_state)

        current = next_state

        if is_terminal(current):
            break

    np.random.set_state(rng_state)
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


def plot_value_heatmap(V, title="Heatmap dos valores finais"):
    grid = values_to_grid(V)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="viridis")

    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = (r, c)

            if is_wall(s):
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        fill=True,
                        alpha=0.7,
                        color="gray",
                    )
                )
                ax.text(
                    c,
                    r,
                    "W\n-100",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )
            elif is_terminal(s):
                ax.text(
                    c,
                    r,
                    "T\n0.00",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
            else:
                ax.text(
                    c,
                    r,
                    f"{V[s]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )

    ax.set_title(title)
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(N_ROWS - 0.5, -0.5)
    ax.grid(True)

    fig.colorbar(im, ax=ax, shrink=0.8)

    finalize_figure(fig, "heatmap_valores_policy_iteration.png")
    return fig, ax


def plot_policy(policy, title="Política final do Policy Iteration"):
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
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        fill=True,
                        alpha=0.7,
                        color="gray",
                    )
                )
                ax.text(
                    c,
                    r,
                    "W",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                )
            elif is_terminal(s):
                ax.text(
                    c,
                    r,
                    "T",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                )
            else:
                active_actions = [a for a, p in policy[s].items() if p > 0]
                arrows = "".join(ACTION_SYMBOLS[a] for a in active_actions)
                ax.text(c, r, arrows, ha="center", va="center", fontsize=14)

    finalize_figure(fig, "politica_final_policy_iteration.png")
    return fig, ax


def plot_trajectory(V, trajectory, title="Run com política final"):
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
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        fill=True,
                        alpha=0.7,
                        color="gray",
                    )
                )
                ax.text(
                    c,
                    r,
                    "W",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                )
            elif is_terminal(s):
                ax.text(
                    c,
                    r,
                    "T",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                )
            else:
                ax.text(c, r, f"{V[s]:.1f}", ha="center", va="center", fontsize=9)

    xs = [s[1] for s in trajectory]
    ys = [s[0] for s in trajectory]

    ax.plot(xs, ys, marker="o", linewidth=2)

    sr, sc = trajectory[0]
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold")

    for t, s in enumerate(trajectory):
        ax.text(s[1] + 0.18, s[0] - 0.18, str(t), fontsize=8)

    finalize_figure(fig, "trajetoria_policy_iteration.png")
    return fig, ax


def plot_action_preferences_initial_and_final(V_initial, V_final, state=START_STATE):
    q_initial = get_action_values(V_initial, state)
    q_final = get_action_values(V_final, state)

    actions = list(q_initial.keys())
    initial_values = [q_initial[a] for a in actions]
    final_values = [q_final[a] for a in actions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    bars1 = axes[0].bar(actions, initial_values)
    axes[0].set_title(f"Preferência por ação no início\nestado {state}")
    axes[0].set_ylabel("Q(s,a) = r + γV(s')")
    max_init = max(initial_values)

    for bar, val in zip(bars1, initial_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
        )

        if np.isclose(val, max_init):
            bar.set_linewidth(2)
            bar.set_edgecolor("black")

    bars2 = axes[1].bar(actions, final_values)
    axes[1].set_title(f"Preferência por ação ao final\nestado {state}")
    axes[1].set_ylabel("Q(s,a) = r + γV(s')")
    max_final = max(final_values)

    for bar, val in zip(bars2, final_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
        )

        if np.isclose(val, max_final):
            bar.set_linewidth(2)
            bar.set_edgecolor("black")

    finalize_figure(fig, "preferencia_acoes_inicio_e_final.png")
    return fig, axes


def plot_action_preferences(V, state=START_STATE, title=None):
    q = get_action_values(V, state)
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
            va="bottom" if val >= 0 else "top",
        )

        if np.isclose(val, max_val):
            bar.set_linewidth(2)
            bar.set_edgecolor("black")

    finalize_figure(fig, "preferencia_acoes_estado_inicial_final.png")
    return fig, ax


def plot_policy_eval_iterations(policy_eval_counts):
    fig, ax = plt.subplots(figsize=(7, 4))

    xs = np.arange(1, len(policy_eval_counts) + 1)
    ax.plot(xs, policy_eval_counts, marker="o")

    ax.set_title("Iterações internas de policy evaluation por ciclo")
    ax.set_xlabel("Ciclo de policy iteration")
    ax.set_ylabel("Número de iterações internas")

    finalize_figure(fig, "iteracoes_policy_evaluation_por_ciclo.png")
    return fig, ax


# =========================
# IMPRESSÃO AUXILIAR
# =========================

def print_run_summary(trajectory, actions_taken, rewards):
    print("\n=== RESUMO DA RUN COM POLÍTICA FINAL ===")

    for t in range(len(actions_taken)):
        print(
            f"Passo {t}: estado={trajectory[t]} | ação={actions_taken[t]} | "
            f"recompensa={rewards[t]} | próximo={trajectory[t + 1]}"
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

    V_initial, V_final, policy, n_policy_iterations, policy_eval_counts = policy_iteration()

    trajectory, actions_taken, rewards = policy_run(
        policy,
        start_state=START_STATE,
        max_steps=100,
        seed=42,
    )

    plot_value_heatmap(
        V_final,
        title="Heatmap dos valores finais (Policy Iteration)",
    )

    plot_policy(
        policy,
        title="Política final (Policy Iteration)",
    )

    plot_trajectory(
        V_final,
        trajectory,
        title="Trajetória com política final",
    )

    plot_action_preferences_initial_and_final(
        V_initial,
        V_final,
        state=START_STATE,
    )

    plot_policy_eval_iterations(policy_eval_counts)

    print(f"\nNúmero de ciclos de policy iteration: {n_policy_iterations}")
    print(f"Iterações internas por ciclo: {policy_eval_counts}")

    print_run_summary(trajectory, actions_taken, rewards)
    print_policy_probabilities(policy)