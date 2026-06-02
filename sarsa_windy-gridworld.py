import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

OUTPUT_DIR = r"C:\Kegle_Jojo\RL-examples\SARSA-Windy-Gridworld"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_GPU = False
SEED = 42

np.random.seed(SEED)

if USE_GPU:
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    print(f"GPU habilitada: {DEVICE}")

else:
    DEVICE = "cpu"
    print("Executando em CPU")

# ============================================================
# PARÂMETROS DO GRIDWORLD
# ============================================================

WORLD_HEIGHT = 7
WORLD_WIDTH = 10

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1.0
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]

EPISODE_LIMIT = 5000

# ============================================================
# Q TABLE CPU / GPU
# ============================================================

def create_q_value():

    if USE_GPU:
        return torch.zeros(
            (WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)),
            dtype=torch.float32,
            device=DEVICE
        )

    return np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))


def get_q_values(q_value, state):

    i, j = state

    if USE_GPU:
        return q_value[i, j, :].detach().cpu().numpy()

    return q_value[i, j, :]


def update_q_value(q_value, state, action, next_state, next_action):

    i, j = state
    ni, nj = next_state

    td_target = (
        REWARD
        + GAMMA * q_value[ni, nj, next_action]
    )

    td_error = td_target - q_value[i, j, action]

    q_value[i, j, action] += ALPHA * td_error


# ============================================================
# DINÂMICA DO AMBIENTE
# ============================================================

def step(state, action):

    i, j = state

    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]

    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]

    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]

    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]

    raise ValueError("Ação inválida")


# ============================================================
# POLÍTICAS
# ============================================================

def epsilon_greedy(q_value, state, epsilon=EPSILON):

    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)

    values = get_q_values(q_value, state)

    best_actions = np.flatnonzero(
        values == np.max(values)
    )

    return np.random.choice(best_actions)


def stochastic_visual_policy(q_value, state):
    return epsilon_greedy(q_value, state, epsilon=0.1)


def greedy_policy(q_value, state):

    values = get_q_values(q_value, state)

    best_actions = np.flatnonzero(
        values == np.max(values)
    )

    return np.random.choice(best_actions)


# ============================================================
# EPISÓDIO SARSA
# ============================================================

def episode(q_value, max_steps=1000):

    time = 0

    state = START.copy()

    action = epsilon_greedy(q_value, state)

    visited = {}

    while state != GOAL:

        state_tuple = tuple(state)

        visited[state_tuple] = (
            visited.get(state_tuple, 0) + 1
        )

        # ==============================================
        # DETECÇÃO DE LOOP
        # ==============================================

        if visited[state_tuple] > 30:
            return time, False

        if time >= max_steps:
            return time, False

        next_state = step(state, action)

        next_action = epsilon_greedy(
            q_value,
            next_state
        )

        update_q_value(
            q_value=q_value,
            state=state,
            action=action,
            next_state=next_state,
            next_action=next_action
        )

        state = next_state
        action = next_action

        time += 1

    return time, True


# ============================================================
# TRAJETÓRIAS
# ============================================================

def generate_trajectory(
    q_value,
    policy_func,
    max_steps=100
):

    trajectory = [START.copy()]
    actions_taken = []

    state = START.copy()

    visited = {}

    loop_detected = False

    for _ in range(max_steps):

        if state == GOAL:
            break

        state_tuple = tuple(state)

        visited[state_tuple] = (
            visited.get(state_tuple, 0) + 1
        )

        if visited[state_tuple] > 4:
            loop_detected = True
            break

        action = policy_func(q_value, state)

        next_state = step(state, action)

        actions_taken.append(action)

        trajectory.append(next_state.copy())

        state = next_state

    reached_goal = trajectory[-1] == GOAL

    return (
        trajectory,
        actions_taken,
        reached_goal,
        loop_detected
    )


def action_to_symbol(action):

    if action == ACTION_UP:
        return "↑"

    elif action == ACTION_DOWN:
        return "↓"

    elif action == ACTION_LEFT:
        return "←"

    elif action == ACTION_RIGHT:
        return "→"


# ============================================================
# PLOT TRAJETÓRIA
# ============================================================

def plot_trajectory(
    q_value,
    title,
    filename,
    policy_func,
    max_steps=100
):

    (
        trajectory,
        actions_taken,
        reached_goal,
        loop_detected
    ) = generate_trajectory(
        q_value,
        policy_func,
        max_steps
    )

    fig, ax = plt.subplots(figsize=(13, 8))

    ax.set_xlim(-0.5, WORLD_WIDTH - 0.5)
    ax.set_ylim(WORLD_HEIGHT - 0.5, -1.7)

    ax.set_xticks(np.arange(WORLD_WIDTH))
    ax.set_yticks(np.arange(WORLD_HEIGHT))

    ax.set_xlabel("Coluna")
    ax.set_ylabel("Linha")

    ax.set_title(title)

    ax.grid(True)

    # ========================================================
    # VENTO
    # ========================================================

    for j, wind_strength in enumerate(WIND):

        ax.text(
            j,
            -1.15,
            f"↑ {wind_strength}",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

        if wind_strength > 0:

            ax.arrow(
                j,
                -0.85,
                0,
                -0.2 * wind_strength,
                head_width=0.08,
                head_length=0.08,
                length_includes_head=True
            )

    # ========================================================
    # START / GOAL
    # ========================================================

    ax.scatter(
        START[1],
        START[0],
        s=240,
        marker="s",
        label="Start",
        zorder=5
    )

    ax.scatter(
        GOAL[1],
        GOAL[0],
        s=260,
        marker="*",
        label="Goal",
        zorder=5
    )

    ax.text(
        START[1],
        START[0],
        "S",
        ha="center",
        va="center",
        fontweight="bold"
    )

    ax.text(
        GOAL[1],
        GOAL[0],
        "G",
        ha="center",
        va="center",
        fontweight="bold"
    )

    # ========================================================
    # TRANSIÇÕES
    # ========================================================

    for k in range(len(trajectory) - 1):

        r1, c1 = trajectory[k]
        r2, c2 = trajectory[k + 1]

        dc = c2 - c1
        dr = r2 - r1

        ax.arrow(
            c1,
            r1,
            dc * 0.82,
            dr * 0.82,
            head_width=0.12,
            head_length=0.12,
            linewidth=1.5,
            length_includes_head=True,
            zorder=4
        )

        ax.text(
            c1 + 0.12,
            r1 - 0.12,
            action_to_symbol(actions_taken[k]),
            fontsize=10,
            fontweight="bold"
        )

        ax.text(
            c1 - 0.15,
            r1 + 0.15,
            str(k),
            fontsize=7
        )

    rows = [s[0] for s in trajectory]
    cols = [s[1] for s in trajectory]

    ax.scatter(cols, rows, s=35, zorder=5)

    # ========================================================
    # STATUS
    # ========================================================

    if reached_goal:

        info = (
            f"Chegou ao objetivo em "
            f"{len(trajectory)-1} passos"
        )

    elif loop_detected:

        info = (
            f"Loop detectado após "
            f"{len(trajectory)-1} transições"
        )

    else:

        info = (
            f"Não chegou ao objetivo "
            f"em {max_steps} passos"
        )

    ax.text(
        0,
        WORLD_HEIGHT + 0.2,
        info,
        fontsize=11,
        fontweight="bold"
    )

    ax.legend()

    image_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Imagem salva em: {image_path}")


# ============================================================
# CURVA CLÁSSICA DO LIVRO
# ============================================================

def plot_learning_curve(steps):

    cumulative_steps = np.add.accumulate(steps)

    plt.figure(figsize=(10, 6))

    plt.plot(
        cumulative_steps,
        np.arange(1, len(cumulative_steps) + 1),
        linewidth=1.5
    )

    plt.xlabel("Time steps")
    plt.ylabel("Episodes")

    plt.title(
        "Curva clássica do SARSA "
        "(Time steps vs Episodes)"
    )

    plt.grid(True)

    image_path = os.path.join(
        OUTPUT_DIR,
        "learning_curve.png"
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Curva salva em: {image_path}")


# ============================================================
# NOVO GRÁFICO DE DESEMPENHO
# ============================================================

def plot_episode_performance(
    episode_steps,
    success_flags
):

    plt.figure(figsize=(14, 7))

    successful_x = []
    successful_y = []

    failed_x = []
    failed_y = []

    for ep, (steps, success) in enumerate(
        zip(episode_steps, success_flags),
        start=1
    ):

        if success:

            successful_x.append(ep)
            successful_y.append(steps)

        else:

            failed_x.append(ep)
            failed_y.append(steps)

    # ========================================================
    # EPISÓDIOS BEM SUCEDIDOS
    # ========================================================

    plt.plot(
        successful_x,
        successful_y,
        linewidth=1.0,
        label="Chegou ao objetivo"
    )

    # ========================================================
    # LOOPS / FALHAS
    # ========================================================

    plt.scatter(
        failed_x,
        failed_y,
        marker="x",
        s=50,
        linewidths=1.5,
        label="Loop ou falha"
    )

    plt.xlabel("Episódio")
    plt.ylabel("Passos")

    plt.title(
        "Passos até o objetivo "
        "ao longo do treinamento"
    )

    plt.yscale("log")

    plt.grid(True)

    plt.legend()

    image_path = os.path.join(
        OUTPUT_DIR,
        "episode_performance.png"
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Performance salva em: {image_path}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def example_6_5():

    q_value = create_q_value()

    checkpoints = {

        0: {
            "filename":
                "trajetoria_0_inicial.png",

            "title":
                "Política inicial estocástica",

            "policy":
                stochastic_visual_policy
        },

        EPISODE_LIMIT // 3: {
            "filename":
                "trajetoria_1_terco.png",

            "title":
                f"Após {EPISODE_LIMIT // 3} episódios",

            "policy":
                stochastic_visual_policy
        },

        2 * EPISODE_LIMIT // 3: {
            "filename":
                "trajetoria_2_tercos.png",

            "title":
                f"Após {2 * EPISODE_LIMIT // 3} episódios",

            "policy":
                stochastic_visual_policy
        },

        EPISODE_LIMIT: {
            "filename":
                "trajetoria_final_greedy.png",

            "title":
                f"Política final greedy ({EPISODE_LIMIT} episódios)",

            "policy":
                greedy_policy
        }
    }

    # ========================================================
    # TRAJETÓRIA INICIAL
    # ========================================================

    plot_trajectory(
        q_value=q_value,
        title=checkpoints[0]["title"],
        filename=checkpoints[0]["filename"],
        policy_func=checkpoints[0]["policy"]
    )

    steps = []
    success_flags = []

    # ========================================================
    # TREINAMENTO
    # ========================================================

    for ep in range(1, EPISODE_LIMIT + 1):

        ep_steps, success = episode(q_value)

        steps.append(ep_steps)
        success_flags.append(success)

        if ep in checkpoints:

            plot_trajectory(
                q_value=q_value,
                title=checkpoints[ep]["title"],
                filename=checkpoints[ep]["filename"],
                policy_func=checkpoints[ep]["policy"]
            )

    # ========================================================
    # GRÁFICOS FINAIS
    # ========================================================

    plot_learning_curve(steps)

    plot_episode_performance(
        episode_steps=steps,
        success_flags=success_flags
    )

    print("\nTreinamento finalizado.")


# ============================================================
# EXECUÇÃO
# ============================================================

example_6_5()