import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÕES
# ============================================================

OUTPUT_DIR = r"C:\Kegle_Jojo\RL-examples\sarsa-qlearning-expected-Sarsa_cliffwalking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_GPU = False

DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

print(f"Usando device: {DEVICE}")

WORLD_HEIGHT = 4
WORLD_WIDTH = 12

EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1.0

EPISODES = 500
RUNS = 200
MAX_STEPS_PER_EPISODE = 5000

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
N_ACTIONS = 4

START = torch.tensor([3, 0], device=DEVICE)
GOAL = torch.tensor([3, 11], device=DEVICE)

CHECKPOINTS = [
    0,
    EPISODES // 3,
    2 * EPISODES // 3,
    EPISODES
]


# ============================================================
# FUNÇÕES DO AMBIENTE
# ============================================================

def batched_step(states, actions):
    """
    states: tensor [runs, 2]
    actions: tensor [runs]
    """

    i = states[:, 0]
    j = states[:, 1]

    next_i = i.clone()
    next_j = j.clone()

    next_i = torch.where(actions == ACTION_UP, torch.clamp(i - 1, min=0), next_i)
    next_i = torch.where(actions == ACTION_DOWN, torch.clamp(i + 1, max=WORLD_HEIGHT - 1), next_i)

    next_j = torch.where(actions == ACTION_LEFT, torch.clamp(j - 1, min=0), next_j)
    next_j = torch.where(actions == ACTION_RIGHT, torch.clamp(j + 1, max=WORLD_WIDTH - 1), next_j)

    rewards = torch.full((states.shape[0],), -1.0, device=DEVICE)

    fall_from_above = (
        (actions == ACTION_DOWN)
        & (i == 2)
        & (j >= 1)
        & (j <= 10)
    )

    fall_from_start = (
        (actions == ACTION_RIGHT)
        & (i == 3)
        & (j == 0)
    )

    fallen = fall_from_above | fall_from_start

    rewards = torch.where(fallen, torch.tensor(-100.0, device=DEVICE), rewards)

    next_i = torch.where(fallen, START[0], next_i)
    next_j = torch.where(fallen, START[1], next_j)

    next_states = torch.stack([next_i, next_j], dim=1)

    return next_states, rewards


def choose_action_batched(states, q_values):
    """
    Política epsilon-greedy paralelizada.
    q_values: [runs, height, width, actions]
    states: [runs, 2]
    """

    runs = states.shape[0]

    q_state = q_values[
        torch.arange(runs, device=DEVICE),
        states[:, 0],
        states[:, 1],
        :
    ]

    greedy_actions = torch.argmax(q_state, dim=1)

    random_actions = torch.randint(
        low=0,
        high=N_ACTIONS,
        size=(runs,),
        device=DEVICE
    )

    explore = torch.rand(runs, device=DEVICE) < EPSILON

    actions = torch.where(explore, random_actions, greedy_actions)

    return actions


def is_goal(states):
    return (states[:, 0] == GOAL[0]) & (states[:, 1] == GOAL[1])


# ============================================================
# EPISÓDIO BATELADO
# ============================================================

def run_episode_batched(q_values, algorithm="sarsa", alpha=ALPHA):
    """
    algorithm:
        "sarsa"
        "expected_sarsa"
        "q_learning"
    """

    runs = q_values.shape[0]

    states = START.repeat(runs, 1)
    actions = choose_action_batched(states, q_values)

    total_rewards = torch.zeros(runs, device=DEVICE)
    done = torch.zeros(runs, dtype=torch.bool, device=DEVICE)

    for _ in range(MAX_STEPS_PER_EPISODE):

        active = ~done

        if not torch.any(active):
            break

        next_states, rewards = batched_step(states, actions)

        total_rewards += torch.where(active, rewards, torch.zeros_like(rewards))

        next_actions = choose_action_batched(next_states, q_values)

        run_ids = torch.arange(runs, device=DEVICE)

        q_current = q_values[
            run_ids,
            states[:, 0],
            states[:, 1],
            actions
        ]

        q_next_all = q_values[
            run_ids,
            next_states[:, 0],
            next_states[:, 1],
            :
        ]

        if algorithm == "sarsa":

            target_next = q_values[
                run_ids,
                next_states[:, 0],
                next_states[:, 1],
                next_actions
            ]

        elif algorithm == "expected_sarsa":

            greedy_actions = torch.argmax(q_next_all, dim=1)

            probs = torch.full_like(q_next_all, EPSILON / N_ACTIONS)

            probs[
                run_ids,
                greedy_actions
            ] += 1.0 - EPSILON

            target_next = torch.sum(probs * q_next_all, dim=1)

        elif algorithm == "q_learning":

            target_next = torch.max(q_next_all, dim=1).values

        else:
            raise ValueError("Algoritmo inválido.")

        target = rewards + GAMMA * target_next

        td_error = target - q_current

        updated_q = q_current + alpha * td_error

        q_values[
            run_ids[active],
            states[active, 0],
            states[active, 1],
            actions[active]
        ] = updated_q[active]

        states = torch.where(active.unsqueeze(1), next_states, states)

        if algorithm in ["sarsa", "expected_sarsa"]:
            actions = torch.where(active, next_actions, actions)
        else:
            actions = choose_action_batched(states, q_values)

        done = done | is_goal(states)

    return total_rewards


# ============================================================
# POLÍTICA GREEDY E TRAJETÓRIA
# ============================================================

def greedy_policy_numpy(q_single):
    policy = np.argmax(q_single, axis=2)
    return policy


def simulate_greedy_path(q_single, max_steps=100):
    """
    Simula caminho usando política greedy aprendida.
    """

    state = [3, 0]
    path = [tuple(state)]

    for _ in range(max_steps):

        if state == [3, 11]:
            break

        action = int(np.argmax(q_single[state[0], state[1], :]))

        i, j = state

        if action == ACTION_UP:
            next_state = [max(i - 1, 0), j]
        elif action == ACTION_DOWN:
            next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
        elif action == ACTION_LEFT:
            next_state = [i, max(j - 1, 0)]
        elif action == ACTION_RIGHT:
            next_state = [i, min(j + 1, WORLD_WIDTH - 1)]

        if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
            action == ACTION_RIGHT and state == [3, 0]
        ):
            next_state = [3, 0]

        state = next_state
        path.append(tuple(state))

    return path


def action_to_arrow(action):
    if action == ACTION_UP:
        return "↑"
    if action == ACTION_DOWN:
        return "↓"
    if action == ACTION_LEFT:
        return "←"
    if action == ACTION_RIGHT:
        return "→"


def plot_policy_path(ax, q_single, title):
    policy = greedy_policy_numpy(q_single)
    path = simulate_greedy_path(q_single)

    ax.set_title(title)
    ax.set_xlim(-0.5, WORLD_WIDTH - 0.5)
    ax.set_ylim(WORLD_HEIGHT - 0.5, -0.5)
    ax.set_xticks(range(WORLD_WIDTH))
    ax.set_yticks(range(WORLD_HEIGHT))
    ax.grid(True)

    for i in range(WORLD_HEIGHT):
        for j in range(WORLD_WIDTH):

            if [i, j] == [3, 11]:
                ax.text(j, i, "G", ha="center", va="center", fontsize=12)
            elif i == 3 and 1 <= j <= 10:
                ax.text(j, i, "C", ha="center", va="center", fontsize=12)
            elif [i, j] == [3, 0]:
                ax.text(j, i, "S", ha="center", va="center", fontsize=12)
            else:
                ax.text(
                    j,
                    i,
                    action_to_arrow(policy[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12
                )

    if len(path) > 1:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, marker="o", linewidth=2)


# ============================================================
# TREINAMENTO PRINCIPAL
# ============================================================

def train_all_algorithms():

    algorithms = {
        "Sarsa": "sarsa",
        "Expected Sarsa": "expected_sarsa",
        "Q-Learning": "q_learning"
    }

    rewards_history = {}

    checkpoints_q = {
        name: {}
        for name in algorithms.keys()
    }

    for name, algorithm in algorithms.items():

        print(f"\nTreinando {name}...")

        q_values = torch.zeros(
            (RUNS, WORLD_HEIGHT, WORLD_WIDTH, N_ACTIONS),
            device=DEVICE
        )

        rewards = torch.zeros(EPISODES, device=DEVICE)

        for ep in tqdm(range(EPISODES)):

            if ep in CHECKPOINTS:
                checkpoints_q[name][ep] = q_values[0].detach().cpu().numpy().copy()

            episode_rewards = run_episode_batched(
                q_values,
                algorithm=algorithm,
                alpha=ALPHA
            )

            rewards[ep] = episode_rewards.mean()

        checkpoints_q[name][EPISODES] = q_values[0].detach().cpu().numpy().copy()

        rewards_history[name] = rewards.detach().cpu().numpy()

        np.save(
            os.path.join(OUTPUT_DIR, f"q_values_{name.replace(' ', '_')}.npy"),
            q_values.detach().cpu().numpy()
        )

    return rewards_history, checkpoints_q


# ============================================================
# PLOTS
# ============================================================

def plot_average_rewards(rewards_history):

    plt.figure(figsize=(10, 6))

    for name, rewards in rewards_history.items():
        plt.plot(rewards, label=name)

    plt.xlabel("Episodes")
    plt.ylabel("Average reward per episode")
    plt.title("Recompensa média por episódio")
    plt.ylim([-100, 0])
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(OUTPUT_DIR, "average_rewards_sarsa_expected_qlearning.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_all_checkpoint_paths(checkpoints_q):

    for algorithm_name, checkpoints in checkpoints_q.items():

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        for ax, checkpoint in zip(axes, CHECKPOINTS):

            q_single = checkpoints[checkpoint]

            if checkpoint == 0:
                title = f"{algorithm_name}\nPolítica inicial"
            elif checkpoint == EPISODES:
                title = f"{algorithm_name}\nPolítica final"
            else:
                title = f"{algorithm_name}\nEpisódio {checkpoint}"

            plot_policy_path(ax, q_single, title)

        plt.tight_layout()

        filename = f"greedy_paths_{algorithm_name.replace(' ', '_')}.png"

        plt.savefig(
            os.path.join(OUTPUT_DIR, filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    rewards_history, checkpoints_q = train_all_algorithms()

    plot_average_rewards(rewards_history)

    plot_all_checkpoint_paths(checkpoints_q)

    print("\nOutputs salvos em:")
    print(OUTPUT_DIR)