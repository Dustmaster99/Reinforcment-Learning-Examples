# train_a2c_simple_pong_parallel.py

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import sys

PROJECT_DIR = r"C:\Kegle_Jojo\Reinforcment-Learning-Examples\Pong_A2C"
sys.path.insert(0, PROJECT_DIR)

from single_pong_env import SimplePongEnv


# ============================================================
# PARÂMETROS DE TREINAMENTO
# ============================================================

EPISODES = 10000
NUM_WORKERS = 16
N_STEPS = 64
GAMMA = 0.99
LR = 3e-4

HIDDEN_SIZE = 256

ENTROPY_COEF = 0.001
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

USE_GPU = True
NORMALIZE_ADVANTAGE = False

PLOT_REWARD_PATH = "a2c_parallel_rewards_mean.png"
PLOT_LOSS_PATH = "a2c_parallel_losses_advantage.png"


class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


def build_obs_normalizer(env):
    obs_low = env.observation_space.low.astype(np.float32)
    obs_high = env.observation_space.high.astype(np.float32)

    if np.any(np.isinf(obs_low)) or np.any(np.isinf(obs_high)):
        raise ValueError("observation_space possui limites infinitos.")

    if np.any(obs_high <= obs_low):
        raise ValueError("observation_space possui limites inválidos.")

    def normalize_obs(obs):
        obs = np.asarray(obs, dtype=np.float32)

        obs_norm = 2.0 * (obs - obs_low) / (obs_high - obs_low + 1e-8) - 1.0

        return np.clip(obs_norm, -1.0, 1.0).astype(np.float32)

    return normalize_obs


def compute_n_step_returns_parallel(rewards, dones, last_values, gamma, device):
    """
    Calcula os retornos n-step separadamente por worker.

    rewards: lista com N_STEPS tensores/arrays de shape [NUM_WORKERS]
    dones:   lista com N_STEPS tensores/arrays de shape [NUM_WORKERS]
    last_values: tensor shape [NUM_WORKERS]

    Retorna tensor [N_STEPS, NUM_WORKERS].
    """

    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.bool_)

    n_steps, num_workers = rewards.shape

    returns = np.zeros((n_steps, num_workers), dtype=np.float32)

    R = last_values.detach().cpu().numpy().astype(np.float32)

    for t in reversed(range(n_steps)):
        R = rewards[t] + gamma * R * (1.0 - dones[t].astype(np.float32))
        returns[t] = R

    return torch.tensor(returns, dtype=torch.float32, device=device)


def should_render_episode(total_finished_episodes, total_episodes):
    return (
        total_finished_episodes < 5
        or total_finished_episodes >= total_episodes - 5
    )


def train():
    start_time = time.time()

    device = torch.device(
        "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    )

    print(f"Usando dispositivo: {device}")
    print(f"Workers paralelos: {NUM_WORKERS}")
    print(f"Batch por update: {NUM_WORKERS} x {N_STEPS} = {NUM_WORKERS * N_STEPS}")

    envs = [
        SimplePongEnv(render_mode=None)
        for _ in range(NUM_WORKERS)
    ]

    normalize_obs = build_obs_normalizer(envs[0])

    obs_dim = envs[0].observation_space.shape[0]
    n_actions = envs[0].action_space.n

    model = ActorCriticMLP(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=HIDDEN_SIZE
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs_list = []
    for i, env in enumerate(envs):
        obs, info = env.reset(seed=1000 + i)
        obs_list.append(normalize_obs(obs))

    obs_batch = np.stack(obs_list, axis=0)

    total_finished_episodes = 0

    episode_rewards_per_worker = [[] for _ in range(NUM_WORKERS)]
    current_episode_rewards = np.zeros(NUM_WORKERS, dtype=np.float32)

    mean_episode_rewards = []

    actor_losses_history = []
    critic_losses_history = []
    total_losses_history = []
    advantage_mean_history = []

    update_count = 0

    while total_finished_episodes < EPISODES:
        update_count += 1

        batch_log_probs = []
        batch_values = []
        batch_rewards = []
        batch_dones = []
        batch_entropies = []

        for step in range(N_STEPS):
            render_this_step = should_render_episode(
                total_finished_episodes,
                EPISODES
            )

            for i, env in enumerate(envs):
                env.render_mode = "human" if i == 0 and render_this_step else None

            obs_tensor = torch.tensor(
                obs_batch,
                dtype=torch.float32,
                device=device
            )

            logits, values = model(obs_tensor)
            values = values.squeeze(-1)

            dist = Categorical(logits=logits)

            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

            actions_np = actions.detach().cpu().numpy()

            next_obs_list = []
            rewards_step = []
            dones_step = []

            for worker_id, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(
                    int(actions_np[worker_id])
                )

                done = terminated or truncated

                current_episode_rewards[worker_id] += reward

                if done:
                    episode_reward = current_episode_rewards[worker_id]

                    episode_rewards_per_worker[worker_id].append(float(episode_reward))
                    mean_episode_rewards.append(float(episode_reward))

                    current_episode_rewards[worker_id] = 0.0
                    total_finished_episodes += 1

                    next_obs, info = env.reset()

                    if total_finished_episodes >= EPISODES:
                        pass

                next_obs = normalize_obs(next_obs)

                next_obs_list.append(next_obs)
                rewards_step.append(float(reward))
                dones_step.append(bool(done))

            batch_log_probs.append(log_probs)
            batch_values.append(values)
            batch_rewards.append(rewards_step)
            batch_dones.append(dones_step)
            batch_entropies.append(entropies)

            obs_batch = np.stack(next_obs_list, axis=0)

            if total_finished_episodes >= EPISODES:
                break

        with torch.no_grad():
            next_obs_tensor = torch.tensor(
                obs_batch,
                dtype=torch.float32,
                device=device
            )

            _, next_values = model(next_obs_tensor)
            next_values = next_values.squeeze(-1)

        returns_tensor = compute_n_step_returns_parallel(
            rewards=batch_rewards,
            dones=batch_dones,
            last_values=next_values,
            gamma=GAMMA,
            device=device
        )

        values_tensor = torch.stack(batch_values, dim=0)
        log_probs_tensor = torch.stack(batch_log_probs, dim=0)
        entropies_tensor = torch.stack(batch_entropies, dim=0)

        returns_flat = returns_tensor.reshape(-1)
        values_flat = values_tensor.reshape(-1)
        log_probs_flat = log_probs_tensor.reshape(-1)
        entropies_flat = entropies_tensor.reshape(-1)

        advantages = returns_flat - values_flat.detach()

        if NORMALIZE_ADVANTAGE and len(advantages) > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs_flat * advantages).mean()
        critic_loss = (returns_flat - values_flat).pow(2).mean()
        entropy_loss = entropies_flat.mean()

        loss = (
            actor_loss
            + VALUE_COEF * critic_loss
            - ENTROPY_COEF * entropy_loss
        )

        optimizer.zero_grad()
        loss.backward()

        if MAX_GRAD_NORM > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
            )

        optimizer.step()

        actor_losses_history.append(float(actor_loss.detach().cpu().item()))
        critic_losses_history.append(float(critic_loss.detach().cpu().item()))
        total_losses_history.append(float(loss.detach().cpu().item()))
        advantage_mean_history.append(float(advantages.detach().mean().cpu().item()))

        if total_finished_episodes % 10 == 0 and total_finished_episodes > 0:
            avg_20 = np.mean(mean_episode_rewards[-20:])
            remaining = EPISODES - total_finished_episodes

            print(
                f"[{total_finished_episodes}/{EPISODES}] "
                f"Faltam: {remaining} episódios | "
                f"Média geral(20): {avg_20:.2f} | "
                f"Update: {update_count}"
            )

    for env in envs:
        env.close()

    plot_rewards_parallel(
        episode_rewards_per_worker,
        mean_episode_rewards
    )

    plot_losses_and_advantage(
        actor_losses_history,
        critic_losses_history,
        total_losses_history,
        advantage_mean_history
    )

    elapsed = time.time() - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n===================================")
    print("TREINAMENTO FINALIZADO")
    print("===================================")
    print(f"Tempo total: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print(f"Episódios finalizados: {total_finished_episodes}")
    print(f"Updates realizados: {update_count}")
    print("===================================")


def rolling_mean(values, window=20):
    values = np.array(values, dtype=np.float32)

    if len(values) == 0:
        return np.array([])

    result = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i + 1]))

    return np.array(result)


def plot_rewards_parallel(episode_rewards_per_worker, mean_episode_rewards):
    for group_start in range(0, NUM_WORKERS, 4):
        group_end = min(group_start + 4, NUM_WORKERS)

        plt.figure(figsize=(12, 6))

        for worker_id in range(group_start, group_end):
            rewards = episode_rewards_per_worker[worker_id]
            rm = rolling_mean(rewards, window=20)

            plt.plot(
                rm,
                label=f"Worker {worker_id + 1} - média móvel(20)"
            )

        plt.xlabel("Episódios finalizados por worker")
        plt.ylabel("Recompensa média móvel")
        plt.title(
            f"Recompensa média - Workers {group_start + 1} a {group_end}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        path = f"a2c_parallel_rewards_workers_{group_start + 1}_{group_end}.png"
        plt.savefig(path)
        plt.show()

        print(f"Gráfico salvo em: {path}")

    global_rm = rolling_mean(mean_episode_rewards, window=20)

    plt.figure(figsize=(10, 5))
    plt.plot(mean_episode_rewards, alpha=0.3, label="Recompensa por episódio")
    plt.plot(global_rm, label="Média móvel global(20)")
    plt.xlabel("Episódio finalizado global")
    plt.ylabel("Recompensa")
    plt.title("Treinamento A2C Paralelo - Recompensa Global")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_REWARD_PATH)
    plt.show()

    print(f"Gráfico salvo em: {PLOT_REWARD_PATH}")


def plot_losses_and_advantage(
    actor_losses,
    critic_losses,
    total_losses,
    advantage_means
):
    plt.figure(figsize=(12, 6))
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.plot(total_losses, label="Global Loss")
    plt.plot(advantage_means, label="Vantagem média")
    plt.xlabel("Update")
    plt.ylabel("Valor")
    plt.title("Losses e Vantagem Média - Batch Concatenado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_LOSS_PATH)
    plt.show()

    print(f"Gráfico salvo em: {PLOT_LOSS_PATH}")


if __name__ == "__main__":
    if N_STEPS < 1:
        raise ValueError("N_STEPS precisa ser maior ou igual a 1.")

    if NUM_WORKERS < 1:
        raise ValueError("NUM_WORKERS precisa ser maior ou igual a 1.")

    train()