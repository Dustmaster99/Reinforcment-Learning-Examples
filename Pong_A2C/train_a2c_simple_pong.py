# train_a2c_simple_pong.py

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


EPISODES = 10000
N_STEPS = 64
GAMMA = 0.99
LR = 3e-4

HIDDEN_SIZE = 256

ENTROPY_COEF = 0.001
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

USE_GPU = False
NORMALIZE_ADVANTAGE = False

PLOT_PATH = "a2c_simple_pong_rewards.png"


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


def compute_n_step_returns(rewards, dones, last_value, gamma):
    returns = []
    R = last_value

    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0.0

        R = reward + gamma * R
        returns.insert(0, R)

    return returns


def should_render_episode(episode, total_episodes):
    return episode <= 5 or episode > total_episodes - 5


def train():
    start_time = time.time()

    device = torch.device(
        "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    )

    print(f"Usando dispositivo: {device}")

    env = SimplePongEnv(render_mode=None)

    normalize_obs = build_obs_normalizer(env)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCriticMLP(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=HIDDEN_SIZE
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_rewards = []

    for episode in range(1, EPISODES + 1):
        render_this_episode = should_render_episode(episode, EPISODES)
        env.render_mode = "human" if render_this_episode else None

        obs, info = env.reset()
        obs = normalize_obs(obs)

        done = False
        episode_reward = 0

        log_probs = []
        values = []
        rewards = []
        dones = []
        entropies = []

        while not done:
            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            logits, value = model(obs_tensor)
            dist = Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            next_obs = normalize_obs(next_obs)

            done = terminated or truncated

            log_probs.append(log_prob.squeeze())
            values.append(value.squeeze())
            rewards.append(float(reward))
            dones.append(done)
            entropies.append(entropy.squeeze())

            obs = next_obs
            episode_reward += reward

            if len(rewards) >= N_STEPS or done:
                with torch.no_grad():
                    if done:
                        last_value = 0.0
                    else:
                        next_obs_tensor = torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=device
                        ).unsqueeze(0)

                        _, next_value = model(next_obs_tensor)
                        last_value = next_value.item()

                returns = compute_n_step_returns(
                    rewards=rewards,
                    dones=dones,
                    last_value=last_value,
                    gamma=GAMMA
                )

                returns = torch.tensor(
                    returns,
                    dtype=torch.float32,
                    device=device
                )

                values_tensor = torch.stack(values)
                log_probs_tensor = torch.stack(log_probs)
                entropies_tensor = torch.stack(entropies)

                advantages = returns - values_tensor.detach()

                if NORMALIZE_ADVANTAGE and len(advantages) > 1:
                    advantages = (
                        advantages - advantages.mean()
                    ) / (advantages.std() + 1e-8)

                actor_loss = -(log_probs_tensor * advantages).mean()
                critic_loss = (returns - values_tensor).pow(2).mean()
                entropy_loss = entropies_tensor.mean()

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

                log_probs = []
                values = []
                rewards = []
                dones = []
                entropies = []

        episode_rewards.append(episode_reward)

        if episode == 5:
            env.close()

        if episode % 10 == 0:
            avg_20 = np.mean(episode_rewards[-20:])
            remaining = EPISODES - episode

            print(
                f"[{episode}/{EPISODES}] "
                f"Faltam: {remaining} episódios | "
                f"Reward: {episode_reward:.2f} | "
                f"Média(20): {avg_20:.2f}"
            )

    env.close()
    plot_rewards(episode_rewards)

    elapsed = time.time() - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n===================================")
    print("TREINAMENTO FINALIZADO")
    print("===================================")
    print(f"Tempo total: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print("===================================")


def plot_rewards(episode_rewards):
    rewards = np.array(episode_rewards)
    rolling_mean = []

    for i in range(len(rewards)):
        start = max(0, i - 19)
        rolling_mean.append(np.mean(rewards[start:i + 1]))

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Recompensa por episódio")
    plt.plot(rolling_mean, label="Média móvel últimos 20 episódios")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa")
    plt.title("Treinamento A2C n-step no Simple Pong")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()

    print(f"Gráfico salvo em: {PLOT_PATH}")


if __name__ == "__main__":
    if N_STEPS < 1:
        raise ValueError("N_STEPS precisa ser maior ou igual a 1.")

    train()