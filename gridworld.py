import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
# Gridworld + TD(0) + epsilon-greedy
# ============================================================
#
# Objetivo:
# - Gridworld n x n
# - agente começa em posição aleatória a cada episódio
#   (exceto o estado terminal)
# - objetivo é chegar em (n-1, n-1)
#
# Recompensas:
# - movimento válido: step_reward
# - bater na parede: wall_reward
# - chegar ao objetivo: goal_reward
#
# Bellman para avaliação de política:
#
#   V^pi(s) = E_pi [ R_{t+1} + gamma * V^pi(S_{t+1}) | S_t = s ]
#
# TD(0) usa uma amostra dessa esperança:
#
#   V(S_t) <- V(S_t) + alpha * [ R_{t+1} + gamma * V(S_{t+1}) - V(S_t) ]
#
# O termo entre colchetes é o erro TD:
#
#   delta_t = R_{t+1} + gamma * V(S_{t+1}) - V(S_t)
#
# A política usada no treino é epsilon-greedy sobre V(s):
# escolhemos, em geral, a ação que leva ao próximo estado com maior V,
# mas com probabilidade epsilon exploramos uma ação aleatória.
# ============================================================


State = Tuple[int, int]


@dataclass
class GridWorldConfig:
    n: int = 6
    step_reward: float = -1.0
    wall_reward: float = -100.0
    goal_reward: float = 0.0
    gamma: float = 0.95
    alpha: float = 0.10
    epsilon: float = 0.10
    n_episodes: int = 500
    max_steps_per_episode: int = 200
    random_seed: int = 42


class GridWorld:
    """
    Gridworld n x n.

    - O objetivo é a célula (n-1, n-1)
    - O início do treino é aleatório a cada episódio,
      exceto a célula objetivo
    """

    def __init__(self, config: GridWorldConfig):
        self.cfg = config
        self.n = config.n
        self.default_start_state: State = (0, 0)  # usado só em referências visuais, se necessário
        self.goal_state: State = (self.n - 1, self.n - 1)
        self.rng = np.random.default_rng(config.random_seed)

        self.actions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

    def is_terminal(self, state: State) -> bool:
        return state == self.goal_state

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def reset(self, random_start: bool = True) -> State:
        """
        Reinicia o episódio.

        Se random_start=True:
            sorteia uniformemente qualquer estado que não seja a meta.
        """
        if not random_start:
            return self.default_start_state

        valid_states = [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if (r, c) != self.goal_state
        ]
        idx = self.rng.integers(0, len(valid_states))
        return valid_states[idx]

    def step(self, state: State, action_name: str):
        """
        Executa uma ação.

        Retorna:
            next_state, reward, done
        """
        if self.is_terminal(state):
            return state, 0.0, True

        dr, dc = self.actions[action_name]
        nr, nc = state[0] + dr, state[1] + dc

        # bateu na parede -> permanece no mesmo estado
        if not self.in_bounds(nr, nc):
            return state, self.cfg.wall_reward, False

        next_state = (nr, nc)

        if next_state == self.goal_state:
            return next_state, self.cfg.goal_reward, True

        return next_state, self.cfg.step_reward, False


class TDValueAgent:
    """
    Agente que aprende V(s) com TD(0).

    A política gulosa usa o valor do próximo estado:
        a*(s) = argmax_a V(next_state(s,a))
    """

    def __init__(self, env: GridWorld, config: GridWorldConfig):
        self.env = env
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)

        # Inicialização pedida: todos os estados com valor 0
        self.V = np.zeros((config.n, config.n), dtype=float)
        self.V[self.env.goal_state] = 0.0

        self.actions = list(self.env.actions.keys())

    def greedy_action(self, state: State) -> str:
        """
        Escolhe a ação cujo próximo estado tenha maior valor estimado.
        Desempate aleatório.
        """
        values = []
        for a in self.actions:
            next_state, _, _ = self.env.step(state, a)
            values.append(self.V[next_state])

        max_value = np.max(values)
        best_actions = [a for a, v in zip(self.actions, values) if np.isclose(v, max_value)]
        return self.rng.choice(best_actions)

    def epsilon_greedy_action(self, state: State) -> str:
        if self.rng.random() < self.cfg.epsilon:
            return self.rng.choice(self.actions)
        return self.greedy_action(state)

    def td_update(self, state: State, reward: float, next_state: State, done: bool) -> float:
        """
        Atualização TD(0):

            V(S_t) <- V(S_t) + alpha * [R_{t+1} + gamma*V(S_{t+1}) - V(S_t)]

        Se next_state for terminal:
            target = reward
        senão:
            target = reward + gamma * V(next_state)
        """
        target = reward if done else reward + self.cfg.gamma * self.V[next_state]
        td_error = target - self.V[state]
        self.V[state] += self.cfg.alpha * td_error
        return td_error

    def extract_greedy_path(self, start_state: Optional[State] = None,
                            max_steps: Optional[int] = None) -> List[State]:
        """
        Extrai o caminho guloso a partir de um estado inicial.
        """
        if max_steps is None:
            max_steps = self.cfg.max_steps_per_episode

        if start_state is None:
            start_state = self.env.reset(random_start=True)

        path = [start_state]
        state = start_state
        visited = {state}

        for _ in range(max_steps):
            if self.env.is_terminal(state):
                break

            action = self.greedy_action(state)
            next_state, _, done = self.env.step(state, action)
            path.append(next_state)

            if next_state in visited and not done:
                break

            visited.add(next_state)
            state = next_state

            if done:
                break

        return path

    def train(self):
        """
        Treinamento com início aleatório em cada episódio.

        Também salva 5 checkpoints igualmente espaçados.
        """
        steps_per_episode = []
        returns_per_episode = []
        wall_hits_per_episode = []

        checkpoints = np.linspace(1, self.cfg.n_episodes, 5, dtype=int)
        checkpoint_set = set(checkpoints.tolist())

        snapshots = []
        snapshot_episodes = []

        for episode in range(1, self.cfg.n_episodes + 1):
            state = self.env.reset(random_start=True)
            total_reward = 0.0
            n_steps = 0
            wall_hits = 0

            for _ in range(self.cfg.max_steps_per_episode):
                action = self.epsilon_greedy_action(state)
                next_state, reward, done = self.env.step(state, action)

                if reward == self.cfg.wall_reward:
                    wall_hits += 1

                # Atualização TD(0), associada diretamente à equação de Bellman
                self.td_update(state, reward, next_state, done)

                total_reward += reward
                n_steps += 1
                state = next_state

                if done:
                    break

            steps_per_episode.append(n_steps)
            returns_per_episode.append(total_reward)
            wall_hits_per_episode.append(wall_hits)

            if episode in checkpoint_set:
                snapshots.append(self.V.copy())
                snapshot_episodes.append(episode)

        return snapshots, snapshot_episodes, steps_per_episode, returns_per_episode, wall_hits_per_episode


def build_policy_arrows(agent: TDValueAgent):
    arrows = np.full((agent.cfg.n, agent.cfg.n), '', dtype=object)
    symbol = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→",
    }

    for r in range(agent.cfg.n):
        for c in range(agent.cfg.n):
            s = (r, c)
            if s == agent.env.goal_state:
                arrows[r, c] = "G"
            else:
                arrows[r, c] = symbol[agent.greedy_action(s)]

    return arrows


def plot_value_heatmap(ax, V, title, start, goal, path=None):
    im = ax.imshow(V, cmap="viridis", origin="upper")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(V.shape[1]))
    ax.set_yticks(range(V.shape[0]))
    ax.grid(True, color="white", linewidth=0.7, alpha=0.5)

    for r in range(V.shape[0]):
        for c in range(V.shape[1]):
            ax.text(c, r, f"{V[r, c]:.1f}", ha="center", va="center", color="white", fontsize=8)

    ax.scatter(start[1], start[0], marker="s", s=150)
    ax.scatter(goal[1], goal[0], marker="*", s=220)

    if path is not None and len(path) > 1:
        ys = [s[0] for s in path]
        xs = [s[1] for s in path]
        ax.plot(xs, ys, linewidth=2)
        ax.scatter(xs, ys, s=30)

    return im


def sample_fixed_start_states(env: GridWorld, n_starts: int, seed_offset: int = 999):
    """
    Sorteia estados iniciais fixos, uma única vez,
    para serem reaproveitados em todos os checkpoints.
    """
    rng = np.random.default_rng(env.cfg.random_seed + seed_offset)

    valid_starts = [
        (r, c)
        for r in range(env.n)
        for c in range(env.n)
        if (r, c) != env.goal_state
    ]

    if n_starts <= len(valid_starts):
        indices = rng.choice(len(valid_starts), size=n_starts, replace=False)
        fixed_starts = [valid_starts[i] for i in indices]
    else:
        fixed_starts = [
            valid_starts[rng.integers(0, len(valid_starts))]
            for _ in range(n_starts)
        ]

    return fixed_starts


def plot_policy_evolution(snapshots, snapshot_episodes, env, cfg):
    """
    Mostra a evolução da política usando os MESMOS estados iniciais
    nos checkpoints, para permitir comparação melhor.

    Aqui:
    - sorteamos 5 estados iniciais fixos uma única vez
    - cada checkpoint usa o seu start correspondente
    """
    n_plots = len(snapshots)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    fixed_starts = sample_fixed_start_states(env, n_plots, seed_offset=999)

    for ax, V_snapshot, ep, start_state in zip(axes, snapshots, snapshot_episodes, fixed_starts):
        temp_agent = TDValueAgent(env, cfg)
        temp_agent.V = V_snapshot.copy()
        path = temp_agent.extract_greedy_path(start_state=start_state)

        plot_value_heatmap(
            ax,
            V_snapshot,
            title=f"Checkpoint episódio {ep}\nstart fixo={start_state}",
            start=start_state,
            goal=env.goal_state,
            path=path,
        )

    for ax in axes[len(snapshots):]:
        ax.axis("off")

    fig.suptitle("Evolução da política gulosa implícita em V(s)", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_learning_curves(steps_per_episode, returns_per_episode, wall_hits_per_episode):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(steps_per_episode)
    axes[0].set_title("Passos por episódio")
    axes[0].set_xlabel("Episódio")
    axes[0].set_ylabel("Passos")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(returns_per_episode)
    axes[1].set_title("Retorno por episódio")
    axes[1].set_xlabel("Episódio")
    axes[1].set_ylabel("Retorno")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(wall_hits_per_episode)
    axes[2].set_title("Batidas na parede por episódio")
    axes[2].set_xlabel("Episódio")
    axes[2].set_ylabel("Colisões")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_final_policy_and_path(agent: TDValueAgent):
    """
    Visualização final:
    - mapa de V(s)
    - caminho guloso a partir de um estado aleatório
    - política gulosa em setas
    """
    start_state = agent.env.reset(random_start=True)
    final_path = agent.extract_greedy_path(start_state=start_state)
    arrows = build_policy_arrows(agent)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_value_heatmap(
        axes[0],
        agent.V,
        title=f"Mapa final de V(s) + caminho guloso\nstart={start_state}",
        start=start_state,
        goal=agent.env.goal_state,
        path=final_path,
    )

    axes[1].imshow(agent.V, cmap="viridis", origin="upper")
    axes[1].set_title("Política gulosa final (setas)")
    axes[1].set_xticks(range(agent.cfg.n))
    axes[1].set_yticks(range(agent.cfg.n))
    axes[1].grid(True, color="white", linewidth=0.7, alpha=0.5)

    for r in range(agent.cfg.n):
        for c in range(agent.cfg.n):
            axes[1].text(c, r, arrows[r, c], ha="center", va="center", color="white", fontsize=16)

    axes[1].scatter(start_state[1], start_state[0], marker="s", s=150)
    axes[1].scatter(agent.env.goal_state[1], agent.env.goal_state[0], marker="*", s=220)

    fig.tight_layout()
    plt.show()

    print("\nCaminho guloso final extraído:")
    print(final_path)


def main():
    cfg = GridWorldConfig(
        n=6,
        step_reward=-1.0,
        wall_reward=-100.0,
        goal_reward=0.0,
        gamma=0.95,
        alpha=0.10,
        epsilon=0.10,
        n_episodes=500,
        max_steps_per_episode=200,
        random_seed=42,
    )

    env = GridWorld(cfg)
    agent = TDValueAgent(env, cfg)

    print("=" * 60)
    print("Gridworld TD(0) com início aleatório por episódio")
    print("=" * 60)
    print(f"Tamanho do grid          : {cfg.n}x{cfg.n}")
    print("Estado inicial no treino : aleatório a cada episódio")
    print(f"Estado objetivo          : {env.goal_state}")
    print(f"Recompensa por passo     : {cfg.step_reward}")
    print(f"Recompensa por parede    : {cfg.wall_reward}")
    print(f"Gamma                    : {cfg.gamma}")
    print(f"Alpha                    : {cfg.alpha}")
    print(f"Epsilon                  : {cfg.epsilon}")
    print(f"Episódios                : {cfg.n_episodes}")
    print("=" * 60)

    print("\nValores iniciais V(s)=0:\n")
    print(agent.V)

    snapshots, snapshot_episodes, steps_per_episode, returns_per_episode, wall_hits_per_episode = agent.train()

    print("\nValores finais estimados:\n")
    print(np.round(agent.V, 2))

    # 5 checkpoints com starts fixos sorteados uma única vez
    plot_policy_evolution(snapshots, snapshot_episodes, env, cfg)

    # curvas do treinamento
    plot_learning_curves(steps_per_episode, returns_per_episode, wall_hits_per_episode)

    # política final
    plot_final_policy_and_path(agent)


if __name__ == "__main__":
    main()