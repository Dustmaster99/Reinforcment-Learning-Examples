# -*- coding: utf-8 -*-

import random
import math
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

EPISODES = 10_000_00
EPSILON_DECAY_STEPS = 1_000_000
WINDOW = 2000

EPSILON_START = 0.35
EPSILON_END = 0.05

ACE_BASE_POWER = 6000
ACE_INITIAL_LIFE = 1
ACE_INITIAL_HAND_SIZE = 2

TOTAL_DON = 10

USE_GPU = False# True para tentar usar GPU no sorteio das cartas

RANDOM_BUFFER_SIZE = 1_000_000

ATTACKERS = {
    "Leader Boa": 5000,
    "Gorgon 1": 5000,
    "Gorgon 2": 5000,
    "Boa 9c": 9000,
}

CARD_POOL = [1000] * 16 + [2000] * 8 + [0] * 26


# ============================================================
# GPU / TORCH
# ============================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    USE_GPU = False


if USE_GPU and TORCH_AVAILABLE and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU ativada: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    if USE_GPU:
        print("USE_GPU=True, mas CUDA não está disponível. Usando CPU.")
    else:
        print("GPU desativada. Usando CPU.")


# ============================================================
# BUFFER DE COMPRA DE CARTAS
# ============================================================

class CardDrawer:
    def __init__(self, card_pool, buffer_size=1_000_000, use_gpu=False):
        self.card_pool = card_pool
        self.buffer_size = buffer_size
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.index = 0
        self.buffer = []
        self.refill()

    def refill(self):
        if self.use_gpu:
            pool = torch.tensor(self.card_pool, device=DEVICE)
            idx = torch.randint(
                low=0,
                high=len(self.card_pool),
                size=(self.buffer_size,),
                device=DEVICE
            )
            self.buffer = pool[idx].cpu().tolist()
        else:
            self.buffer = random.choices(
                self.card_pool,
                k=self.buffer_size
            )

        self.index = 0

    def draw(self):
        if self.index >= len(self.buffer):
            self.refill()

        card = self.buffer[self.index]
        self.index += 1
        return card

    def draw_cards(self, n):
        return tuple(self.draw() for _ in range(n))


card_drawer = CardDrawer(
    CARD_POOL,
    buffer_size=RANDOM_BUFFER_SIZE,
    use_gpu=USE_GPU
)


def draw_card():
    return card_drawer.draw()


def draw_cards(n):
    return card_drawer.draw_cards(n)


# ============================================================
# ESTADO
# ============================================================

@dataclass(frozen=True)
class State:
    ace_life: int
    ace_hand: tuple
    marco_alive: bool
    ace_effect_used: bool
    attackers_used: tuple
    dons_remaining: int
    history: tuple = tuple()


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def epsilon_schedule(ep):
    progress = min(ep / EPSILON_DECAY_STEPS, 1.0)

    return (
        EPSILON_START * (1 - progress)
        + EPSILON_END * progress
    )


def action_key(action):
    normalized = {}

    for k, v in action.items():
        if isinstance(v, list):
            v = tuple(v)
        normalized[k] = v

    return tuple(sorted(normalized.items()))


def initial_state():
    return State(
        ace_life=ACE_INITIAL_LIFE,
        ace_hand=tuple(sorted(draw_cards(ACE_INITIAL_HAND_SIZE))),
        marco_alive=True,
        ace_effect_used=False,
        attackers_used=tuple(),
        dons_remaining=TOTAL_DON,
        history=tuple()
    )


# ============================================================
# AÇÕES DA BOA
# ============================================================

def boa_actions(state):
    actions = []

    for attacker_name, base_power in ATTACKERS.items():

        if attacker_name in state.attackers_used:
            continue

        for don in range(state.dons_remaining + 1):
            actions.append({
                "attacker": attacker_name,
                "base_power": base_power,
                "don": don,
                "attack_power": base_power + 1000 * don
            })

    return actions


# ============================================================
# AÇÕES DO ACE
# ============================================================

def ace_actions(state, boa_action):
    actions = []

    attack_power = boa_action["attack_power"]
    hand = list(state.ace_hand)
    n = len(hand)

    if state.marco_alive:
        actions.append({
            "type": "block"
        })

    if state.ace_life > 0:
        actions.append({
            "type": "take_hit"
        })

    for mask in range(1, 2 ** n):
        indices = []
        total_counter = 0

        for i in range(n):
            if mask & (1 << i):
                indices.append(i)
                total_counter += hand[i]

        defense = ACE_BASE_POWER + total_counter

        if defense > attack_power:
            actions.append({
                "type": "counter",
                "indices": tuple(indices),
                "leader_effect": False
            })

    if not state.ace_effect_used and n > 0:

        for discard_idx in range(n):
            defense = ACE_BASE_POWER + 2000

            if defense > attack_power:
                actions.append({
                    "type": "counter",
                    "indices": (discard_idx,),
                    "leader_effect": True
                })

        for discard_idx in range(n):
            remaining = [i for i in range(n) if i != discard_idx]

            for mask in range(2 ** len(remaining)):
                used = [discard_idx]
                total_counter = 2000

                for j, real_i in enumerate(remaining):
                    if mask & (1 << j):
                        used.append(real_i)
                        total_counter += hand[real_i]

                defense = ACE_BASE_POWER + total_counter

                if defense > attack_power:
                    actions.append({
                        "type": "counter",
                        "indices": tuple(sorted(used)),
                        "leader_effect": True
                    })

    if not actions:
        actions.append({
            "type": "cannot_defend"
        })

    return actions


# ============================================================
# TRANSIÇÃO
# ============================================================

def apply_transition(state, boa_action, ace_action):
    history = list(state.history)
    hand = list(state.ace_hand)

    attacker = boa_action["attacker"]
    don = boa_action["don"]
    atk = boa_action["attack_power"]

    history.append(
        f"Boa ataca com {attacker}, usando {don} DON, poder {atk}."
    )

    marco_alive = state.marco_alive
    ace_life = state.ace_life
    ace_effect_used = state.ace_effect_used

    attackers_used = tuple(list(state.attackers_used) + [attacker])
    dons_remaining = state.dons_remaining - don

    terminal = False
    reward = 0

    if ace_action["type"] == "block":
        marco_alive = False
        hand.extend(draw_cards(2))
        hand = sorted(hand)

        history.append(
            "Ace bloqueia com Marco. Marco sai de campo e Ace compra 2 cartas."
        )

    elif ace_action["type"] == "take_hit":
        ace_life -= 1
        hand.extend(draw_cards(1))
        hand = sorted(hand)

        history.append(
            "Ace toma o ataque, perde 1 vida e compra 1 carta."
        )

    elif ace_action["type"] == "counter":
        used = sorted(ace_action["indices"], reverse=True)

        for idx in used:
            if 0 <= idx < len(hand):
                hand.pop(idx)

        hand = sorted(hand)

        if ace_action["leader_effect"]:
            ace_effect_used = True
            history.append("Ace defende usando efeito do leader e/ou counters.")
        else:
            history.append("Ace defende usando counters.")

    elif ace_action["type"] == "cannot_defend":
        if ace_life > 0:
            ace_life -= 1
            hand.extend(draw_cards(1))
            hand = sorted(hand)

            history.append(
                "Ataque conecta. Ace perde a última vida e compra 1 carta."
            )
        else:
            terminal = True
            reward = 1

            history.append(
                "Ataque conecta com Ace em 0 vidas. Boa vence."
            )

    if len(attackers_used) >= len(ATTACKERS) and not terminal:
        terminal = True
        reward = 0
        history.append("Boa não conseguiu finalizar neste turno.")

    new_state = State(
        ace_life=ace_life,
        ace_hand=tuple(hand),
        marco_alive=marco_alive,
        ace_effect_used=ace_effect_used,
        attackers_used=attackers_used,
        dons_remaining=dons_remaining,
        history=tuple(history)
    )

    return new_state, reward, terminal


# ============================================================
# Q-TABLES
# ============================================================

boa_Q = defaultdict(float)
boa_N = defaultdict(int)

ace_Q = defaultdict(float)
ace_N = defaultdict(int)


# ============================================================
# POLÍTICAS
# ============================================================

def choose_boa_action(state, epsilon):
    actions = boa_actions(state)

    if random.random() < epsilon:
        return random.choice(actions)

    best_action = None
    best_value = -1e9

    for action in actions:
        key = (state, action_key(action))
        value = boa_Q[key]

        if value > best_value:
            best_value = value
            best_action = action

    return best_action if best_action is not None else random.choice(actions)


def choose_ace_action(state, boa_action, epsilon):
    actions = ace_actions(state, boa_action)

    if random.random() < epsilon:
        return random.choice(actions)

    best_action = None
    best_value = 1e9

    for action in actions:
        key = (state, action_key(boa_action), action_key(action))
        value = ace_Q[key]

        if value < best_value:
            best_value = value
            best_action = action

    return best_action if best_action is not None else random.choice(actions)


def update_q(Q, N, key, reward):
    N[key] += 1
    Q[key] += (reward - Q[key]) / N[key]


# ============================================================
# EPISÓDIO
# ============================================================

def run_episode(training=True, epsilon=0.1):
    state = initial_state()

    trajectory_boa = []
    trajectory_ace = []

    terminal = False
    final_reward = 0

    while not terminal:

        boa_action = choose_boa_action(state, epsilon)

        trajectory_boa.append((
            state,
            action_key(boa_action)
        ))

        ace_action = choose_ace_action(state, boa_action, epsilon)

        trajectory_ace.append((
            state,
            action_key(boa_action),
            action_key(ace_action)
        ))

        state, reward, terminal = apply_transition(
            state,
            boa_action,
            ace_action
        )

        final_reward = reward

    if training:
        for key in trajectory_boa:
            update_q(boa_Q, boa_N, key, final_reward)

        for key in trajectory_ace:
            update_q(ace_Q, ace_N, key, final_reward)

    return final_reward, state


# ============================================================
# TREINAMENTO
# ============================================================

rewards = []
moving_avg = []

running_sum = 0

for ep in range(EPISODES):

    epsilon = epsilon_schedule(ep)

    reward, _ = run_episode(
        training=True,
        epsilon=epsilon
    )

    rewards.append(reward)
    running_sum += reward

    if len(rewards) > WINDOW:
        running_sum -= rewards[-WINDOW - 1]

    avg = running_sum / min(len(rewards), WINDOW)
    moving_avg.append(avg)

    if ep % 10_000 == 0:
        print(
            f"EP {ep} | "
            f"epsilon={epsilon:.3f} | "
            f"winrate Boa={avg:.3f}"
        )

print("\nTreinamento finalizado.")
print(f"Winrate final médio da Boa: {moving_avg[-1]:.3f}")


# ============================================================
# GRÁFICO
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(moving_avg)
plt.title("Winrate médio da Boa ao longo do treinamento")
plt.xlabel("Episódios")
plt.ylabel("Média móvel da recompensa")
plt.grid(True)
plt.tight_layout()
plt.show(block=True)


# ============================================================
# ANÁLISE DA POLÍTICA DA BOA
# ============================================================

def summarize_boa_policy(top_n=20):
    action_stats = defaultdict(lambda: {"total": 0.0, "count": 0})

    for key, value in list(boa_Q.items()):
        try:
            _, action_tuple = key
            action_dict = dict(action_tuple)

            attacker = action_dict["attacker"]
            don = action_dict["don"]
            attack_power = action_dict["attack_power"]

            action_name = (attacker, don, attack_power)

            action_stats[action_name]["total"] += value
            action_stats[action_name]["count"] += 1

        except Exception:
            continue

    summary = []

    for action_name, stats in action_stats.items():
        avg_value = stats["total"] / stats["count"]
        summary.append((avg_value, stats["count"], action_name))

    summary.sort(reverse=True, key=lambda x: x[0])

    print("\n======================================")
    print("POLÍTICA APRENDIDA DA BOA")
    print("Melhores ações médias encontradas")
    print("======================================\n")

    for avg_value, count, action_name in summary[:top_n]:
        attacker, don, attack_power = action_name

        print(
            f"{attacker} com {don} DON "
            f"-> {attack_power} power | "
            f"valor médio={avg_value:.3f} | "
            f"ocorrências={count}"
        )


def summarize_first_actions(top_n=20):
    action_stats = []

    base_state = State(
        ace_life=ACE_INITIAL_LIFE,
        ace_hand=(0, 0),
        marco_alive=True,
        ace_effect_used=False,
        attackers_used=tuple(),
        dons_remaining=TOTAL_DON,
        history=tuple()
    )

    for action in boa_actions(base_state):
        key = (base_state, action_key(action))
        value = boa_Q[key]

        action_stats.append((
            value,
            action["attacker"],
            action["don"],
            action["attack_power"]
        ))

    action_stats.sort(reverse=True, key=lambda x: x[0])

    print("\n======================================")
    print("POLÍTICA DA BOA NO ESTADO INICIAL EXEMPLO")
    print("Mão do Ace assumida como [0, 0]")
    print("======================================\n")

    for value, attacker, don, power in action_stats[:top_n]:
        print(
            f"{attacker} com {don} DON "
            f"-> {power} power | Q={value:.3f}"
        )


# ============================================================
# DEBUG GREEDY
# ============================================================

def greedy_boa_action_debug(state):
    actions = boa_actions(state)

    best_action = None
    best_value = -1e9

    for action in actions:
        key = (state, action_key(action))
        value = boa_Q[key]

        if value > best_value:
            best_value = value
            best_action = action

    return best_action, best_value


def greedy_ace_action_debug(state, boa_action):
    actions = ace_actions(state, boa_action)

    best_action = None
    best_value = 1e9

    for action in actions:
        key = (state, action_key(boa_action), action_key(action))
        value = ace_Q[key]

        if value < best_value:
            best_value = value
            best_action = action

    return best_action, best_value


def describe_ace_action(action):
    if action["type"] == "block":
        return "Ace bloqueia com Marco"

    if action["type"] == "take_hit":
        return "Ace toma o ataque"

    if action["type"] == "cannot_defend":
        return "Ace não consegue defender"

    if action["type"] == "counter":
        if action["leader_effect"]:
            return f"Ace usa efeito do leader/counter com cartas {action['indices']}"
        else:
            return f"Ace usa counters das cartas {action['indices']}"

    return str(action)


# ============================================================
# 10 JOGOS GREEDY
# ============================================================

def run_greedy_game_verbose(game_number=1):
    state = initial_state()

    terminal = False
    reward = 0
    step = 1

    print("\n======================================")
    print(f"JOGO GREEDY #{game_number}")
    print("======================================")

    print(f"Mão inicial do Ace: {list(state.ace_hand)}")
    print(f"Vida inicial do Ace: {state.ace_life}")
    print(f"Marco vivo: {state.marco_alive}")
    print(f"DON inicial da Boa: {state.dons_remaining}\n")

    while not terminal:
        boa_action, boa_value = greedy_boa_action_debug(state)

        ace_action, ace_value = greedy_ace_action_debug(
            state,
            boa_action
        )

        print(f"--- Ataque {step} ---")

        print(
            f"Boa ataca com {boa_action['attacker']} "
            f"usando {boa_action['don']} DON "
            f"-> {boa_action['attack_power']} power "
            f"| Q Boa={boa_value:.3f}"
        )

        print(
            f"{describe_ace_action(ace_action)} "
            f"| Q Ace={ace_value:.3f}"
        )

        state, reward, terminal = apply_transition(
            state,
            boa_action,
            ace_action
        )

        print(
            f"Estado após ação: "
            f"vida Ace={state.ace_life}, "
            f"mão Ace={list(state.ace_hand)}, "
            f"Marco vivo={state.marco_alive}, "
            f"efeito Ace usado={state.ace_effect_used}, "
            f"DON restante={state.dons_remaining}"
        )

        print()
        step += 1

    if reward == 1:
        print("Resultado: Boa venceu.")
    else:
        print("Resultado: Ace sobreviveu.")

    return reward


def run_10_greedy_games():
    results = []

    for i in range(1, 11):
        reward = run_greedy_game_verbose(i)
        results.append(reward)

    wins = sum(results)

    print("\n======================================")
    print("RESUMO DOS 10 JOGOS GREEDY")
    print("======================================")
    print(f"Vitórias da Boa: {wins}/10")
    print(f"Sobrevivências do Ace: {10 - wins}/10")


# ============================================================
# CHAMADAS FINAIS
# ============================================================

summarize_boa_policy(top_n=20)
summarize_first_actions(top_n=20)
run_10_greedy_games()