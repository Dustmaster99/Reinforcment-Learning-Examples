"""
PyTorch CUDA version for Monte Carlo Blackjack action-value evolution plots.

This version saves all plots separately, without overlapping figures.

For each checkpoint at 25%, 50%, 75%, and 100%, it generates:

1) Individual heatmaps:
   - Q(s,HIT), usable ace
   - Q(s,STAND), usable ace
   - Q(s,HIT), no usable ace
   - Q(s,STAND), no usable ace
   - action advantage, usable ace
   - action advantage, no usable ace
   - greedy policy, usable ace
   - greedy policy, no usable ace

2) Individual 3D surface plots:
   - Q(s,HIT), usable ace
   - Q(s,STAND), usable ace
   - Q(s,HIT), no usable ace
   - Q(s,STAND), no usable ace
   - action advantage, usable ace
   - action advantage, no usable ace

Install:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install "numpy==1.26.4" matplotlib seaborn ipython
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
import torch
import os

os.chdir(r"C:\Kegle_Jojo\RL-examples")


SCRIPT_START_TIME = time.perf_counter()

OUTPUT_DIR = "Montecarlo_blackjack"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_HIT = 0
ACTION_STAND = 1


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível no PyTorch.\n"
            "Instale o PyTorch com CUDA, por exemplo:\n"
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        )

    device = torch.device("cuda")
    print("PyTorch:", torch.__version__)
    print("CUDA disponível no PyTorch:", torch.cuda.is_available())
    print("Versão CUDA usada pelo PyTorch:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    return device


def torch_get_card(shape, device):
    card = torch.randint(1, 14, shape, device=device, dtype=torch.int16)
    return torch.clamp(card, max=10)


def torch_card_value(card):
    return torch.where(
        card == 1,
        torch.tensor(11, device=card.device, dtype=torch.int16),
        card,
    )


def safe_name(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
    )


def simulate_batch_and_update_q(q_sum, q_count, n, device):
    """
    Simulates n episodes with exploring starts and updates Q accumulators.
    """
    visited_sa = torch.zeros((n, 2, 10, 10, 2), device=device, dtype=torch.bool)

    # Exploring starts
    initial_usable = torch.randint(0, 2, (n,), device=device, dtype=torch.int16)
    player_sum = torch.randint(12, 22, (n,), device=device, dtype=torch.int16)
    dealer_card1 = torch.randint(1, 11, (n,), device=device, dtype=torch.int16)
    initial_action = torch.randint(0, 2, (n,), device=device, dtype=torch.int16)

    ace_count_player = initial_usable.clone()

    dealer_card2 = torch_get_card((n,), device)
    dealer_sum = torch_card_value(dealer_card1) + torch_card_value(dealer_card2)
    ace_count_dealer = (
        (dealer_card1 == 1).to(torch.int16)
        + (dealer_card2 == 1).to(torch.int16)
    )

    adjust = (dealer_sum > 21) & (ace_count_dealer > 0)
    while bool(adjust.any()):
        dealer_sum[adjust] -= 10
        ace_count_dealer[adjust] -= 1
        adjust = (dealer_sum > 21) & (ace_count_dealer > 0)

    reward = torch.zeros(n, device=device, dtype=torch.float32)
    active_player = torch.ones(n, device=device, dtype=torch.bool)
    player_busted = torch.zeros(n, device=device, dtype=torch.bool)
    first_action_pending = torch.ones(n, device=device, dtype=torch.bool)

    # Player turn
    while bool(active_player.any()):
        active_idx = torch.where(active_player)[0]

        ps_idx = (player_sum[active_idx] - 12).long()
        dc_idx = (dealer_card1[active_idx] - 1).long()
        usable = (ace_count_player[active_idx] > 0).long()

        actions = torch.empty(active_idx.numel(), device=device, dtype=torch.int16)

        pending = first_action_pending[active_idx]
        if bool(pending.any()):
            actions[pending] = initial_action[active_idx[pending]]
            first_action_pending[active_idx[pending]] = False

        not_pending = ~pending
        if bool(not_pending.any()):
            actions[not_pending] = torch.where(
                player_sum[active_idx[not_pending]] >= 20,
                torch.tensor(ACTION_STAND, device=device, dtype=torch.int16),
                torch.tensor(ACTION_HIT, device=device, dtype=torch.int16),
            )

        valid = (ps_idx >= 0) & (ps_idx < 10) & (dc_idx >= 0) & (dc_idx < 10)
        if bool(valid.any()):
            valid_idx = active_idx[valid]
            visited_sa[
                valid_idx,
                usable[valid],
                ps_idx[valid],
                dc_idx[valid],
                actions[valid].long(),
            ] = True

        stand = actions == ACTION_STAND
        if bool(stand.any()):
            active_player[active_idx[stand]] = False

        hit_idx = active_idx[actions == ACTION_HIT]
        if hit_idx.numel() == 0:
            continue

        card = torch_get_card((hit_idx.numel(),), device)
        ace_count_player[hit_idx] += (card == 1).to(torch.int16)
        player_sum[hit_idx] += torch_card_value(card)

        adjust = (player_sum > 21) & (ace_count_player > 0) & active_player
        while bool(adjust.any()):
            player_sum[adjust] -= 10
            ace_count_player[adjust] -= 1
            adjust = (player_sum > 21) & (ace_count_player > 0) & active_player

        bust = (player_sum > 21) & active_player
        if bool(bust.any()):
            reward[bust] = -1.0
            player_busted[bust] = True
            active_player[bust] = False

    # Dealer turn
    active_dealer = ~player_busted

    while bool((active_dealer & (dealer_sum < 17)).any()):
        hit = active_dealer & (dealer_sum < 17)
        hit_idx = torch.where(hit)[0]

        card = torch_get_card((hit_idx.numel(),), device)
        ace_count_dealer[hit_idx] += (card == 1).to(torch.int16)
        dealer_sum[hit_idx] += torch_card_value(card)

        adjust = (dealer_sum > 21) & (ace_count_dealer > 0) & active_dealer
        while bool(adjust.any()):
            dealer_sum[adjust] -= 10
            ace_count_dealer[adjust] -= 1
            adjust = (dealer_sum > 21) & (ace_count_dealer > 0) & active_dealer

        dealer_bust = (dealer_sum > 21) & active_dealer
        if bool(dealer_bust.any()):
            reward[dealer_bust] = 1.0
            active_dealer[dealer_bust] = False

    compare = active_dealer & (dealer_sum <= 21)
    if bool(compare.any()):
        reward[compare & (player_sum > dealer_sum)] = 1.0
        reward[compare & (player_sum == dealer_sum)] = 0.0
        reward[compare & (player_sum < dealer_sum)] = -1.0

    # First-visit action-value update
    nz = visited_sa.nonzero(as_tuple=False)

    if nz.numel() > 0:
        episode_idx = nz[:, 0]
        usable = nz[:, 1]
        ps_idx = nz[:, 2]
        dc_idx = nz[:, 3]
        action = nz[:, 4]

        flat_sa = usable * 200 + ps_idx * 20 + dc_idx * 2 + action

        reward_weights = reward[episode_idx]
        count_weights = torch.ones_like(reward_weights)

        sa_reward_sum = torch.bincount(flat_sa, weights=reward_weights, minlength=400)
        sa_count_sum = torch.bincount(flat_sa, weights=count_weights, minlength=400)

        q_sum += sa_reward_sum.reshape(2, 10, 10, 2)
        q_count += sa_count_sum.reshape(2, 10, 10, 2)


def save_single_heatmap(data, title, filename, episodes_done, total_episodes, vmin=-1, vmax=1, cmap="YlGnBu", center=None, annot=None):
    fig, ax = plt.subplots(figsize=(14, 10))

    kwargs = {
        "data": np.flipud(data),
        "cmap": cmap,
        "ax": ax,
        "xticklabels": range(1, 11),
        "yticklabels": list(reversed(range(12, 22))),
        "cbar_kws": {"label": f"value after {episodes_done:,}/{total_episodes:,} episodes"},
    }

    if center is not None:
        kwargs["center"] = center
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if annot is not None:
        kwargs["annot"] = np.flipud(annot)
        kwargs["fmt"] = ""
        kwargs["cbar"] = False

    sns.heatmap(**kwargs)

    ax.set_xlabel("dealer showing", fontsize=16)
    ax.set_ylabel("player sum", fontsize=16)
    ax.set_title(f"{title}\nEpisodes: {episodes_done:,}/{total_episodes:,}", fontsize=18)

    path = f"{OUTPUT_DIR}/{filename}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_single_surface(data, title, filename, episodes_done, total_episodes, zlabel="Value"):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    x_range = np.arange(1, 11)
    y_range = np.arange(12, 22)
    X, Y = np.meshgrid(x_range, y_range)
    Z = data

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor="none")
    ax.set_xlabel("Dealer Showing", fontsize=12)
    ax.set_ylabel("Player Sum", fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(f"{title}\nEpisodes: {episodes_done:,}/{total_episodes:,}", fontsize=14)
    ax.set_zlim(-1, 1)

    path = f"{OUTPUT_DIR}/{filename}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_all_separate_plots(q_values, episodes_done, total_episodes):
    generated = []

    q_hit_no_usable = q_values[0, :, :, ACTION_HIT]
    q_stand_no_usable = q_values[0, :, :, ACTION_STAND]
    q_hit_usable = q_values[1, :, :, ACTION_HIT]
    q_stand_usable = q_values[1, :, :, ACTION_STAND]

    advantage_no_usable = q_stand_no_usable - q_hit_no_usable
    advantage_usable = q_stand_usable - q_hit_usable

    policy_no_usable = np.argmax(q_values[0], axis=-1)
    policy_usable = np.argmax(q_values[1], axis=-1)

    items = [
        ("Q(s,HIT) - Usable Ace", q_hit_usable, "q_hit_usable"),
        ("Q(s,STAND) - Usable Ace", q_stand_usable, "q_stand_usable"),
        ("Q(s,HIT) - No Usable Ace", q_hit_no_usable, "q_hit_no_usable"),
        ("Q(s,STAND) - No Usable Ace", q_stand_no_usable, "q_stand_no_usable"),
    ]

    for title, data, name in items:
        generated.append(
            save_single_heatmap(
                data=data,
                title=title,
                filename=f"heatmap_{name}_{episodes_done}",
                episodes_done=episodes_done,
                total_episodes=total_episodes,
                vmin=-1,
                vmax=1,
                cmap="YlGnBu",
            )
        )
        generated.append(
            save_single_surface(
                data=data,
                title=title,
                filename=f"surface_{name}_{episodes_done}",
                episodes_done=episodes_done,
                total_episodes=total_episodes,
                zlabel="Q value",
            )
        )

    # Advantage plots
    advantage_items = [
        ("Action advantage Q(STAND)-Q(HIT) - Usable Ace", advantage_usable, "advantage_usable"),
        ("Action advantage Q(STAND)-Q(HIT) - No Usable Ace", advantage_no_usable, "advantage_no_usable"),
    ]

    for title, data, name in advantage_items:
        generated.append(
            save_single_heatmap(
                data=data,
                title=title,
                filename=f"heatmap_{name}_{episodes_done}",
                episodes_done=episodes_done,
                total_episodes=total_episodes,
                cmap="coolwarm",
                center=0,
            )
        )
        generated.append(
            save_single_surface(
                data=data,
                title=title,
                filename=f"surface_{name}_{episodes_done}",
                episodes_done=episodes_done,
                total_episodes=total_episodes,
                zlabel="Q(STAND)-Q(HIT)",
            )
        )

    # Policy plots, only heatmaps because policy is discrete H/S.
    generated.append(
        save_single_heatmap(
            data=policy_usable,
            title="Greedy policy from Q - Usable Ace",
            filename=f"policy_usable_{episodes_done}",
            episodes_done=episodes_done,
            total_episodes=total_episodes,
            vmin=0,
            vmax=1,
            cmap="YlGnBu",
            annot=np.where(policy_usable == ACTION_STAND, "S", "H"),
        )
    )

    generated.append(
        save_single_heatmap(
            data=policy_no_usable,
            title="Greedy policy from Q - No Usable Ace",
            filename=f"policy_no_usable_{episodes_done}",
            episodes_done=episodes_done,
            total_episodes=total_episodes,
            vmin=0,
            vmax=1,
            cmap="YlGnBu",
            annot=np.where(policy_no_usable == ACTION_STAND, "S", "H"),
        )
    )

    return generated


def monte_carlo_action_values_with_checkpoints_torch_gpu(
    total_episodes=500_000,
    seed=7,
    batch_size=200_000,
):
    """
    Runs MC ES and saves separate heatmaps and 3D surfaces at 25%, 50%, 75%, and 100%.
    """
    device = get_device()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q_sum = torch.zeros((2, 10, 10, 2), device=device, dtype=torch.float32)
    q_count = torch.ones((2, 10, 10, 2), device=device, dtype=torch.float32)

    checkpoints = [
        total_episodes // 4,
        total_episodes // 2,
        (3 * total_episodes) // 4,
        total_episodes,
    ]
    checkpoints = sorted(set(checkpoints))

    generated_paths = []
    processed = 0

    for checkpoint in checkpoints:
        episodes_to_run = checkpoint - processed

        while episodes_to_run > 0:
            n = min(batch_size, episodes_to_run)
            simulate_batch_and_update_q(q_sum, q_count, n, device)
            processed += n
            episodes_to_run -= n
            print(f"Processados {processed:,}/{total_episodes:,} episódios")

        torch.cuda.synchronize()
        q_values = (q_sum / q_count).detach().cpu().numpy()

        print(f"Gerando gráficos separados para checkpoint {processed:,} episódios...")
        generated_paths.extend(save_all_separate_plots(q_values, processed, total_episodes))

    return generated_paths


if __name__ == "__main__":
    TOTAL_EPISODES = 500_000

    paths = monte_carlo_action_values_with_checkpoints_torch_gpu(
        total_episodes=TOTAL_EPISODES,
        seed=7,
        batch_size=200_000,
    )

    print("\nArquivos gerados:")
    for p in paths:
        print(p)

    # Display a small final sample in Spyder/Jupyter.
    display(Image(filename=f"{OUTPUT_DIR}/heatmap_q_hit_usable_{TOTAL_EPISODES}.png"))
    display(Image(filename=f"{OUTPUT_DIR}/surface_q_hit_usable_{TOTAL_EPISODES}.png"))
    display(Image(filename=f"{OUTPUT_DIR}/heatmap_advantage_usable_{TOTAL_EPISODES}.png"))
    display(Image(filename=f"{OUTPUT_DIR}/surface_advantage_usable_{TOTAL_EPISODES}.png"))
    display(Image(filename=f"{OUTPUT_DIR}/policy_usable_{TOTAL_EPISODES}.png"))

    SCRIPT_END_TIME = time.perf_counter()
    total_seconds = SCRIPT_END_TIME - SCRIPT_START_TIME

    print(f"\nTempo total de execução: {total_seconds:.3f} segundos")
    print(f"Tempo total de execução: {total_seconds / 60:.3f} minutos")
