import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env.ethical_env import EthicalRescueEnv
from mappo.network import Actor
from mappo.buffer import ReplayBuffer
from mappo.trainer import MAPPOTrainer
from mappo.utils import compute_gae

# ---------------------------- Config ----------------------------
NUM_EPISODES = 1000
STEPS_PER_EPISODE = 20
GAMMA = 0.99
LAMBDA = 0.95
LR = 5e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NUM_RUNS = 1
SEED_RANGE = (1, 10000)
RANDOM_SEEDS = random.sample(range(*SEED_RANGE), NUM_RUNS)
SAVE_PATH = "model"
LOGS_PATH = os.path.join(SAVE_PATH, "logs")

# Convergence parameters
CONVERGENCE_WINDOW = 100
CONVERGENCE_THRESHOLD = 0.05


# ---------------------------- Helper Functions ----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def has_converged(rewards, window=CONVERGENCE_WINDOW, std_threshold=0.05):
    if len(rewards) < window:
        return False
    recent = rewards[-window:]
    return np.std(recent) < std_threshold


def evaluate_alignment(env, actor_critic, device):
    obs = env.reset()
    num_agents = env.num_agents
    obs_tensor = torch.tensor(obs.reshape(num_agents, -1), dtype=torch.float32, device=device)

    with torch.no_grad():
        actions, _, _ = actor_critic.act(obs_tensor)
        logits = actor_critic.actor_head(actor_critic.shared(obs_tensor))
        probs = torch.softmax(logits, dim=-1)
        confidences = probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach().cpu().numpy()

    correct = 0
    for agent_id in range(num_agents):
        victims = env.victims[agent_id]
        transparency = confidences[agent_id]
        ethical_scores = []

        for idx, (_, features) in enumerate(victims):
            severity = features[0]
            urgency = features[1]
            distance = features[4]
            reliability = env._calculate_reliability(agent_id, severity, urgency, idx)
            fairness = (severity + urgency + (1 - distance)) / 3
            ethical_score = 0.4 * fairness + 0.3 * transparency + 0.3 * reliability
            ethical_scores.append(ethical_score)

        selected = actions[agent_id].item()
        max_score = max(ethical_scores)
        if abs(ethical_scores[selected] - max_score) < 1e-6:
            correct += 1

    alignment_rate = 100 * correct / num_agents
    return round(alignment_rate, 2)


# ---------------------------- Training Loop ----------------------------
def run_training(seed):
    print(f"\n🚀 Training with seed {seed}")
    set_seed(seed)

    run_path = os.path.join(SAVE_PATH, f"seed")
    os.makedirs(run_path, exist_ok=True)

    env = EthicalRescueEnv(max_steps=STEPS_PER_EPISODE, victims_per_agent=3)
    obs = env.reset()
    obs_size = obs.shape[2] * env.victims_per_agent
    action_size = env.action_space.n
    num_agents = env.num_agents

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = Actor(obs_size, action_size).to(device)
    buffer = ReplayBuffer(STEPS_PER_EPISODE, obs_size, 1, num_agents)
    trainer = MAPPOTrainer(actor_critic, buffer, LR, CLIP_EPS, VF_COEF, ENTROPY_COEF, MAX_GRAD_NORM, device)

    episode_rewards = []
    reward_log = []

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        ep_rewards = np.zeros(num_agents)
        step_logs_all = []

        for step in range(STEPS_PER_EPISODE):
            obs_tensor = torch.tensor(obs.reshape(num_agents, -1), dtype=torch.float32, device=device)
            actions, values, log_probs = actor_critic.act(obs_tensor)

            logits = actor_critic.actor_head(actor_critic.shared(obs_tensor))
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach().cpu().numpy()

            next_obs, rewards, dones, _ = env.step(actions.cpu().numpy(), confidences)

            buffer.store(
                obs.reshape(num_agents, -1),
                actions.cpu().numpy().reshape(num_agents, 1),
                rewards,
                dones,
                values.detach().cpu().numpy(),
                log_probs.detach().cpu().numpy()
            )

            obs = next_obs
            ep_rewards += rewards
            step_logs_all.extend(env.step_logs)

        _, next_values = actor_critic.forward(torch.tensor(obs.reshape(num_agents, -1), dtype=torch.float32, device=device))
        advantages, returns = compute_gae(
            buffer.rew_buf, buffer.done_buf, buffer.val_buf,
            next_values.detach().cpu().numpy(), GAMMA, LAMBDA
        )
        trainer.update(advantages, returns)
        buffer.reset()

        avg_reward = np.mean(ep_rewards)
        episode_rewards.append(avg_reward)
        reward_log.append({"Episode": episode + 1, "Avg Reward": avg_reward})
        print(f"Seed {seed} | Episode {episode+1} | Avg Reward: {avg_reward:.2f}")

        pd.DataFrame(step_logs_all).to_csv(os.path.join(run_path, f"episode_{episode+1}.csv"), index=False)

        # 🔍 Evaluate ethical alignment
        # if (episode + 1) % 50 == 0:
        #     alignment = evaluate_alignment(env, actor_critic, device)
        #     print(f"[Eval] Episode {episode + 1} | Ethical Alignment Rate: {alignment:.2f}%")

        # ✅ Early stop if converged
        if has_converged(episode_rewards):
            print(f"✅ Converged at episode {episode + 1} with avg reward ~ {avg_reward:.2f}")
            break

    # Save model and logs
    torch.save(actor_critic.state_dict(), os.path.join(run_path, "final_model.pth"))
    np.save(os.path.join(run_path, "episode_rewards.npy"), episode_rewards)
    pd.DataFrame(reward_log).to_csv(os.path.join(run_path, "rewards_log.csv"), index=False)

    # Plot
    plt.figure()
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label=f"Seed {seed}")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title(f"Training Progress (Seed {seed})")
    plt.grid(True)
    plt.savefig(os.path.join(run_path, "reward_plot.png"))
    plt.close()

    return pd.DataFrame(reward_log)


# ---------------------------- Run All Seeds ----------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

all_dfs = []
for seed in RANDOM_SEEDS:
    df = run_training(seed)
    df["Seed"] = seed
    all_dfs.append(df)

# Combined CSV
combined_df = pd.concat(all_dfs)
combined_df.to_csv(os.path.join(LOGS_PATH, "combined_rewards.csv"), index=False)

# Summary CSV
summary_df = combined_df.groupby("Seed")["Avg Reward"].mean().reset_index()
summary_df.columns = ["Seed", "Average Reward Across Episodes"]
summary_df.to_csv(os.path.join(LOGS_PATH, "avg_reward_summary.csv"), index=False)

# Combined plot
plt.figure()
for seed in RANDOM_SEEDS:
    seed_df = combined_df[combined_df["Seed"] == seed]
    plt.plot(seed_df["Episode"], seed_df["Avg Reward"], label=f"Seed {seed}")
plt.xlabel("Episode")
plt.ylabel("Avg Reward")
plt.title("Training Progress Across Seeds")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(LOGS_PATH, "combined_plot.png"))
plt.close()

print("\n✅ All seed runs completed and logged.")
