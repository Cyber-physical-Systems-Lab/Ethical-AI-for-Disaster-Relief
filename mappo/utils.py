import numpy as np

def compute_gae(rewards, dones, values, next_values, gamma=0.99, lam=0.95):
    """
    rewards: [T, N]         - rewards at each step
    dones: [T, N]           - done flags at each step
    values: [T, N]          - estimated state values
    next_values: [N]        - value of final state
    returns:
        advantages: [T, N]
        returns: [T, N]
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_adv = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_values
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv

    returns = advantages + values
    return advantages, returns
