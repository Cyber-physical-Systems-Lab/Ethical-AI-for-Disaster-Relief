import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(64, act_dim)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, obs, return_entropy=False):
        x = self.shared(obs)
        logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)

        if return_entropy:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean() 
            return logits, value, entropy

        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    # 🟡 Confidence: probability of the selected action
        confidence = probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, value, log_prob, confidence


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_head(x)
