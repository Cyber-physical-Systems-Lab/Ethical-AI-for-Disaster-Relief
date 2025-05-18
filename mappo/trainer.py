import torch
import torch.nn.functional as F

class MAPPOTrainer:
    def __init__(self, actor_critic, buffer, lr, clip_eps, vf_coef, entropy_coef, max_grad_norm, device):
        self.actor_critic = actor_critic
        self.buffer = buffer
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

    def update(self, advantages, returns):
        data = self.buffer.get()
        obs = torch.tensor(data['obs'], dtype=torch.float32, device=self.device)
        act = torch.tensor(data['act'].squeeze(-1), dtype=torch.long, device=self.device)
        logp_old = torch.tensor(data['logp'], dtype=torch.float32, device=self.device)
        val = torch.tensor(data['val'], dtype=torch.float32, device=self.device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Flatten (B, N, ...) → (B * N, ...)
        B, N = obs.shape[:2]
        obs = obs.reshape(B * N, -1)
        act = act.reshape(B * N)
        logp_old = logp_old.reshape(B * N)
        val = val.reshape(B * N)
        adv = adv.reshape(B * N)
        ret = ret.reshape(B * N)

        # Normalize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Forward pass
        logits, values, entropy = self.actor_critic.forward(obs, return_entropy=True)
        dist = F.softmax(logits, dim=-1)
        dist_entropy = -torch.sum(dist * torch.log(dist + 1e-8), dim=-1)

        new_logp = F.log_softmax(logits, dim=-1).gather(1, act.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(new_logp - logp_old)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(), ret)
        entropy_bonus = dist_entropy.mean()

        loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
