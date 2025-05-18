import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim, num_agents):
        self.obs_buf = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size, num_agents, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.done_buf = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.val_buf = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.logp_buf = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.ptr = 0
        self.max_size = buffer_size

    def store(self, obs, act, rew, done, val, logp):
        if self.ptr >= self.max_size:
            return
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act.reshape(-1, 1)
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def get(self):
        return {
        "obs": self.obs_buf[:self.ptr],
        "act": self.act_buf[:self.ptr],
        "rew": self.rew_buf[:self.ptr],
        "done": self.done_buf[:self.ptr],
        "logp": self.logp_buf[:self.ptr],
        "val": self.val_buf[:self.ptr],
    }

    def reset(self):
        self.ptr = 0
