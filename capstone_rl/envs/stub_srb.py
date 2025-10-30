import numpy as np
import gymnasium as gym
from gymnasium import spaces

class StubSRBEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, obs_dim=8, act_dim=2, horizon=200):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        self.t = 0
        self.obs = np.zeros(self.obs_dim, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.obs = np.zeros(self.obs_dim, dtype=np.float32)
        return self.obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta = np.concatenate([action, np.zeros(self.obs_dim - self.act_dim)])
        self.obs = self.obs + delta * 0.1 + 0.01 * np.random.randn(self.obs_dim).astype(np.float32)
        reward = -np.linalg.norm(self.obs)
        self.t += 1
        terminated = self.t >= self.horizon
        truncated = False
        return self.obs, reward, terminated, truncated, {}

    def render(self, mode="rgb_array"):
        return None

    def close(self):
        pass