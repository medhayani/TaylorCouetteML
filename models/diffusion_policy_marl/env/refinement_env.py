"""Cooperative MARL environment for switch-window refinement.

Each episode = one window of T points. State = (obs_seq, static_vec, T_a^pred).
Joint action = corrections from 5 agents combined into delta of length T.
Reward = -|| T_a^true - (T_a^pred + delta) ||_w^2 + beta_phys * Phi(delta).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _HAS_GYM = False


def _norm_grad(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x)


class RefinementEnv:
    """Lightweight (non-gym) version of the env. Keep gym wrapper optional."""

    def __init__(
        self,
        windows: Dict[str, np.ndarray],
        window_length: int = 49,
        reward_clip: float = 12.0,
        weight_local: float = 1.4,
        weight_slope: float = 0.75,
        beta_phys: float = 0.1,
    ):
        # windows is a dict of arrays produced by Step F.
        for k in ["obs_seq", "static_vec", "y_true", "y_pred"]:
            if k not in windows:
                raise ValueError(f"Missing key '{k}' in windows.")
        self.windows = windows
        self.N = windows["y_true"].shape[0]
        self.T = window_length
        self.reward_clip = reward_clip
        self.weight_local = weight_local
        self.weight_slope = weight_slope
        self.beta_phys = beta_phys
        self._idx = 0

    def reset(self, idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        self._idx = int(idx) if idx is not None else int(np.random.randint(self.N))
        return self._observation()

    def _observation(self) -> Dict[str, np.ndarray]:
        i = self._idx
        return {
            "obs_seq": self.windows["obs_seq"][i],
            "static_vec": self.windows["static_vec"][i],
            "y_pred": self.windows["y_pred"][i],
        }

    def step(self, joint_correction: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        i = self._idx
        y_true = self.windows["y_true"][i]
        y_pred = self.windows["y_pred"][i]
        delta = np.asarray(joint_correction, dtype=np.float32).reshape(-1)[: y_pred.shape[0]]
        y_corr = y_pred + delta

        # weighted L2 (focus on center)
        T = y_true.shape[0]
        center = T // 2
        w = np.ones(T, dtype=np.float32)
        w[max(0, center - 5): center + 6] *= self.weight_local
        err = (y_true - y_corr) ** 2
        loss_val = float((w * err).mean())

        slope_err = (_norm_grad(y_true) - _norm_grad(y_corr)) ** 2
        loss_slope = float(slope_err.mean())

        phys_pen = float(np.mean(np.abs(_norm_grad(_norm_grad(delta)))))

        reward = -loss_val - self.weight_slope * loss_slope - self.beta_phys * phys_pen
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        info = {"loss_val": loss_val, "loss_slope": loss_slope, "phys_pen": phys_pen}
        return self._observation(), reward, True, info


if _HAS_GYM:

    class GymRefinementEnv(gym.Env):
        """Single-agent gym wrapper (joint action space). For SB3 baselines."""

        metadata = {"render_modes": []}

        def __init__(self, windows: Dict[str, np.ndarray], window_length: int = 49,
                     joint_action_dim: int = 49):
            super().__init__()
            self._inner = RefinementEnv(windows, window_length=window_length)
            obs_seq_dim = windows["obs_seq"].shape[1] * windows["obs_seq"].shape[2]
            static_dim = windows["static_vec"].shape[1]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_seq_dim + static_dim + window_length,), dtype=np.float32,
            )
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(joint_action_dim,), dtype=np.float32)

        def _flat_obs(self, obs):
            return np.concatenate(
                [obs["obs_seq"].reshape(-1), obs["static_vec"].reshape(-1),
                 obs["y_pred"].reshape(-1)]
            ).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs = self._inner.reset()
            return self._flat_obs(obs), {}

        def step(self, action):
            obs, reward, done, info = self._inner.step(action)
            return self._flat_obs(obs), reward, done, False, info
else:
    GymRefinementEnv = None  # type: ignore
