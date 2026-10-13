"""Hyperparameters for PPO, as a single dataclass that the CLI is generated from."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class PPOConfig:
    # --- experiment ---
    env_id: str = "HalfCheetah-v5"
    seed: int = 1
    total_timesteps: int = 1_000_000
    log_dir: str = "runs/default"
    eval_episodes: int = 10

    # --- rollout ---
    num_envs: int = 1
    num_steps: int = 2048  # steps per env per rollout; batch = num_envs * num_steps

    # --- optimisation ---
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    num_minibatches: int = 32
    max_grad_norm: float = 0.5
    target_kl: float = 0.0  # 0 disables early stopping on KL

    # --- PPO objective ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    norm_adv: bool = True

    # --- implementation details that matter (ablated in the study) ---
    norm_obs: bool = True
    norm_reward: bool = True
    hidden_size: int = 64

    # --- domain randomisation (training-time perturbations; all 0 = nominal physics) ---
    dr_strength: float = 0.0  # physics scales ~ logU[1/(1+s), 1+s]; 0 disables
    dr_params: str = "friction,mass,gain"  # comma-separated subset of friction, mass, gain
    train_obs_noise: float = 0.0  # max sensor-noise level (fraction of obs std), sampled U(0, max) per episode
    train_action_delay: int = 0  # max actuation delay in steps, sampled uniformly from {0..max} per episode

    # --- misc ---
    torch_threads: int = 1

    def randomization(self):
        from .perturb import DomainRandomization

        return DomainRandomization(
            strength=self.dr_strength,
            params=tuple(p for p in self.dr_params.split(",") if p),
            obs_noise_max=self.train_obs_noise,
            action_delay_max=self.train_action_delay,
        )

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PPOConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
