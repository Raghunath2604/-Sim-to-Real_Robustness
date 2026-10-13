"""Physics and sensing perturbations for MuJoCo gymnasium environments.

Five perturbation axes are supported:

* ``friction``     multiplies every geom's sliding friction
* ``mass``         multiplies every body's mass *and* inertia
* ``gain``         multiplies every actuator's gear ratio (motor strength)
* ``obs_noise``    Gaussian sensor noise, as a fraction of each observation dimension's typical std
* ``action_delay`` actuation latency, in control steps

A ``PerturbationSpec`` is one fixed setting; ``DomainRandomization`` samples specs at every episode reset.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace

import gymnasium as gym
import mujoco
import numpy as np

PHYSICS_AXES = ("friction", "mass", "gain")
AXES = (*PHYSICS_AXES, "obs_noise", "action_delay")


@dataclass(frozen=True)
class PerturbationSpec:
    friction: float = 1.0
    mass: float = 1.0
    gain: float = 1.0
    obs_noise: float = 0.0
    action_delay: int = 0

    def is_nominal(self) -> bool:
        return self == PerturbationSpec()

    def to_dict(self) -> dict:
        return asdict(self)


def spec_for(axis: str, value: float) -> PerturbationSpec:
    """A spec that perturbs exactly one axis, leaving the others nominal."""
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
    return replace(PerturbationSpec(), **{axis: int(value) if axis == "action_delay" else float(value)})


@dataclass
class DomainRandomization:
    """Samples a PerturbationSpec per episode.

    Physical scales are log-uniform in [1/(1+strength), 1+strength], so a strength of 0.5 means
    "between x0.67 and x1.5", symmetric on a log scale. ``strength == 0`` disables physics randomisation.
    """

    strength: float = 0.0
    params: tuple[str, ...] = PHYSICS_AXES
    obs_noise_max: float = 0.0
    action_delay_max: int = 0

    def enabled(self) -> bool:
        return self.strength > 0 or self.obs_noise_max > 0 or self.action_delay_max > 0

    def scale_range(self) -> tuple[float, float]:
        return 1.0 / (1.0 + self.strength), 1.0 + self.strength

    def sample(self, rng: np.random.Generator) -> PerturbationSpec:
        kw: dict = {}
        if self.strength > 0:
            lo, hi = np.log(self.scale_range()[0]), np.log(self.scale_range()[1])
            for p in self.params:
                if p not in PHYSICS_AXES:
                    raise ValueError(f"unknown physics param {p!r}")
                kw[p] = float(np.exp(rng.uniform(lo, hi)))
        if self.obs_noise_max > 0:
            kw["obs_noise"] = float(rng.uniform(0.0, self.obs_noise_max))
        if self.action_delay_max > 0:
            kw["action_delay"] = int(rng.integers(0, self.action_delay_max + 1))
        return PerturbationSpec(**kw)


_OBS_SCALE_CACHE: dict[str, np.ndarray] = {}


def estimate_obs_scale(env_id: str, steps: int = 2000, seed: int = 0) -> np.ndarray:
    """Per-dimension std of observations under a uniform-random policy (used to size sensor noise)."""
    if env_id not in _OBS_SCALE_CACHE:
        env = gym.make(env_id)
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)
        buf = [obs]
        for _ in range(steps):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            buf.append(obs)
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        _OBS_SCALE_CACHE[env_id] = np.maximum(np.std(np.asarray(buf), axis=0), 1e-3)
    return _OBS_SCALE_CACHE[env_id]


class PerturbedEnv(gym.Wrapper):
    """Applies a (fixed or randomised) perturbation to a MuJoCo gymnasium environment.

    Physics are changed in-place on the underlying ``MjModel`` just *before* each reset, always relative to the
    nominal values captured at construction, so perturbations never accumulate.
    """

    def __init__(
        self,
        env: gym.Env,
        fixed: PerturbationSpec | None = None,
        randomization: DomainRandomization | None = None,
        obs_scale: np.ndarray | None = None,
        seed: int = 0,
    ):
        super().__init__(env)
        if fixed is not None and randomization is not None and randomization.enabled():
            raise ValueError("pass either `fixed` or `randomization`, not both")
        self.fixed = fixed
        self.randomization = randomization
        self.rng = np.random.default_rng(seed)
        self._model: mujoco.MjModel = env.unwrapped.model
        self._data: mujoco.MjData = env.unwrapped.data
        m = self._model
        self._nominal = {
            "friction": m.geom_friction.copy(),
            "mass": m.body_mass.copy(),
            "inertia": m.body_inertia.copy(),
            "gear": m.actuator_gear.copy(),
        }
        self._obs_scale = obs_scale
        self._buffer: deque[np.ndarray] = deque()
        self.current = PerturbationSpec()

    # ---- physics ----
    def apply_spec(self, spec: PerturbationSpec) -> None:
        m, n = self._model, self._nominal
        m.geom_friction[:] = n["friction"]
        m.geom_friction[:, 0] = n["friction"][:, 0] * spec.friction  # sliding friction only
        m.body_mass[:] = n["mass"] * spec.mass
        m.body_inertia[:] = n["inertia"] * spec.mass
        m.actuator_gear[:] = n["gear"]
        m.actuator_gear[:, 0] = n["gear"][:, 0] * spec.gain
        mujoco.mj_setConst(m, self._data)
        self.current = spec

    def _next_spec(self) -> PerturbationSpec:
        if self.fixed is not None:
            return self.fixed
        if self.randomization is not None and self.randomization.enabled():
            return self.randomization.sample(self.rng)
        return PerturbationSpec()

    # ---- gym API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.apply_spec(self._next_spec())
        obs, info = self.env.reset(seed=seed, options=options)
        zero = np.zeros(self.action_space.shape, dtype=np.float32)
        self._buffer = deque([zero.copy() for _ in range(self.current.action_delay)])
        info = {**info, "perturbation": self.current.to_dict()}
        return self._noisy(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self.current.action_delay > 0:
            self._buffer.append(action)
            action = self._buffer.popleft()
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy(obs), reward, terminated, truncated, info

    def _noisy(self, obs: np.ndarray) -> np.ndarray:
        if self.current.obs_noise <= 0:
            return obs
        if self._obs_scale is None:
            raise RuntimeError("obs_scale is required when obs_noise > 0")
        noise = self.rng.normal(0.0, 1.0, size=obs.shape) * self._obs_scale * self.current.obs_noise
        return (obs + noise).astype(obs.dtype)


def make_perturbed_env(
    env_id: str,
    fixed: PerturbationSpec | None = None,
    randomization: DomainRandomization | None = None,
    seed: int = 0,
) -> gym.Env:
    env = gym.make(env_id)
    needs_noise = (fixed is not None and fixed.obs_noise > 0) or (
        randomization is not None and randomization.obs_noise_max > 0
    )
    scale = estimate_obs_scale(env_id) if needs_noise else None
    return PerturbedEnv(env, fixed=fixed, randomization=randomization, obs_scale=scale, seed=seed)
