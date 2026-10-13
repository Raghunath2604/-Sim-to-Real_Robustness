import gymnasium as gym
import numpy as np
import pytest

from robust_rl.perturb import (
    DomainRandomization,
    PerturbationSpec,
    estimate_obs_scale,
    make_perturbed_env,
    spec_for,
)

ENV = "HalfCheetah-v5"


def rollout(env, actions, seed=0):
    obs, _ = env.reset(seed=seed)
    traj = [obs.copy()]
    for a in actions:
        obs, *_ = env.step(a)
        traj.append(obs.copy())
    return np.array(traj)


@pytest.fixture(scope="module")
def actions():
    rng = np.random.default_rng(0)
    return [rng.uniform(-1, 1, size=6).astype(np.float32) for _ in range(25)]


def test_nominal_wrapper_is_identical_to_plain_env(actions):
    plain = rollout(gym.make(ENV), actions)
    wrapped = rollout(make_perturbed_env(ENV, fixed=PerturbationSpec()), actions)
    np.testing.assert_allclose(plain, wrapped)


@pytest.mark.parametrize("axis,value", [("friction", 3.0), ("mass", 3.0), ("gain", 2.0), ("action_delay", 3)])
def test_each_axis_changes_the_dynamics(actions, axis, value):
    plain = rollout(gym.make(ENV), actions)
    perturbed = rollout(make_perturbed_env(ENV, fixed=spec_for(axis, value)), actions)
    assert np.abs(plain - perturbed).max() > 1e-3


def test_physics_scale_exactly_and_do_not_accumulate():
    ref = gym.make(ENV).unwrapped.model
    env = make_perturbed_env(ENV, fixed=spec_for("mass", 3.0))
    m = env.unwrapped.model
    for seed in (0, 1, 2):  # repeated resets must not compound the scale
        env.reset(seed=seed)
        assert np.isclose(m.body_mass.sum() / ref.body_mass.sum(), 3.0)
        np.testing.assert_allclose(m.body_inertia, ref.body_inertia * 3.0)
    env.fixed = PerturbationSpec()
    env.reset(seed=0)
    np.testing.assert_allclose(m.body_mass, ref.body_mass)  # nominal restored exactly


def test_friction_scales_only_sliding_friction():
    ref = gym.make(ENV).unwrapped.model
    env = make_perturbed_env(ENV, fixed=spec_for("friction", 2.0))
    env.reset(seed=0)
    m = env.unwrapped.model
    np.testing.assert_allclose(m.geom_friction[:, 0], ref.geom_friction[:, 0] * 2.0)
    np.testing.assert_allclose(m.geom_friction[:, 1:], ref.geom_friction[:, 1:])


def test_gain_scales_actuator_gear():
    ref = gym.make(ENV).unwrapped.model
    env = make_perturbed_env(ENV, fixed=spec_for("gain", 1.5))
    env.reset(seed=0)
    np.testing.assert_allclose(env.unwrapped.model.actuator_gear[:, 0], ref.actuator_gear[:, 0] * 1.5)


def test_action_delay_applies_older_actions():
    env = make_perturbed_env(ENV, fixed=spec_for("action_delay", 2))
    env.reset(seed=0)
    applied = []
    for i in range(1, 6):
        env.step(np.full(6, i / 10, dtype=np.float32))
        applied.append(float(env.unwrapped.data.ctrl[0]))
    np.testing.assert_allclose(applied, [0.0, 0.0, 0.1, 0.2, 0.3], atol=1e-6)


def test_obs_noise_has_requested_scale_and_is_reproducible():
    scale = estimate_obs_scale(ENV)
    env = make_perturbed_env(ENV, fixed=spec_for("obs_noise", 0.5), seed=0)
    clean = gym.make(ENV)
    obs_n, _ = env.reset(seed=0)
    obs_c, _ = clean.reset(seed=0)
    diffs = []
    for _ in range(300):
        a = np.zeros(6, dtype=np.float32)
        obs_n, *_ = env.step(a)
        obs_c, *_ = clean.step(a)
        diffs.append(obs_n - obs_c)
    # dynamics are identical (noise is sensing-only), so the difference is pure sensor noise
    noise_std = np.std(np.array(diffs), axis=0)
    assert np.allclose(noise_std / (0.5 * scale), 1.0, atol=0.25)
    a1 = make_perturbed_env(ENV, fixed=spec_for("obs_noise", 0.5)).reset(seed=7)[0]
    a2 = make_perturbed_env(ENV, fixed=spec_for("obs_noise", 0.5)).reset(seed=7)[0]
    np.testing.assert_allclose(a1, a2)


def test_domain_randomization_ranges_and_seeding():
    dr = DomainRandomization(strength=0.5, obs_noise_max=0.2, action_delay_max=3)
    rng = np.random.default_rng(0)
    specs = [dr.sample(rng) for _ in range(3000)]
    for axis in ("friction", "mass", "gain"):
        v = np.array([getattr(s, axis) for s in specs])
        assert v.min() >= 1 / 1.5 - 1e-9 and v.max() <= 1.5 + 1e-9
        assert abs(np.log(v).mean()) < 0.03  # symmetric on a log scale
    assert {s.action_delay for s in specs} == {0, 1, 2, 3}
    assert max(s.obs_noise for s in specs) <= 0.2
    env_a = make_perturbed_env(ENV, randomization=dr, seed=0)
    env_b = make_perturbed_env(ENV, randomization=dr, seed=0)
    assert env_a.reset(seed=3)[1]["perturbation"] == env_b.reset(seed=3)[1]["perturbation"]


def test_disabled_randomization_is_nominal():
    assert not DomainRandomization().enabled()
    assert DomainRandomization().sample(np.random.default_rng(0)).is_nominal()


def test_conflicting_arguments_rejected():
    with pytest.raises(ValueError):
        make_perturbed_env(ENV, fixed=PerturbationSpec(), randomization=DomainRandomization(strength=0.3))
    with pytest.raises(ValueError):
        spec_for("not_an_axis", 1.0)
