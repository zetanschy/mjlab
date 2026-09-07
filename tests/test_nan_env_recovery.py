"""Tests for per-env recovery from a non-finite physics state."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

# MjWarp's CPU backend is unreliable here; see test_nan_guard.py.
pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp"
)


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_env(device, recover: bool) -> ManagerBasedRlEnv:
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 4
  cfg.recover_nan_envs = recover
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  return env


def _obs_tensors(obs) -> list[tuple[str, torch.Tensor]]:
  """Flatten an observation dict into (name, tensor) pairs."""
  out = []
  for group, value in obs.items():
    if isinstance(value, torch.Tensor):
      out.append((group, value))
    else:
      out.extend((f"{group}/{k}", v) for k, v in value.items())
  return out


def _step_with_nan_in(env: ManagerBasedRlEnv, env_id: int):
  """Poison one env's velocity, then step. Returns step() outputs."""
  env.sim.data.qvel[env_id] = float("nan")
  action = torch.zeros((env.num_envs, 1), device=env.device)
  return env.step(action)


def test_nan_env_is_recovered_and_truncated(device):
  """A NaN env is restored, truncated, and kept out of the observation buffer."""
  env = _make_env(device, recover=True)
  obs, reward, terminated, truncated, _ = _step_with_nan_in(env, env_id=1)

  for name, tensor in _obs_tensors(obs):
    assert torch.isfinite(tensor).all(), f"obs '{name}' is not finite"
  assert torch.isfinite(reward).all()

  # Truncated, not terminated: the episode ended outside the MDP, so the critic
  # should bootstrap rather than learn the pre-blow-up state is worth zero.
  assert truncated[1] and not terminated[1]
  assert env._nan_recovery_count == 1

  # The physics state is finite again and the episode counter was reset.
  assert torch.isfinite(env.sim.data.qvel).all()
  assert torch.isfinite(env.sim.data.qacc_warmstart).all()
  assert env.episode_length_buf[1] == 0
  env.close()


def test_recovery_leaves_other_envs_alone(device):
  """Only the NaN env is reset; its neighbours keep running their episodes."""
  env = _make_env(device, recover=True)
  for _ in range(3):
    env.step(torch.zeros((env.num_envs, 1), device=env.device))
  before = env.episode_length_buf.clone()

  _, _, terminated, truncated, _ = _step_with_nan_in(env, env_id=2)

  others = [0, 1, 3]
  assert not (terminated[others] | truncated[others]).any()
  assert (env.episode_length_buf[others] == before[others] + 1).all()
  env.close()


def test_recovery_can_be_disabled(device):
  """With recover_nan_envs=False the NaN reaches the observation buffer."""
  env = _make_env(device, recover=False)
  obs, _, _, _, _ = _step_with_nan_in(env, env_id=1)

  assert any(not torch.isfinite(t).all() for _, t in _obs_tensors(obs))
  assert env._nan_recovery_count == 0
  env.close()
