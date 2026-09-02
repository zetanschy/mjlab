"""Per-component metrics for the Push-T task.

Run 7 logged only the aggregate ``Episode_Reward/maniskill``, so every claim
about whether the block was being rotated had to be reconstructed analytically
after the fact -- and two such reconstructions turned out to be wrong. The
orientation reward was read as peaking and decaying when it was in fact
stationary at 98.1% of its uniform-spawn null (mean 0.1839 against 0.1875,
first half 0.1837, second half 0.1842); the "peak" and "decay" were the max and
min of noise with sd 0.031. These terms measure the quantities directly so no
reconstruction is needed.

``rot_component`` is the ungated ((cos dyaw + 1) / 2)**2 / 2, whose analytic
null under a uniform +/-pi spawn is exactly 0.1875. Logging it next to a known
null is what makes "is the block being rotated at all" answerable at a glance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.manipulation.mdp.rewards import _goal_errors, object_goal_coverage

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_RAD2DEG = 57.29577951308232

# Analytic mean of ((cos(dyaw) + 1) / 2)**2 / 2 for dyaw uniform on [-pi, pi].
UNIFORM_YAW_NULL = 0.1875


def yaw_err_deg(
  env: ManagerBasedRlEnv, object_name: str, goal_name: str
) -> torch.Tensor:
  """Absolute yaw error to the goal, in degrees. Chance is ~90 for a +/-pi spawn."""
  _, yaw_err = _goal_errors(env, object_name, goal_name)
  return yaw_err * _RAD2DEG


def pos_err_m(env: ManagerBasedRlEnv, object_name: str, goal_name: str) -> torch.Tensor:
  """Planar distance from the block to the goal, in metres."""
  pos_err_sq, _ = _goal_errors(env, object_name, goal_name)
  return pos_err_sq.clamp_min(1e-12).sqrt()


def rot_component(
  env: ManagerBasedRlEnv, object_name: str, goal_name: str
) -> torch.Tensor:
  """Ungated rotation kernel. Compare against UNIFORM_YAW_NULL = 0.1875."""
  _, yaw_err = _goal_errors(env, object_name, goal_name)
  return torch.square((torch.cos(yaw_err) + 1.0) / 2.0) / 2.0


def coverage(env: ManagerBasedRlEnv, object_name: str, goal_name: str) -> torch.Tensor:
  """Area overlap of block and goal footprint, in [0, 1]."""
  return object_goal_coverage(env, object_name, goal_name)


def success_pose(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  pos_tol: float = 0.02,
  yaw_tol_deg: float = 15.0,
  height_tol: float = 0.005,
  min_up_z: float = 0.99,
  table_height: float = 0.0,
) -> torch.Tensor:
  """Solved: within tolerance in position and yaw, and still flat on the table.

  The flatness conjuncts are not decoration. Position and yaw alone can be
  satisfied by a lifted or tilted block, and lifting is a cheaper skill than
  rotating -- exactly the kind of substitution that has already cost this task
  several runs.
  """
  pos_err_sq, yaw_err = _goal_errors(env, object_name, goal_name)
  obj: Entity = env.scene[object_name]

  within_pos = pos_err_sq <= pos_tol**2
  within_yaw = yaw_err <= (yaw_tol_deg / _RAD2DEG)

  # table_height is the working surface above the environment origin. It is 0 for
  # a robot working off the ground plane and 0.75 for one working at a table; the
  # flatness test is meaningless without it.
  height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - table_height
  flat_z = height.abs() <= height_tol
  up_z = _block_up_z(obj)
  upright = up_z >= min_up_z
  return (within_pos & within_yaw & flat_z & upright).float()


def success_coverage(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  threshold: float = 0.90,
) -> torch.Tensor:
  """The old success predicate, kept for comparison against run 7."""
  return (object_goal_coverage(env, object_name, goal_name) >= threshold).float()


def block_travel_m(
  env: ManagerBasedRlEnv, object_name: str, goal_name: str
) -> torch.Tensor:
  """Planar distance from the block to its spawn position.

  Distinguishes "the policy pushed the block to the goal" from "the policy
  never touched it", which the goal-relative error alone cannot do.
  """
  obj: Entity = env.scene[object_name]
  default = obj.data.default_root_state
  assert default is not None
  spawn_xy = default[:, :2] + env.scene.env_origins[:, :2]
  return (obj.data.root_link_pos_w[:, :2] - spawn_xy).norm(dim=-1)


def block_height_mm(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  table_height: float = 0.0,
) -> torch.Tensor:
  """Block height above the working surface, in mm. Near 0 for pushing."""
  obj: Entity = env.scene[object_name]
  height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - table_height
  return height * 1000.0


def block_tilt_dot(
  env: ManagerBasedRlEnv, object_name: str, goal_name: str
) -> torch.Tensor:
  """Block +z axis dotted with world +z. 1.0 is flat; below 0.99 is tipping."""
  return _block_up_z(env.scene[object_name])


def _block_up_z(obj: Entity) -> torch.Tensor:
  """Third column of the block's rotation matrix, i.e. its own +z in world."""
  q = obj.data.root_link_quat_w
  x, y = q[:, 1], q[:, 2]
  return 1.0 - 2.0 * (x * x + y * y)
