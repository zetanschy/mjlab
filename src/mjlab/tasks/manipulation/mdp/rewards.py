from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
)
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Curriculum reward that gates lifting bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before lift.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  position_error = torch.sum(
    torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
  )
  return torch.exp(-position_error / std**2)


def multi_cube_staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward for the target cube selected by MultiCubeLiftingCommand."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = command.target_object_pos()
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def multi_cube_bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian reward for bringing the selected target cube to goal."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  obj_pos_w = command.target_object_pos()
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  return torch.exp(-position_error / std**2)


def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)


# --- Push-T ------------------------------------------------------------------
# Planar pushing: the block stays on the table, so every error below is measured
# in the xy plane. Using 3D position error would hand out reward for lifting the
# block, which is the opposite of what this task wants.


def _planar_pose(
  env: ManagerBasedRlEnv, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  entity: Entity = env.scene[name]
  return entity.data.root_link_pos_w[:, :2], euler_xyz_from_quat(
    entity.data.root_link_quat_w
  )[2]


def _goal_errors(
  env: ManagerBasedRlEnv, object_name: str, goal_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Squared planar position error and absolute wrapped yaw error."""
  obj_xy, obj_yaw = _planar_pose(env, object_name)
  goal_xy, goal_yaw = _planar_pose(env, goal_name)
  pos_err_sq = torch.sum(torch.square(goal_xy - obj_xy), dim=-1)
  yaw_err = wrap_to_pi(goal_yaw - obj_yaw).abs()
  return pos_err_sq, yaw_err


def push_staged_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  reaching_std: float,
  placing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gate the placing bonus on the end effector reaching the block.

  Same shape as :func:`staged_position_reward`: ``reaching * (1 + placing)``.
  Without the reaching gate a pushing policy gets no gradient at all until it
  happens to touch the block, which almost never occurs from random actions.
  """
  robot: Entity = env.scene[asset_cfg.name]
  ee_xy = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)[:, :2]
  obj_xy, _ = _planar_pose(env, object_name)
  reach_err = torch.sum(torch.square(ee_xy - obj_xy), dim=-1)
  reaching = torch.exp(-reach_err / reaching_std**2)

  pos_err_sq, _ = _goal_errors(env, object_name, goal_name)
  placing = torch.exp(-pos_err_sq / placing_std**2)
  return reaching * (1.0 + placing)


def object_goal_pose_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  pos_std: float,
  yaw_std: float,
) -> torch.Tensor:
  """Precise pose match, as the product of position and yaw kernels.

  The product rather than a sum: a T sitting on the footprint but rotated 90
  degrees is not half-solved, and summing would pay out for translation alone
  and stall there.
  """
  pos_err_sq, yaw_err = _goal_errors(env, object_name, goal_name)
  position = torch.exp(-pos_err_sq / pos_std**2)
  orientation = torch.exp(-torch.square(yaw_err) / yaw_std**2)
  return position * orientation


def object_goal_position_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  std: float,
) -> torch.Tensor:
  """Planar position-only kernel, for early shaping before yaw matters."""
  pos_err_sq, _ = _goal_errors(env, object_name, goal_name)
  return torch.exp(-pos_err_sq / std**2)


def push_success_bonus(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  pos_tol: float = 0.02,
  yaw_tol: float = 0.26,
) -> torch.Tensor:
  """Sparse 1.0 while the block is settled on the footprint.

  Paid every step it holds, so the policy is pushed to arrive early and then
  leave the block alone rather than nudging it back and forth.
  """
  pos_err_sq, yaw_err = _goal_errors(env, object_name, goal_name)
  solved = (pos_err_sq <= pos_tol**2) & (yaw_err <= yaw_tol)
  return solved.float()


def object_displacement_penalty(
  env: ManagerBasedRlEnv,
  object_name: str,
) -> torch.Tensor:
  """Penalize the block leaving the table plane (being lifted or flipped).

  Returns squared vertical deviation from its resting height plus how far the
  block's own +z axis has tipped away from world +z.
  """
  obj: Entity = env.scene[object_name]
  height_err = torch.square(
    obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  )
  up_axis = quat_apply(
    obj.data.root_link_quat_w,
    torch.tensor([0.0, 0.0, 1.0], device=obj.data.root_link_pos_w.device).expand(
      obj.data.root_link_pos_w.shape[0], 3
    ),
  )
  tilt = 1.0 - up_axis[:, 2]
  return height_err + tilt
