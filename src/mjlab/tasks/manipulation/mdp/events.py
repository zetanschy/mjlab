"""Event terms shared by the manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def hold_default_joint_targets(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Seed every joint's position target with its default pose.

  Position targets are zeroed on reset and thereafter only written for joints an
  action term drives. A robot whose action term covers one arm therefore has the
  rest of its servos commanded to zero, which for a posture with a nonzero
  default (a bent torso, a tilted head) means they fight the pose instead of
  holding it.

  Targets persist for the rest of the episode, so writing them once at reset is
  enough; the action term overwrites the joints it owns on every step.
  """
  env_ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene[asset_cfg.name]
  robot.set_joint_position_target(
    robot.data.default_joint_pos[env_ids], env_ids=env_ids
  )
