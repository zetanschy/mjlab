from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def object_out_of_bounds(
  env: ManagerBasedRlEnv,
  object_name: str,
  x_range: tuple[float, float],
  y_range: tuple[float, float],
) -> torch.Tensor:
  """End the episode once the block is shoved outside the reachable workspace.

  Bounds are relative to the environment origin, so they hold for every env in
  the grid. Without this the arm can swat the block away and then spend the rest
  of the episode collecting nothing.
  """
  obj: Entity = env.scene[object_name]
  local_xy = obj.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  outside_x = (local_xy[:, 0] < x_range[0]) | (local_xy[:, 0] > x_range[1])
  outside_y = (local_xy[:, 1] < y_range[0]) | (local_xy[:, 1] > y_range[1])
  return outside_x | outside_y
