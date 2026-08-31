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


def _lazy_success_pose(*a, **k):
  from mjlab.tasks.manipulation.mdp.metrics import success_pose as _sp

  return _sp(*a, **k)


success_pose = _lazy_success_pose

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


# --- Push-T, coverage-based ---------------------------------------------------
# Modelled on the two reference Push-T implementations rather than the lifting
# rewards above, which do not transfer to pushing:
#
#   gym-pusht:  reward = clip(coverage / 0.95, 0, 1), coverage being the
#               intersection area of block and goal T divided by goal area.
#   ManiSkill:  dense reward sums a rotation term ((cos(dyaw)+1)/2)**2/2 and a
#               position term (1-tanh(5d))**2/2, plus a much smaller end
#               effector guidance term scaled by 1/20.
#
# Two properties matter and were missing before. The terms are *added*, not
# multiplied, so a wrong yaw cannot zero out the translation gradient. And yaw
# enters through cos, which decays gracefully to zero at 180 degrees instead of
# a Gaussian kernel that is already ~6e-5 at the mean error of a uniform +/-pi
# goal, leaving nothing to descend.

_T_POINTS_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _t_points(device: torch.device, spacing: float = 0.0025) -> torch.Tensor:
  """Points tiling the T's area in body frame, for area overlap by sampling.

  Sampling rather than exact polygon clipping: the shape is fixed and batched
  over thousands of envs, so a cached point set reduces coverage to one rigid
  transform plus an inside test.
  """
  from mjlab.tasks.manipulation.push_t_scene import (
    _CROSSBAR_HALF,
    _CROSSBAR_POS,
    _STEM_HALF,
    _STEM_POS,
    CENTROID_OFFSET_Y,
  )

  key = (str(device), int(spacing * 1e6))
  cached = _T_POINTS_CACHE.get(key)
  if cached is not None:
    return cached

  boxes = []
  for half, pos in ((_CROSSBAR_HALF, _CROSSBAR_POS), (_STEM_HALF, _STEM_POS)):
    cy = pos[1] + CENTROID_OFFSET_Y
    boxes.append((-half[0], half[0], cy - half[1], cy + half[1]))

  xs = torch.arange(-0.05 + spacing / 2, 0.05, spacing, device=device)
  ys = torch.arange(-0.07 + spacing / 2, 0.04, spacing, device=device)
  gx, gy = torch.meshgrid(xs, ys, indexing="ij")
  pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

  inside = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
  for x0, x1, y0, y1 in boxes:
    inside |= (
      (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    )
  pts = pts[inside]
  _T_POINTS_CACHE[key] = pts
  return pts


def _inside_t(pts: torch.Tensor) -> torch.Tensor:
  """Inside test for points already expressed in a T's body frame."""
  from mjlab.tasks.manipulation.push_t_scene import (
    _CROSSBAR_HALF,
    _CROSSBAR_POS,
    _STEM_HALF,
    _STEM_POS,
    CENTROID_OFFSET_Y,
  )

  out = torch.zeros(pts.shape[:-1], dtype=torch.bool, device=pts.device)
  for half, pos in ((_CROSSBAR_HALF, _CROSSBAR_POS), (_STEM_HALF, _STEM_POS)):
    cy = pos[1] + CENTROID_OFFSET_Y
    out |= (pts[..., 0].abs() <= half[0]) & ((pts[..., 1] - cy).abs() <= half[1])
  return out


def object_goal_coverage(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
) -> torch.Tensor:
  """Fraction of the block's area lying inside the goal footprint, in [0, 1].

  Block and goal are the same shape, so this equals the intersection over the
  goal area that gym-pusht reports.
  """
  obj_xy, obj_yaw = _planar_pose(env, object_name)
  goal_xy, goal_yaw = _planar_pose(env, goal_name)

  pts = _t_points(obj_xy.device)  # (P, 2)
  rel_yaw = obj_yaw - goal_yaw  # (B,)
  cos_t, sin_t = torch.cos(rel_yaw), torch.sin(rel_yaw)

  # Block points into the goal frame: R(-yaw_g) @ (R(yaw_o) p + t_o - t_g).
  px = pts[None, :, 0] * cos_t[:, None] - pts[None, :, 1] * sin_t[:, None]
  py = pts[None, :, 0] * sin_t[:, None] + pts[None, :, 1] * cos_t[:, None]

  d = obj_xy - goal_xy  # (B, 2)
  cos_g, sin_g = torch.cos(goal_yaw), torch.sin(goal_yaw)
  ox = d[:, 0] * cos_g + d[:, 1] * sin_g
  oy = -d[:, 0] * sin_g + d[:, 1] * cos_g

  local = torch.stack([px + ox[:, None], py + oy[:, None]], dim=-1)  # (B, P, 2)
  return _inside_t(local).float().mean(dim=-1)


def push_position_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  scale: float = 5.0,
) -> torch.Tensor:
  """ManiSkill's position term: (1 - tanh(scale * d))**2 / 2, max 0.5.

  tanh rather than a Gaussian so the gradient survives at the spawn distance
  instead of underflowing.
  """
  pos_err_sq, _ = _goal_errors(env, object_name, goal_name)
  dist = pos_err_sq.clamp_min(1e-12).sqrt()
  return torch.square(1.0 - torch.tanh(scale * dist)) / 2.0


def push_orientation_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  gate_distance: float = 0.08,
) -> torch.Tensor:
  """ManiSkill's rotation term ((cos(dyaw)+1)/2)**2/2, gated on position.

  Non-zero and differentiable at every yaw error, including the +/-pi/2 mean of
  a uniformly randomized goal, where a Gaussian kernel underflows.

  The gate is the important part. Yaw can be changed without moving the block
  anywhere: spun in place, this term pays in full for a skill that needs no
  pushing at all. Ungated and weighted above position it is strictly the better
  deal, and a policy trained that way rotated the block on the spot and
  translated it 0.001 m over an entire episode. Gating on distance to the goal
  means the rotation bonus only exists once the block has been delivered, so
  the sole route to it runs through translation.
  """
  pos_err_sq, yaw_err = _goal_errors(env, object_name, goal_name)
  aligned = torch.square((torch.cos(yaw_err) + 1.0) / 2.0) / 2.0
  dist = pos_err_sq.clamp_min(1e-12).sqrt()
  gate = (1.0 - dist / gate_distance).clamp(0.0, 1.0)
  return aligned * gate


def push_ee_guidance_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  scale: float = 5.0,
  standoff: float = 0.06,
  gate_distance: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """End effector proximity to a push pose, in 3D, capped at 0.05 and gated.

  The distance is 3D. Measured in the xy plane only, this term is maximized by
  hovering directly above the block at any altitude: a policy trained against
  the planar version parked its gripper 23 cm up, pinned the term at its cap,
  and never touched the block.

  The target is a standoff point on the far side of the block from the goal,
  which is the right place to start a push from but the wrong place to finish.
  That point lies on the line through the block's centre of mass, so force
  applied there produces no torque. Left active it teaches pure translation,
  and a policy trained that way placed the block within 2 cm while leaving yaw
  at 80 degrees of error, because the contact it was being paid to hold cannot
  rotate the block.

  So the term is gated by how far the block still is from its goal position: at
  full strength while the block has to travel, and off once it has arrived,
  leaving the policy free to take the off-centre contact that rotation needs.
  """
  from mjlab.tasks.manipulation.push_t_scene import T_THICKNESS

  robot: Entity = env.scene[asset_cfg.name]
  ee = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj: Entity = env.scene[object_name]
  goal: Entity = env.scene[goal_name]

  obj_pos = obj.data.root_link_pos_w
  to_goal = goal.data.root_link_pos_w[:, :2] - obj_pos[:, :2]
  goal_dist = to_goal.norm(dim=-1)
  direction = to_goal / goal_dist.clamp_min(1e-6).unsqueeze(-1)

  push_pose = obj_pos.clone()
  push_pose[:, :2] = obj_pos[:, :2] - direction * standoff
  push_pose[:, 2] = obj_pos[:, 2] + T_THICKNESS / 2.0

  dist = (ee - push_pose).norm(dim=-1)
  proximity = (1.0 - torch.tanh(scale * dist)).clamp_min(0.0).sqrt() / 20.0
  gate = (goal_dist / gate_distance).clamp(0.0, 1.0)
  return proximity * gate


def push_coverage_success(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  threshold: float = 0.90,
) -> torch.Tensor:
  """Sparse bonus once area overlap clears the threshold (ManiSkill uses 0.90)."""
  return (object_goal_coverage(env, object_name, goal_name) >= threshold).float()


def maniskill_push_t_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  success_threshold: float = 0.90,
  success_mode: str = "coverage",
  pos_tol: float = 0.02,
  yaw_tol_deg: float = 15.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """ManiSkill's PushT-v1 dense reward, reproduced as a single term.

  A faithful baseline, deliberately not tuned. Everything else in this file has
  been modified in response to failures on the randomized-goal variant, which
  is a harder task than any published Push-T attempts, so none of it is
  evidence about whether the reference reward works on this robot. This term
  changes nothing:

    rotation      ((cos(dyaw) + 1) / 2)**2 / 2          max 0.50
    position      (1 - tanh(5 * d))**2 / 2              max 0.50
    ee guidance   sqrt(1 - tanh(5 * d_tcp)) / 20        max 0.05

  summed, with success *replacing* the total with 3.0 rather than adding to it,
  and the result normalized by 3.0. Guidance targets the block's centre in 3D,
  not a standoff point, and nothing is gated.
  """
  obj: Entity = env.scene[object_name]
  robot: Entity = env.scene[asset_cfg.name]

  _, yaw_err = _goal_errors(env, object_name, goal_name)
  rotation = torch.square((torch.cos(yaw_err) + 1.0) / 2.0) / 2.0

  pos_err_sq, _ = _goal_errors(env, object_name, goal_name)
  dist = pos_err_sq.clamp_min(1e-12).sqrt()
  position = torch.square(1.0 - torch.tanh(5.0 * dist)) / 2.0

  ee = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  tcp_dist = (ee - obj.data.root_link_pos_w).norm(dim=-1)
  guidance = (1.0 - torch.tanh(5.0 * tcp_dist)).clamp_min(0.0).sqrt() / 20.0

  reward = rotation + position + guidance

  if success_mode == "pose":
    # coverage >= 0.90 requires |dyaw| <= 4.6 deg AND |d| <= 2.9 mm jointly. It
    # has fired zero times in roughly 590M steps, so the largest term in the
    # reward (3.0, i.e. 4.1x the shaped ceiling) has never produced a gradient.
    # The pose predicate is reachable, and the flatness conjuncts stop it being
    # satisfied by lifting -- a cheaper skill than rotating.
    solved = success_pose(
      env, object_name, goal_name, pos_tol=pos_tol, yaw_tol_deg=yaw_tol_deg
    ).bool()
  else:
    solved = object_goal_coverage(env, object_name, goal_name) >= success_threshold
  reward = torch.where(solved, torch.full_like(reward, 3.0), reward)
  return reward / 3.0


def staged_pose_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  pos_scale: float = 5.0,
  yaw_weight: float = 1.0,
) -> torch.Tensor:
  """Position kernel multiplied by (1 + orientation kernel). Max 1.0 + yaw_weight.

  This is the shape of ``staged_position_reward`` -- ``reaching * (1 + bringing)``
  -- which is the only reward that has produced a working policy on this arm
  (lift-cube, 93.8% within 5 cm). Applied to an SE(2) goal, position takes the
  role of reaching and orientation the role of bringing.

  It is multiplicative staging rather than the alternatives already tried:
    - additive with equal weights let position win outright and orientation sat
      at its uniform-spawn null for the whole run;
    - orientation weighted above position made rotating on the spot the better
      deal, and the block stopped moving entirely;
    - gating orientation on position punished a rotational correction twice,
      once through the position term and again through the shrinking gate.
  Here orientation is worth nothing where position is poor, so it cannot be
  farmed in place, and it never zeroes the position gradient, so approach is
  always rewarded. Aligning at the goal doubles the return.

  A rotational correction that costs some position still dips before it pays.
  That dip is why the discount has to span the episode: the payoff arrives at
  the end, and a value function with a horizon a tenth of the episode cannot
  see it from where the dip is.
  """
  pos_err_sq, yaw_err = _goal_errors(env, object_name, goal_name)
  dist = pos_err_sq.clamp_min(1e-12).sqrt()
  position = torch.square(1.0 - torch.tanh(pos_scale * dist)) / 2.0
  orientation = torch.square((torch.cos(yaw_err) + 1.0) / 2.0) / 2.0
  return position * (1.0 + yaw_weight * 2.0 * orientation)


def push_precision_bonus(
  env: ManagerBasedRlEnv,
  object_name: str,
  goal_name: str,
  pos_scale: float = 20.0,
  yaw_scale: float = 3.0,
) -> torch.Tensor:
  """Fine-scale pose bonus, in [0, 1], for closing the last centimetre.

  The coarse terms are saturated by the time the policy is any good. At the
  working policy's 17 mm and 8.9 deg, the position kernel (1-tanh(5d))**2/2 sits
  at 0.419 of 0.5 and the rotation kernel ((cos+1)/2)**2/2 at 0.494 of 0.5 --
  1.2% of headroom. There is almost nothing left to earn by getting tighter, so
  the policy correctly stops trying.

  Sharpening those kernels in place would fix the near field and wreck the far
  one: at k=40 the position term is 0.014 at 30 mm, which is the flat region
  that made rotation unlearnable for twelve runs. So this is a SECOND scale,
  added alongside the coarse reward rather than replacing it. Far from the goal
  it is ~0 and contributes nothing; inside about 2 cm it supplies the gradient
  the coarse term no longer has.

  The two factors are multiplied, which is safe here in a way it was not when I
  tried it on the coarse terms: this whole quantity is additive to a reward that
  already has an independent position gradient, so a wrong yaw cannot zero the
  approach signal. It only gates the *bonus*, not the task.
  """
  pos_err_sq, yaw_err = _goal_errors(env, object_name, goal_name)
  dist = pos_err_sq.clamp_min(1e-12).sqrt()
  fine_pos = torch.square(1.0 - torch.tanh(pos_scale * dist)) / 2.0
  fine_yaw = torch.square(1.0 - torch.tanh(yaw_scale * yaw_err)) / 2.0
  return 4.0 * fine_pos * fine_yaw
