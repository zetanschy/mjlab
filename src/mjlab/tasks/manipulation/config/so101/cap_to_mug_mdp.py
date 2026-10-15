"""Cap-to-mug on the agent_101 digital twin: the MDP terms.

The task the pi0.5 policies were collected and scored on ("Put the cap into the red
cup"): one cap and the mug on the mat, drawn as the twin draws a live layout; pick the
cap, carry it over the rim, drop it in, go back to the rest pose and stay there.

Everything physical comes from the twin (asset_zoo/robots/so_arm101/twin.py): the
scene, the cap and mug dimensions, the table height and tilt, the rest pose, the
layout sampler's ranges and clearances (real2sim/live/layouts.py), and the success
test (real2sim/metrics.py cap_in_mug). They are read once, on first use, so importing
this module does not need agent_101.

Frames: the twin's. The arm's base is at each env's origin with forward along -y; the
table top is at table_z (+ the fitted tilt) in that frame.
"""

from __future__ import annotations

import functools
import importlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.asset_zoo.robots.so_arm101 import twin
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor

ROBOT, CAP, MUG = "robot", "cap0", "mug"
GRASP_SITE = "grasp_site"
HOME_RAD = 0.3  # arm_to_rest within this is "home": the success metric and reward_home
FLIP_X = (0.0, 1.0, 0.0, 0.0)  # an open-side-up cap: the twin's _FLIP_X (objects.py)


@functools.cache
def constants() -> dict:
  """The twin's numbers, in SI units and URDF radians (CPU floats / numpy)."""
  t, sc = twin.twin(), twin.scene()
  cap, mug = sc.cap_dims(), sc.mug_dims()
  rest = sc.units().to_urdf(np.asarray(t.layouts.rest_state(sc))[None])[0]
  kin = importlib.import_module("real2sim.kinematics")
  tilt = sc["table"].get("tilt") or (0.0, 0.0)
  lay = t.layouts
  return {
    "cap_h": float(cap["height"]),
    "cap_d": float(cap["diameter"]),
    "mug_ri": float(mug["outer_diameter"] / 2 - mug["wall"]),
    "mug_ro": float(mug["outer_diameter"] / 2),
    "mug_floor": float(mug["floor"]),
    "mug_h": float(mug["height"]),
    "table_z": float(sc.table_z()),
    "tilt": (float(tilt[0]), float(tilt[1])),
    "rest": np.asarray(rest, dtype=float),
    "rest_tip_xy": np.asarray(kin.fingertips(rest)["mid"][:2], dtype=float),
    "cap_r": tuple(lay.CAP_R),
    "mug_r": tuple(lay.MUG_R),
    "azimuth": math.radians(lay.AZIMUTH_DEG),
    "x_min": float(lay.X_MIN),
    "cap_mug": float(lay.CAP_MUG_CLEARANCE),
    "rest_clear": float(lay.REST_CLEARANCE),
  }


def rest_joint_pos() -> dict[str, float]:
  """The rest pose as an entity joint_pos dict (the arm starts every episode here)."""
  names = twin.twin().units.URDF_JOINTS
  return {n: float(v) for n, v in zip(names, constants()["rest"], strict=True)}


def _table_z(xy: torch.Tensor) -> torch.Tensor:
  c = constants()
  return c["table_z"] + c["tilt"][0] * xy[..., 0] + c["tilt"][1] * xy[..., 1]


# --- action -----------------------------------------------------------------------


@dataclass(kw_only=True)
class IntegratedJointPositionActionCfg(BaseActionCfg):
  """target = previous target + scale * action, clamped to the joint limits.

  mjlab's RelativeJointPositionAction adds the delta to the MEASURED position, so a
  zero action leaves no restoring torque and the arm sags under gravity: "stay still"
  cannot be said. Integrating the TARGET is what a teleop leader or a policy sends the
  real servos: an absolute goal every 30 Hz frame, held until the next one. A zero
  action holds it. Reset puts the target at the entity's default joint pose (the twin's
  rest pose), where the reset event also puts the arm.
  """

  def build(self, env: ManagerBasedRlEnv) -> IntegratedJointPositionAction:
    return IntegratedJointPositionAction(self, env)


class IntegratedJointPositionAction(BaseAction):
  def __init__(self, cfg: IntegratedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    ids = self._target_ids
    self._target = self._entity.data.default_joint_pos[:, ids].clone()
    limits = self._entity.data.joint_pos_limits[:, ids]
    self._lo, self._hi = limits[..., 0], limits[..., 1]

  @property
  def target(self) -> torch.Tensor:
    return self._target

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)  # raw -> scale (+ clip): a per-frame delta
    self._target = torch.clamp(
      self._target + self._processed_actions, self._lo, self._hi
    )

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = self._target_ids
    self._target[env_ids] = self._entity.data.default_joint_pos[env_ids][:, ids]


# --- state helpers -------------------------------------------------------------------


def _origins(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene.env_origins


def cap_centre_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Centre of the cap's cylinder (world): origin + R z * h / 2 (objects.cap_centre)."""
  cap: Entity = env.scene[CAP]
  z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
    cap.data.root_link_pos_w
  )
  axis = quat_apply(cap.data.root_link_quat_w, z)
  return cap.data.root_link_pos_w + axis * constants()["cap_h"] / 2


def cap_axis_z(env: ManagerBasedRlEnv) -> torch.Tensor:
  """z of the cap's axis: +1 standing closed side up, -1 open side up."""
  cap: Entity = env.scene[CAP]
  z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
    cap.data.root_link_pos_w
  )
  return quat_apply(cap.data.root_link_quat_w, z)[:, 2]


def cap_in_mug_local(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The cap centre in the mug's frame (metrics.cap_in_mug's `local`)."""
  mug: Entity = env.scene[MUG]
  d = cap_centre_w(env) - mug.data.root_link_pos_w
  return quat_apply(quat_inv(mug.data.root_link_quat_w), d)


def cap_in_mug(env: ManagerBasedRlEnv, margin: float = 0.0) -> torch.Tensor:
  """The twin's success test (metrics.cap_in_mug): the cap centre inside the cavity,
  radius < inner radius - margin, between the floor and the rim."""
  c = constants()
  local = cap_in_mug_local(env)
  r = torch.linalg.norm(local[:, :2], dim=-1)
  return (
    (r < c["mug_ri"] - margin)
    & (local[:, 2] >= c["mug_floor"] - 1e-4)
    & (local[:, 2] <= c["mug_h"])
  )


def metric_cap_in_mug(env: ManagerBasedRlEnv) -> torch.Tensor:
  return cap_in_mug(env).float()


def metric_cap_in_mug_and_home(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (cap_in_mug(env) & (arm_to_rest(env) < HOME_RAD)).float()


def rim_centre_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  mug: Entity = env.scene[MUG]
  z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
    mug.data.root_link_pos_w
  )
  return (
    mug.data.root_link_pos_w
    + quat_apply(mug.data.root_link_quat_w, z) * constants()["mug_h"]
  )


def ee_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene[ROBOT]
  sid = robot.find_sites(GRASP_SITE)[0]
  return robot.data.site_pos_w[:, sid].squeeze(1)


def fingers_on_cap(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(B, 2): the fixed and the moving finger each touching the cap."""
  out = []
  for name in ("fixed_finger_cap", "moving_finger_cap"):
    s: ContactSensor = env.scene[name]
    assert s.data.found is not None
    out.append(s.data.found.reshape(env.num_envs, -1).gt(0).any(dim=-1))
  return torch.stack(out, dim=-1)


def lift_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How far the cap centre is above where it rests on the table."""
  c = cap_centre_w(env)
  return c[:, 2] - (
    _table_z(c[:, :2] - _origins(env)[:, :2]) + constants()["cap_h"] / 2
  )


def arm_to_rest(env: ManagerBasedRlEnv) -> torch.Tensor:
  """|q - rest| over the five arm joints (the jaw excluded), rad."""
  robot: Entity = env.scene[ROBOT]
  rest = torch.as_tensor(
    constants()["rest"][:5], dtype=torch.float32, device=env.device
  )
  return torch.linalg.norm(robot.data.joint_pos[:, :5] - rest, dim=-1)


# --- layouts (events) ----------------------------------------------------------------


def _polar(n: int, r_range, k: int, device) -> torch.Tensor:
  """(n, k, 2) candidates: layouts._polar's distribution, r and azimuth uniform."""
  c = constants()
  r = torch.empty(n, k, device=device).uniform_(*r_range)
  a = torch.empty(n, k, device=device).uniform_(-c["azimuth"], c["azimuth"])
  return torch.stack([r * torch.sin(a), -r * torch.cos(a)], dim=-1)


def _first_valid(
  cand: torch.Tensor, ok: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  idx = torch.argmax(ok.int(), dim=1)
  pick = cand[torch.arange(cand.shape[0], device=cand.device), idx]
  return pick, ok.any(dim=1)


def sample_layouts(n: int, device, p_open: float, max_cap_mug: float | None = None):
  """One cap and the mug per env, as layouts.random draws them (n_caps=1): the mug
  0.16-0.30 m and the cap 0.14-0.34 m out, within 70 deg of straight ahead, x >= -0.27;
  the cap >= 10 cm from the mug and >= 7 cm from the resting fingertips, the mug
  clear of them by 7 + 4.5 cm. max_cap_mug (curriculum) also caps the cap-mug distance.
  Returns cap_xy, open (bool), mug_xy, handle_yaw (rad)."""
  c = constants()
  tip = torch.as_tensor(c["rest_tip_xy"], dtype=torch.float32, device=device)
  mug_c = _polar(n, c["mug_r"], 32, device)
  mug_ok = (mug_c[..., 0] >= c["x_min"]) & (
    torch.linalg.norm(mug_c - tip, dim=-1) >= c["rest_clear"] + 0.045
  )
  mug, ok_m = _first_valid(mug_c, mug_ok)
  cap_c = _polar(n, c["cap_r"], 128, device)
  dist = torch.linalg.norm(cap_c - mug[:, None], dim=-1)
  cap_ok = (
    (cap_c[..., 0] >= c["x_min"])
    & (dist >= c["cap_mug"])
    & (torch.linalg.norm(cap_c - tip, dim=-1) >= c["rest_clear"])
  )
  if max_cap_mug is not None:
    cap_ok &= dist <= max(max_cap_mug, c["cap_mug"] + 0.01)
  cap, ok_c = _first_valid(cap_c, cap_ok)
  if not bool((ok_m & ok_c).all()):  # vanishingly rare with 32 x 128 candidates
    bad = ~(ok_m & ok_c)
    cap[bad] = torch.tensor([0.0, -0.22], device=device)
    mug[bad] = torch.tensor([0.15, -0.20], device=device)
  is_open = torch.rand(n, device=device) < p_open
  yaw = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
  return cap, is_open, mug, yaw


def reset_layout(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  p_open: float = 0.0,
  max_cap_mug: float | None = None,
) -> None:
  """Place the cap and the mug as layouts.random + objects.cap_pose / mug_pose would:
  resting on the (tilted) table, an open-up cap flipped about x, the mug yawed by its
  handle angle. Also records the episode's starting mug position."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  c = constants()
  n = len(env_ids)
  dev = env.device
  st = _state(env)
  cap_xy, is_open, mug_xy, yaw = sample_layouts(n, dev, p_open, max_cap_mug)
  origin = _origins(env)[env_ids]
  cap_z = _table_z(cap_xy) + torch.where(is_open, c["cap_h"], 0.0)
  cap_q = torch.zeros(n, 4, device=dev)
  cap_q[:, 0] = 1.0
  cap_q[is_open] = torch.tensor(FLIP_X, device=dev)
  cap_state = torch.zeros(n, 13, device=dev)
  cap_state[:, :3] = origin + torch.cat([cap_xy, cap_z[:, None]], dim=-1)
  cap_state[:, 3:7] = cap_q
  mug_state = torch.zeros(n, 13, device=dev)
  mug_state[:, :3] = origin + torch.cat([mug_xy, _table_z(mug_xy)[:, None]], dim=-1)
  mug_state[:, 3] = torch.cos(yaw / 2)
  mug_state[:, 6] = torch.sin(yaw / 2)
  env.scene[CAP].write_root_state_to_sim(cap_state, env_ids=env_ids)
  env.scene[MUG].write_root_state_to_sim(mug_state, env_ids=env_ids)
  st["mug_xy0"][env_ids] = mug_xy


def _state(env: ManagerBasedRlEnv) -> dict:
  st = getattr(env, "_cap_to_mug", None)
  if st is None:
    st = {"mug_xy0": torch.zeros(env.num_envs, 2, device=env.device)}
    env._cap_to_mug = st  # type: ignore[attr-defined]
  return st


# --- observations ----------------------------------------------------------------------


def obs_joint_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Joint positions relative to the rest pose (rad)."""
  robot: Entity = env.scene[ROBOT]
  rest = torch.as_tensor(constants()["rest"], dtype=torch.float32, device=env.device)
  return robot.data.joint_pos[:, :6] - rest


def obs_joint_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene[ROBOT]
  return robot.data.joint_vel[:, :6]


def obs_ee(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The grasp site in the base frame (base = env origin, unrotated)."""
  return ee_pos_w(env) - _origins(env)


def obs_cap_from_ee(env: ManagerBasedRlEnv) -> torch.Tensor:
  return cap_centre_w(env) - ee_pos_w(env)


def obs_rim_from_ee(env: ManagerBasedRlEnv) -> torch.Tensor:
  return rim_centre_w(env) - ee_pos_w(env)


def obs_cap_from_rim(env: ManagerBasedRlEnv) -> torch.Tensor:
  return cap_centre_w(env) - rim_centre_w(env)


def obs_cap_up(env: ManagerBasedRlEnv) -> torch.Tensor:
  return cap_axis_z(env)[:, None]


def obs_handle(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The mug's yaw as (cos, sin): where its handle is."""
  q = env.scene[MUG].data.root_link_quat_w
  yaw = 2 * torch.atan2(q[:, 3], q[:, 0])
  return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def obs_fingers_on_cap(env: ManagerBasedRlEnv) -> torch.Tensor:
  return fingers_on_cap(env).float()


# --- rewards -------------------------------------------------------------------------


def reward_reach(env: ManagerBasedRlEnv, std: float = 0.05) -> torch.Tensor:
  """Bring the grasp site to the cap (until the cap is in the mug)."""
  d = torch.linalg.norm(cap_centre_w(env) - ee_pos_w(env), dim=-1)
  return (1 - torch.tanh(d / std)) * (~cap_in_mug(env)).float()


def jaw_open(env: ManagerBasedRlEnv) -> torch.Tensor:
  """0 with the jaw at its lower (closed) limit, 1 at its upper (open) one."""
  robot: Entity = env.scene[ROBOT]
  jaw = robot.joint_names.index("Jaw")
  lo, hi = robot.data.soft_joint_pos_limits[:, jaw].unbind(-1)
  return (robot.data.joint_pos[:, jaw] - lo) / (hi - lo)


def reward_close(env: ManagerBasedRlEnv, std: float = 0.015) -> torch.Tensor:
  """Close the jaw while the grasp site is at the cap.

  Reach alone is maximised with the jaw wide open, and without this the teacher sat
  there: at iteration 2300 the grasp site was 7 mm from the cap with the jaw at its
  open limit all episode, so the moving finger never touched and grasp never paid."""
  d = torch.linalg.norm(cap_centre_w(env) - ee_pos_w(env), dim=-1)
  near = torch.exp(-((d / std) ** 2))
  return near * (1 - jaw_open(env)) * (~cap_in_mug(env)).float()


def reward_release(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Open the jaw once the cap is in the mug.

  Letting go earned nothing, and moving off with the cap gripped pulls it back out: run
  4's teacher had the cap in by 2 s and held it, jaw shut, for the other 18 s."""
  return cap_in_mug(env).float() * jaw_open(env)


def reward_grasp(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Both fingers on the cap."""
  return fingers_on_cap(env).all(dim=-1).float() * (~cap_in_mug(env)).float()


def reward_lift(env: ManagerBasedRlEnv, clearance: float = 0.03) -> torch.Tensor:
  """Held and raised, up to the rim plus `clearance` (the rim is 12 cm up)."""
  target = constants()["mug_h"] + clearance
  held = fingers_on_cap(env).all(dim=-1)
  # Not once it is in the mug: holding it there paid ~1.2 and the teacher never let go.
  frac = torch.clamp(lift_height(env) / target, 0.0, 1.0)
  return held.float() * frac * (~cap_in_mug(env)).float()


def over_opening(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The whole cap within the mug's opening in xy (at any height above its floor)."""
  c = constants()
  r = torch.linalg.norm(cap_in_mug_local(env)[:, :2], dim=-1)
  return r < c["mug_ri"] - c["cap_d"] / 2


def reward_carry(env: ManagerBasedRlEnv, std: float = 0.08) -> torch.Tensor:
  """Over the rim and closing on the mug's axis: the cap's underside above the rim
  (or the cap over the opening, so the reward does not drop out on the way in), its
  centre's xy toward the rim centre's."""
  cap_c, rim = cap_centre_w(env), rim_centre_w(env)
  above = (cap_c[:, 2] - constants()["cap_h"] / 2) > rim[:, 2] + 0.005
  d = torch.linalg.norm(cap_c[:, :2] - rim[:, :2], dim=-1)
  return (above | over_opening(env)).float() * (1 - torch.tanh(d / std))


def reward_lower(env: ManagerBasedRlEnv, std: float = 0.05) -> torch.Tensor:
  """Over the opening, bring the cap down to the rim; full once its centre is inside.

  Without it nothing pays for the descent: the iteration-2999 teacher held the cap over
  the opening a median 84 mm above the rim for 91% of the episode, and never in."""
  dz = cap_in_mug_local(env)[:, 2] - constants()["mug_h"]
  return over_opening(env).float() * (1 - torch.tanh(dz.clamp(min=0.0) / std))


def reward_in_mug(env: ManagerBasedRlEnv) -> torch.Tensor:
  return cap_in_mug(env).float()


# side 0.02, not 0.05: at 0.05 the teacher stopped on the way home 4.6+ cm above the rim
# but a median 3.9 cm out (iteration 9099), where going lower toward rest lost clear.
def reward_clear(
  env: ManagerBasedRlEnv, height: float = 0.04, side: float = 0.02
) -> torch.Tensor:
  """Once the cap is in the mug: the grasp site clear of the mug, `height` above its rim
  or `side` beyond its outer wall, so the way home is free (and stays paid on it).

  After letting go the lowest finger piece sat 3 mm above the rim (iteration 6600), where
  every move toward rest catches the mug, and the arm stayed ~3 rad from home."""
  ee, rim = ee_pos_w(env), rim_centre_w(env)
  up = ((ee[:, 2] - rim[:, 2]) / height).clamp(0.0, 1.0)
  r = torch.linalg.norm(ee[:, :2] - rim[:, :2], dim=-1)
  out = ((r - constants()["mug_ro"]) / side).clamp(0.0, 1.0)
  return cap_in_mug(env).float() * torch.maximum(up, out)


# std 3.0, not 0.3: with the cap in the mug the arm is a median 3.09 rad from rest
# (iteration 3500), where a 0.3 kernel is ~0, so going home earned nothing on the way.
def reward_home(env: ManagerBasedRlEnv, std: float = 3.0) -> torch.Tensor:
  """With the cap in the mug: back to the rest pose, and still once there (the behaviour
  the operator corrected for in the DAgger rounds)."""
  robot: Entity = env.scene[ROBOT]
  still = torch.exp(-torch.linalg.norm(robot.data.joint_vel[:, :5], dim=-1))
  d = arm_to_rest(env)
  near = 1 - torch.tanh(d / std)
  # Stillness is a bonus at home, not a factor: as a factor, arriving while still moving
  # cut the reward, and the teacher stopped a median 0.68 rad short (iteration 10800).
  # Scaled by reward_clear (which carries the cap-in-mug gate): from the rim the straight
  # joint-space way home tipped the mug in 79% of scripted tries, lifted 5 cm first in 14%.
  return reward_clear(env) * (near + (d < HOME_RAD).float() * still)


def penalty_mug_moved(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How far the mug has been pushed from where it started (m)."""
  mug_xy = env.scene[MUG].data.root_link_pos_w[:, :2] - _origins(env)[:, :2]
  return torch.linalg.norm(mug_xy - _state(env)["mug_xy0"], dim=-1)


# --- terminations ---------------------------------------------------------------------


def mug_tipped(env: ManagerBasedRlEnv, min_up: float = 0.7) -> torch.Tensor:
  q = env.scene[MUG].data.root_link_quat_w
  z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(q.shape[0], 3)
  return quat_apply(q, z)[:, 2] < min_up


def cap_lost(env: ManagerBasedRlEnv, reach: float = 0.55) -> torch.Tensor:
  """The cap fell off the table, or out of the arm's reach."""
  c = cap_centre_w(env) - _origins(env)
  below = c[:, 2] < constants()["table_z"] - 0.02
  return below | (torch.linalg.norm(c[:, :2], dim=-1) > reach)
