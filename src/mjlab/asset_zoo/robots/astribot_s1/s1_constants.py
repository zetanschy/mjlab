"""Astribot S1 humanoid, using the simulation's own tabletop scene.

The model is NOT vendored. Its MJCF references meshes under
``astribot_descriptions/urdf/astribot_s1_urdf/meshes``, a 336 MB tree, so this
module reads a local clone of

    https://github.com/Astribot-Dev/astribot_simulation

Override the location with ``ASTRIBOT_SIM_PATH``. A task using this robot will
not load on a machine without that clone.

The scene taken is ``astribot_s1_for_aloha_with_gripper.xml`` -- the repo's own
tabletop setup, complete with its simpleTable (top face at z = 0.76) and its
head, wrist and global cameras. Only the demo's loose red box and the scene
furniture mjlab supplies itself (floor, light, the ``human`` viewing camera) are
removed. Actuator gains are read back out of the compiled model rather than
restated here, so they cannot drift from the source.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_DEFAULT_REPO = Path.home() / "astribot_simulation"
_SCENE = "astribot_s1_for_aloha_with_gripper.xml"

# Joints the policy drives. The gripper is excluded: its actuator drives a tendon
# rather than a joint, so a joint-position action term cannot target it, and a
# Push-T policy handed a free gripper on the YAM used it to fork and carry the
# block instead of pushing.
ARM_JOINTS = tuple(f"astribot_arm_right_joint_{i}" for i in range(1, 8))

# Torso and head are held at the home pose rather than actuated. They sit between
# the base and the head camera, so driving them moves the camera relative to the
# workspace -- a harder perception problem than the wrist-camera tasks whose
# reward this reuses.
POSTURE_JOINTS = tuple(f"astribot_torso_joint_{i}" for i in (1, 2, 3, 4)) + (
  "astribot_head_joint_1",
  "astribot_head_joint_2",
)

# Home pose, found by searching the torso/head/arm joint box under four
# constraints at once:
#
#   * all four corners of the 0.28 x 0.20 m working area project inside the
#     head camera's frame, which is the constraint that actually matters and is
#     not the same as aiming the camera at the middle of it. A first pose sat
#     0.255 m off the table, covering 0.246 m against a task spanning about
#     0.25 m, and the T kept running off the frame edge; backing off to 0.534 m
#     fixed the clipping but shrank the T to 32 px. The camera is therefore
#     pulled in to 0.347 m, covering 0.334 m.
#   * that view is centred on (0.802, -0.086), well clear of the table's y edges
#     at +/-0.375, so a randomly placed T never overhangs.
#   * the robot does not stand in its own way: all 36 sample points across the
#     working area are reachable by a ray from the camera without hitting the
#     robot first. An earlier pose scored 100% on the other three constraints
#     but left the gripper parked between the head and the table, hiding the
#     green footprint -- the goal averaged 8 px, invisible outright in 13 of
#     512 environments.
#   * 98% of the action box brings the gripper below the top of the block, so
#     the arm can actually get alongside it and push. This one has to be part
#     of the search rather than a check afterwards: the YAM Push-T task was
#     debugged for twelve runs on the assumption that its reward was wrong,
#     when in fact 0.00% of its action box could reach the block's side.
#   * the tool frame starts about 0.10 m off the point being looked at, axis
#     0.881 aligned with straight down. Sitting closer is what put the gripper
#     in front of the camera, so the offset is deliberate.
#   * every arm joint keeps 0.365 rad of margin inside its soft limit band --
#     see the note on the arm entries below.
HOME_POSE: dict[str, float] = {
  "astribot_torso_joint_1": 0.1144,
  "astribot_torso_joint_2": -1.0999,
  "astribot_torso_joint_3": 1.5337,
  "astribot_torso_joint_4": -0.3656,
  "astribot_head_joint_1": 0.3506,
  "astribot_head_joint_2": 0.5678,
  # Every arm joint must sit well inside its *soft* limit band (0.9 of range),
  # because the joint_pos_limits reward is weighted -10. The first pose tried
  # here left joint 5 outside that band, so the penalty fired on every step
  # from reset -- a constant -1.2 per episode against a task reward of +0.26,
  # which is the flat, penalty-dominated landscape that stalls learning. The
  # action box is +/-0.3 rad, so the margin is sized to match it.
  "astribot_arm_right_joint_1": 1.4076,
  "astribot_arm_right_joint_2": -0.0758,
  "astribot_arm_right_joint_3": 0.3991,
  "astribot_arm_right_joint_4": 0.3465,
  "astribot_arm_right_joint_5": -1.436,
  "astribot_arm_right_joint_6": -0.3751,
  "astribot_arm_right_joint_7": -0.5102,
}

TABLE_HEIGHT = 0.76
WORKSPACE_CENTRE = (0.4575, -0.0312)
HEAD_CAMERA = "head_rgbd"
EE_BODY = "astribot_gripper_right_base"
EE_SITE = "arm_right_tool"
"""The scene's own tool-frame site, 21 mm behind the fingertip midpoint.

Reused rather than adding a TCP site of our own, since the reward terms want a
single end-effector point and this one already tracks the gripper.
"""


def astribot_repo_path() -> Path:
  """Local clone of astribot_simulation, from env var or the default location."""
  path = Path(os.environ.get("ASTRIBOT_SIM_PATH", _DEFAULT_REPO))
  if not (path / "astribot_descriptions/mjcf/astribot_s1_mjcf").is_dir():
    raise FileNotFoundError(
      f"Astribot S1 model not found under {path}. Clone "
      "https://github.com/Astribot-Dev/astribot_simulation and either put it at "
      f"{_DEFAULT_REPO} or set ASTRIBOT_SIM_PATH. Its meshes are ~336 MB and are "
      "deliberately not vendored into mjlab."
    )
  return path


def get_spec() -> mujoco.MjSpec:
  """The repo's tabletop scene, with a home keyframe added."""
  mjcf = astribot_repo_path() / "astribot_descriptions/mjcf/astribot_s1_mjcf"
  spec = mujoco.MjSpec.from_file(str(mjcf / _SCENE))

  # mjlab supplies terrain, lighting and viewing cameras.
  for geom in list(spec.worldbody.geoms):
    if geom.name == "floor":
      spec.delete(geom)
  for light in list(spec.worldbody.lights):
    spec.delete(light)
  for camera in list(spec.worldbody.cameras):
    if camera.name == "human":
      spec.delete(camera)
  # The demo's loose red box; this task brings its own T block.
  for body in list(spec.worldbody.bodies):
    if body.name == "box_red":
      spec.delete(body)

  # Validate HOME_POSE against the compiled model; mjlab turns the dict itself
  # into the entity's `init_state` keyframe, writing both qpos and ctrl, so no
  # keyframe is added here.
  model = spec.compile()
  for name in HOME_POSE:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
      raise ValueError(f"Joint '{name}' not in the S1 model")

  return spec


def _actuators() -> tuple[XmlActuatorCfg, ...]:
  """Adopt the scene's own actuators rather than declaring new ones.

  ``XmlActuatorCfg`` wraps the existing ``<general>`` elements, so every gain,
  force limit and control range stays exactly as the simulation ships it --
  nothing about the robot's dynamics is restated here.

  Every joint servo is wrapped, not just the right arm's: mjlab writes ``ctrl``
  only for actuators it manages, so an unwrapped servo keeps ``ctrl`` at zero
  and drags its joint there. Wrapping them puts all of them under the position
  targets seeded by ``hold_default_joint_targets``, and the action term then
  overwrites just the right arm on each step.

  The two tendon-driven gripper servos are deliberately left out. They are
  ``<general>`` elements mjlab cannot classify as a position servo, and an
  unmanaged actuator holds ``ctrl`` at zero -- which for these is a closed
  gripper. That is what this task wants: the T is pushed, never grasped.

  The joints are named explicitly rather than matched with ``".*"``, which also
  matches the two gripper tendons and the five sites and warns about each.
  """
  model = get_spec().compile()
  names = tuple(
    n
    for i in range(model.njnt)
    if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)) is not None
  )
  return (XmlActuatorCfg(target_names_expr=names),)


def get_astribot_s1_cfg() -> EntityCfg:
  """EntityCfg for the S1 with the right arm exposed to the action manager."""
  return EntityCfg(
    spec_fn=get_spec,
    # mjlab builds the `init_state` keyframe from this dict and, crucially,
    # fills ctrl for *every* actuator in the spec -- joints absent from
    # HOME_POSE get 0.0, which is their home value. That keeps the scene's own
    # position servos holding the torso, head and left arm in the pose the head
    # camera depends on, instead of driving them to zero.
    #
    # joint_pos=None (adopt the spec's own keyframe) is not usable here: mjlab
    # reads the *scene* keyframe as `key_qpos[nq_root:]`, which for a fixed-base
    # entity picks up the T block's 7 free-joint values as if they were robot
    # joints (42 vs 35).
    init_state=EntityCfg.InitialStateCfg(joint_pos=dict(HOME_POSE)),
    articulation=EntityArticulationInfoCfg(
      actuators=_actuators(),
      soft_joint_pos_limit_factor=0.9,
    ),
  )
