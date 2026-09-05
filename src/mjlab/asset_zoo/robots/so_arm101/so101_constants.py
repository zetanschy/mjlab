"""SO-ARM101 for mjlab, built from agent_101's calibrated URDF.

Not the SO-101 MJCF from zacamaso/mjlab_so101 (kept beside this as SO-102-B.xml for
reference): that is a different calibration. This one compiles so101_new_calib.urdf,
which is the model agent_101's Isaac scene, its FK and its hand-eye calibration all
already agree with -- checked against kinematics.fk_gripper over 200 random
configurations at 0.000 micrometres and 2.2e-15 of rotation. That agreement is the
whole point: it means the wrist camera pose fitted against real photographs over there
can be used here verbatim rather than re-derived.

A URDF carries no actuators, no sites and no MuJoCo geom groups, so get_spec() adds
them. The servo numbers -- damping 2.0, frictionloss 0.052, armature 0.1, kp 17.8,
forcerange 1.5 N-m -- are the sts3215 class from the SO-101 MJCF, i.e. the same
physical servo, rather than agent_101's Isaac gains, which are an implicit-actuator
parameterisation that does not carry across simulators.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

SO101_ROOT: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "so_arm101"
SO101_URDF: Path = SO101_ROOT / "xmls" / "so101_calib.urdf"
assert SO101_URDF.exists(), f"URDF not found: {SO101_URDF}"

# STS3215 servo, from the SO-101 MJCF's own default class.
_JOINT_DAMPING = 2.0
_JOINT_FRICTIONLOSS = 0.052
_JOINT_ARMATURE = 0.1
_ACTUATOR_KP = 17.8
_ACTUATOR_FORCE = 1.5

# mjlab's convention, and the wrist camera renders groups (0, 3): visuals are 2 so the
# camera does not see them, collision boxes are 3 so it does.
_VISUAL_GROUP = 2
_COLLISION_GROUP = 3

# The point the reward measures the "end effector" at, and it is MEASURED, not
# estimated. push-T's ee_guidance term drives the policy toward a standoff point
# beside the block, so a site in the wrong place aims the whole approach at the wrong
# height: the first version of this sat where the Jaw joint is and rode 29.6 mm above
# the table while the fingers were at 4.9 mm -- a 25 mm error, larger than the block
# is tall.
#
# This is the centroid of the three geoms that actually touch anything: the two on
# the gripper body and the moving jaw's, expressed in the gripper frame.
GRASP_SITE_POS = (0.0059, 0.0001, -0.0332)

# The arm is yawed 90 degrees so its working direction is +x, which is the direction
# mjlab's Push-T scene pushes in. Measured rather than assumed: at zero joints the
# gripper sits at y = -0.277 in the base frame, so this arm's forward is base -y, and
# +90 degrees about z takes -y to +x. agent_101's Isaac scene applies the same yaw for
# the same reason.
SO101_YAW_DEG = 90.0
_H = math.radians(SO101_YAW_DEG) / 2
SO101_ROT = (math.cos(_H), 0.0, 0.0, math.sin(_H))

# Home pose, solved in agent_101 with scipy against fk_gripper inside the URDF's real
# joint limits: gripper 40 mm above the table, 16 cm to the side of where the block
# spawns, 1.01 rad of margin to the nearest joint limit. Beside the block because a
# home directly over it rests the jaws on the task object; clear of the limits so a
# full action in any direction stays legal.
SO101_HOME = {
  "Rotation": -0.6992,
  "Pitch": 0.7312,
  "Elbow": 0.4016,
  "Wrist_Pitch": -0.0414,
  "Wrist_Roll": -0.0540,
  "Jaw": -0.1500,
}


def get_spec() -> mujoco.MjSpec:
  """The calibrated arm, with the servo model, groups, site and actuators added."""
  spec = mujoco.MjSpec.from_file(str(SO101_URDF))

  # frictionloss only. Damping and armature are the actuator config's to set below --
  # mjlab's BuiltinPositionActuatorCfg overrides whatever the XML says for those two,
  # so setting them here as well would just be a value that never takes effect.
  for joint in spec.joints:
    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joint.frictionloss = _JOINT_FRICTIONLOSS

  # A URDF's visual and collision geoms all land in group 0, where the policy camera
  # would render every one of them and the viewer would draw the collision hulls on
  # top of the meshes. Split them the way the rest of mjlab does.
  # Named as well as grouped. A URDF import leaves every geom anonymous, and a
  # SceneEntityCfg selects geoms BY NAME -- the fingertip friction randomization has
  # nothing to match without this.
  counts: dict[str, int] = {}
  for geom in spec.geoms:
    visual = geom.contype == 0 and geom.conaffinity == 0
    geom.group = _VISUAL_GROUP if visual else _COLLISION_GROUP
    body = geom.parent.name if geom.parent is not None else "world"
    kind = "visual" if visual else "collision"
    key = f"{body}_{kind}"
    counts[key] = counts.get(key, 0) + 1
    geom.name = f"{key}{counts[key]}"

  gripper = next(b for b in spec.bodies if b.name == "gripper")
  gripper.add_site(name="grasp_site", pos=GRASP_SITE_POS, group=_VISUAL_GROUP)

  # No <actuator> elements: mjlab builds them from SO101_ACTUATORS below, the way the
  # YAM does. A URDF has none anyway.
  return spec


# One entry for all six joints: the SO-101 is six of the same STS3215 servo, unlike the
# YAM, which mixes two motor types and needs a per-joint table.
SO101_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=(".*",),
    stiffness=_ACTUATOR_KP,
    damping=_JOINT_DAMPING,
    effort_limit=_ACTUATOR_FORCE,
    armature=_JOINT_ARMATURE,
  ),
)

SO101_ARTICULATION = EntityArticulationInfoCfg(
  actuators=SO101_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
)


def get_so101_robot_cfg(spec_fn=get_spec) -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0), rot=SO101_ROT, joint_pos=dict(SO101_HOME)
    ),
    spec_fn=spec_fn,
    articulation=SO101_ARTICULATION,
  )


# Uniform, and 0.3 rather than the 0.8 mjlab's YAM Push-T settles on. Measured on this
# arm in agent_101: at 0.8 the gripper travels 46 mm per control step -- 1.4 m/s -- and
# spends 30% of steps below the table surface; at 0.3 it is 20 mm, which is about the
# width of the T's stem, so the arm can approach the block without stepping past it.
# 0.8 is the right number for a YAM, which is a much bigger arm reaching much further.
SO101_ACTION_SCALE: float = 0.3


# The geoms that touch the block when the arm pushes: the moving jaw and the fixed
# finger on the gripper body. Named by get_spec above.
SO101_FINGERTIP_GEOMS = r"(jaw|gripper)_collision\d+"
