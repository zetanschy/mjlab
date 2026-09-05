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
# The MJCF, generated once from so101_calib.urdf (kept beside it) by compiling the
# URDF and wrapping the worldbody's children in a body named "base".
#
# That wrapper is not cosmetic. MuJoCo's URDF importer turns the root link into
# WORLDBODY, so the arm's links hang directly off the world and the entity has no root
# body -- and mjlab then never runs the actuator group at all. Everything looked right
# in that state: six actuators wired to the six joints with the correct gains, the
# action term computing correct targets, and the arm not moving, because
# BuiltinPositionActuator.compute() was called exactly zero times per step.
SO101_XML: Path = SO101_ROOT / "xmls" / "so101_calib.xml"
assert SO101_XML.exists(), f"MJCF not found: {SO101_XML}"

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

# The real SO-ARM101 in agent_101's workshop is printed in WHITE; the URDF ships the
# printed parts in the yellow the CAD was authored in. Same value that repo's
# set_robot_color paints its Isaac arm with, so the two simulators show one robot.
#
# Only the printed parts. The dark (0.1, 0.1, 0.1) geoms are the STS3215 servo bodies
# and they really are that colour -- agent_101 repaints Looks/material_a_3d_printed and
# leaves the servos alone for the same reason.
_PRINTED_YELLOW = (1.0, 0.82, 0.12)
_PRINTED_WHITE = (0.93, 0.93, 0.95, 1.0)
# The servo bodies, explicitly black rather than the URDF's very dark grey.
_SERVO_GREY = (0.1, 0.1, 0.1)
_SERVO_BLACK = (0.05, 0.05, 0.05, 1.0)

# The point the reward measures the "end effector" at, and it is MEASURED, not
# estimated. push-T's ee_guidance term drives the policy toward a standoff point
# beside the block, so a site in the wrong place aims the whole approach at the wrong
# height: the first version of this sat where the Jaw joint is and rode 29.6 mm above
# the table while the fingers were at 4.9 mm -- a 25 mm error, larger than the block
# is tall.
#
# The midpoint of the two FINGERTIPS, found by taking the 40 lowest mesh vertices of
# the fixed finger and of the moving jaw at the home pose and averaging them.
#
# Not the centroid of the geoms' origins, which is what this was first: a collision
# mesh's origin is near the middle of the part, and for this gripper that sat 70 mm
# ABOVE the surfaces that actually touch anything. Asking the guidance term to bring
# that point down to the block's 10 mm buried the real fingers 25 mm into the table.
# Measured at the home pose the two tips are 7.0 and 10.2 mm above the surface while
# the old site was at 80 mm.
GRASP_SITE_POS = (-0.0074, 0.0, -0.1035)

# The arm is yawed 90 degrees so its working direction is +x, which is the direction
# mjlab's Push-T scene pushes in. Measured rather than assumed: at zero joints the
# gripper sits at y = -0.277 in the base frame, so this arm's forward is base -y, and
# +90 degrees about z takes -y to +x. agent_101's Isaac scene applies the same yaw for
# the same reason.
SO101_YAW_DEG = 90.0
_H = math.radians(SO101_YAW_DEG) / 2
SO101_ROT = (math.cos(_H), 0.0, 0.0, math.sin(_H))

# Home pose, solved against this model's own FK inside the URDF's real joint limits:
# the FINGERTIP 35 mm above the table, 16 cm to the side of where the block spawns,
# 1.15 rad of margin to the nearest joint limit.
#
# The fingertip, not the gripper body, and that distinction is the whole point. The
# first version of this pose put the GRIPPER 40 mm up, which left the finger geoms at
# 4.9 mm -- touching the table. The arm was then pinned by friction at its own home:
# commanded +0.3 rad, Rotation and Elbow moved 0.0000 while Pitch tore free and ran
# 2.0 rad to its limit, dragging the others with it. It reads exactly like a robot
# that will not move, which is what it was.
#
# The constraint is CONTACT, evaluated by MuJoCo, not a height threshold on geom
# origins. That distinction cost two wrong poses. A geom's origin says nothing about
# where its surface is: with the fingertip site at 35 mm and the finger geom ORIGINS
# reading 20.7 mm "above the table", MuJoCo reported those same geoms 35.7 mm BELOW
# it -- the collision hulls extend about 55 mm past their own centres. The arm was
# buried in the table at its home pose, the ee_ground_collision sensor fired on every
# step in every environment at 405 N, and the episode reset before the policy could
# move anything. That is what "the arm is not moving" was.
#
# So this pose is chosen by running the collision detector: no geom of the arm may come
# within 2 mm of the floor plane. It puts the fingertip 40 mm above the table at world
# (0.13, +0.05) -- level with the block's spawn line and 5 cm to one side of it.
#
# 5 cm, not the 14 cm this was first set to. The lateral offset and the block's own
# +/-6 cm spawn draw ADD: at 14 cm aside the fingertip started 190 mm from the block on
# an ordinary reset and averaged 218 mm over a rollout, which is most of this arm's
# working range spent travelling to the task instead of doing it. At 5 cm the typical
# gap is 50 mm and the worst case 110 mm.
#
# Closer than this is possible and not worth it: the limit margin falls with the offset
# (0.547 rad here, 0.663 at 11 cm) and it has to stay above the 0.5 action scale or the
# policy's commands start being clipped by the joints. Overlap with a block spawned
# toward this side is fine -- the fingertip sits 40 mm up and the block is 20 mm tall.
SO101_HOME = {
  "Rotation": -0.2585,
  "Pitch": -0.5715,
  "Elbow": 1.0236,
  "Wrist_Pitch": 1.1098,
  "Wrist_Roll": -0.1473,
  "Jaw": -0.1500,
}


def get_spec() -> mujoco.MjSpec:
  """The calibrated arm, with the servo model, groups, site and actuators added."""
  spec = mujoco.MjSpec.from_file(str(SO101_XML))

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

    # Repaint the printed parts, VISUAL AND COLLISION alike. The collision geoms are
    # the ones that matter to the policy: the wrist camera renders groups (0, 3), so
    # what it sees of the arm is the group-3 collision meshes, not the group-2 visuals
    # a person looks at. Painting only the visuals would leave a yellow arm in the
    # observation and a white one on screen.
    shade = tuple(round(float(v), 2) for v in geom.rgba[:3])
    if shade == _PRINTED_YELLOW:
      geom.rgba = _PRINTED_WHITE
    elif shade == _SERVO_GREY:
      geom.rgba = _SERVO_BLACK
    if not visual:
      # NO SELF-COLLISION. contype 1 / conaffinity 0 means these geoms are collided
      # AGAINST by the world and the block -- whose geoms carry conaffinity 1 -- but
      # never against each other, because a pair needs one side's contype to meet the
      # other's conaffinity and arm-versus-arm has neither.
      #
      # A URDF gives every link a collision mesh that is the convex hull of the whole
      # part, and adjacent links overlap heavily by construction: measured here, the
      # base's hulls sit 22 to 28 mm INSIDE the shoulder's, permanently. MuJoCo then
      # spends every step resolving a penetration that cannot be escaped, and the
      # impulses go straight into the joint between them -- the Rotation joint hit
      # +/-27 rad/s under a ZERO action command, walked 0.85 rad off its target, and
      # dragged the whole arm somewhere unrelated to the task. From outside it looks
      # like an arm twitching in the wrong place, which is exactly what it was.
      #
      # The SO-101 MJCF avoids this by giving its links small hand-fitted collision
      # BOXES instead of hulls. That is the better model and a bigger change; this is
      # the one line that makes the imported URDF behave.
      geom.contype = 1
      geom.conaffinity = 0

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


# Uniform, and chosen by mjlab's own test rather than by per-step travel, which is what
# 0.3 was chosen by and which turned out to measure the wrong thing.
#
# With use_default_offset the policy can only ever reach home +/- scale on each joint,
# so the scale IS the workspace. Sampling 40k points of that box and mapping the
# FINGERTIP into world:
#
#   scale   fingertip y range      at block height   gets BEHIND the block
#   0.3     +0.03 .. +0.26 m        12937/40000       674  (1.69%)
#   0.5     -0.03 .. +0.31          10370             2218 (5.54%)
#   0.8     -0.09 .. +0.39           6626             1172 (2.93%)
#   1.2     -0.15 .. +0.44           4895              843 (2.11%)
#
# At 0.3 the reachable y range NEVER CROSSES ZERO. The block sits on y = 0, so the arm
# could not touch it at all -- it jiggled in a small patch beside the task, which is
# exactly what it looked like from outside.
#
# 0.5 is the peak. Beyond it the box gets coarser faster than it gets bigger and less
# of it lands anywhere useful, which is the same shape mjlab found on the YAM: 0.8 gave
# 2.12% there and 1.2 rad was worse at 1.64%. 5.54% is comfortably above the 2.12% that
# trained to 99% success over there.
SO101_ACTION_SCALE: float = 0.5


# The geoms that touch the block when the arm pushes: the moving jaw and the fixed
# finger on the gripper body. Named by get_spec above.
SO101_FINGERTIP_GEOMS = r"(jaw|gripper)_collision\d+"
