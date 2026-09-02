"""YAM with the RealSense D435 on its printed support, replacing the D405.

Poses are taken verbatim from the robot-os description that ships with the
physical arm (``yam_arm_description/urdf/yam_cam.urdf``), so the simulated
camera sits where the real one is bolted:

    link_6 -> camera_support   xyz(-0.02618, 0.04089, 0.05052) rpy(1.5708, 1.5708, 3.14159)
    camera_support -> d435     xyz(-0.03014, 0.00604, 0.06023) rpy(2.6180, 0.0, -1.5708)

The one thing NOT taken from the official RealSense description is the optical
frame. That description assumes an x-forward camera link and rotates by
rpy(-pi/2, 0, -pi/2) to get a z-forward optical frame -- but this mount is
hand-authored and its d435 link is ALREADY z-forward. Measured at the home
pose, the link's +z is [0.5, 0, -0.866], which is 0.907 aligned with the
direction the stock D405 looks and is also the axis closest to pointing at the
block; its +x is [0, 1, 0], i.e. straight sideways. Applying the official
optical rotation on top rotates twice and leaves the camera staring off to the
side. So the camera attaches to the d435 link directly, turned 180 degrees about
x so MuJoCo's -z view axis lines up with the link's +z.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from scipy.spatial.transform import Rotation

from mjlab.asset_zoo.robots.i2rt_yam.yam_constants import get_spec

_ASSETS = Path(__file__).parent / "xmls" / "assets"

# link_6 -> camera_support
_SUPPORT_POS = (-0.02618, 0.04089, 0.05052)
_SUPPORT_RPY = (1.5708, 1.5708, 3.14159)
# camera_support -> d435 body. Placed on the midpoint of the support's two
# mounting holes, found by fitting cylinders to the mesh rather than taken from
# the URDF: both come out at radius 2.10 mm, centres
#   (-0.03065,  0.02851, 0.06327)  and  (-0.03065, -0.01649, 0.06327)
# i.e. 45.00 mm apart, which is exactly the D435's M3 hole spacing, so these are
# the camera holes and not some other feature. The URDF's own d435 origin
# (-0.03014, 0.00604, 0.06023) already sat within 0.03 mm of the midpoint in y
# and 0.5 mm in x; this corrects it by 3 mm in z.
_HOLE_A = (-0.03065, 0.02851, 0.06327)
_HOLE_B = (-0.03065, -0.01649, 0.06327)
_HOLE_RADIUS = 0.0021

# The screws run along this axis, fitted from the hole walls (its singular value
# is 0.000, so the fit is exact). It is oblique in xz, which is why the hole
# showed up in both the x and z projections.
_HOLE_AXIS = (0.5555, 0.0, 0.8315)
# The mounting tab is 4.90 mm thick, faces at +/-2.45 mm from the hole centre.
# All the support's material lies from -35.6 to +2.5 mm along the axis, so +2.45
# is the OUTWARD face and the one the camera bolts onto.
_TAB_HALF_THICKNESS = 0.00245
# The Menagerie mesh spans z in [-25.06, 0] mm with its origin on the z = 0
# face. Under this mount's orientation the body's +z points INWARD, 3.7 degrees
# off the screw axis, so the mesh extends OUTWARD from its origin: the origin is
# the camera's mounting face and the lens ends up at the far side.
#
# So the origin seats directly on the tab's outward face -- the tab
# half-thickness only. Adding the body depth as well (the first thing I tried)
# leaves the camera floating 25 mm off the bracket.
_D435_DEPTH = 0.02506
_HOLE_MID = tuple((a + b) / 2.0 for a, b in zip(_HOLE_A, _HOLE_B, strict=True))
# Which face of the mesh is the camera's base, established from the mesh rather
# than assumed: every optical feature sits at z = -1 to -6 mm (IR_Lens -1.4,
# RGB_Pupil -3.6, IR_Emitter_Lens -4.0, the Black_Acrylic bezel -0.8 to -6.4),
# so z = 0 is the LENS face and z = -25.05 the BASE. The mesh origin is the lens,
# not the base.
#
# The camera hangs under the plate with its base against the underside, so the
# origin (the lens) sits a full body depth below the tab's inner face. Unflipped,
# the body's local +z already points down in this mount, which means origin->base
# points up into the plate -- exactly right, so no rotation is needed. Flipping
# the body was the wrong fix: it put the body below the plate but turned the
# camera over, leaving its mounting holes facing open air.
_HOLE_MID = tuple((a + b) / 2.0 for a, b in zip(_HOLE_A, _HOLE_B, strict=True))
_D435_POS = tuple(
  mid - ax * (_TAB_HALF_THICKNESS + _D435_DEPTH)
  for mid, ax in zip(_HOLE_MID, _HOLE_AXIS, strict=True)
)
_D435_RPY = (2.6180, 0.0, -1.5708)
# Lateral position of the optical centre within the camera body.
#
# 15 mm of this is the D435's depth-to-colour baseline. The other 17.2 mm is an
# empirical correction: with the baseline alone the gripper sits well off to one
# side of the frame, which a real image from this camera shows it should not --
# the fingers straddle the centre there. Measured in the camera frame at the home
# pose, left_finger was at +1.1 mm and right_finger at -35.4 mm, so the gripper
# centreline sat 17.2 mm off the optical axis. The gripper and the bracket are
# both rigid on link_6, so correcting it once centres it at every arm pose.
#
# This is a fit to a photograph, not a figure from the hardware description, and
# it is the one number here that is not traceable to either the URDF or the mesh.
# If the bracket's own geometry is what is off, the honest fix is upstream of
# this line.
_COLOR_OFFSET = (0.03219, 0.0, 0.0)

# Body meshes from mujoco_menagerie/realsense_d435i. The D435.obj shipped with
# the arm description cannot be used: it is a single OBJ holding 681 separate
# objects and 30 materials, and MuJoCo's loader takes one mesh per file, so it
# silently loads only the first tiny fragment. That is what produced the "mesh
# volume is too small" error and a camera that compiled but rendered nothing at
# all. The Menagerie files are one object each, which is why they work.
#
# d435i_8 is the outer shell and carries the silhouette (89.9 x 25.0 x 25.1 mm,
# matching the real part); the rest are lens and rim details. d435i_4, the front
# face, is left out -- 11.4 MB for a surface almost coincident with the shell
# front. These are group 2, and the policy's camera renders groups (0, 3), so
# none of this costs anything at training time; only the viewer draws them.
# Mesh -> rgba, copied from mujoco_menagerie/realsense_d435i/d435i.xml so the
# camera looks like the part rather than a black box. The casing is
# Metal_Casing rgba="1 1 1 1", i.e. bare aluminium -- painting it dark grey was
# simply wrong.
_D435_MESHES = {
  "d435i_0": (0.035601, 0.035601, 0.035601, 1.0),  # IR_Lens
  "d435i_1": (0.287440, 0.665387, 0.327778, 1.0),  # IR_Emitter_Lens
  "d435i_2": (0.799102, 0.806952, 0.799103, 1.0),  # IR_Rim
  "d435i_3": (0.035601, 0.035601, 0.035601, 1.0),  # IR_Lens
  "d435i_4": (0.296138, 0.296138, 0.296138, 1.0),  # Cameras_Gray
  "d435i_5": (0.070360, 0.070360, 0.070360, 1.0),  # Black_Acrylic
  "d435i_6": (0.070360, 0.070360, 0.070360, 1.0),  # Black_Acrylic
  "d435i_7": (0.087140, 0.002866, 0.009346, 1.0),  # RGB_Pupil
  "d435i_8": (1.0, 1.0, 1.0, 1.0),  # Metal_Casing
}

_SUPPORT_MASS = 0.1  # kg, from the URDF inertial block
_D435_MASS = 0.072

# Intel D435 colour stream: 69.4 x 42.5 deg. Specified as true intrinsics rather
# than a single fovy, because fovy is vertical only and MuJoCo derives the
# horizontal from the render aspect -- a square 32x32 render would give a 42.5
# deg horizontal too, less than two thirds of the real camera's width. The stock
# D405 in this arm is specified the same way. focal is the D405's 1.93 mm; the
# sensor size then follows from the two FOV figures.
D435_FOCAL = (0.00193, 0.00193)
D435_SENSORSIZE = (0.002673, 0.001501)  # -> 69.4 x 42.5 deg, aspect 1.78
D435_RESOLUTION = (424, 240)  # 16:9, as the real colour stream delivers
D435_ASPECT = D435_SENSORSIZE[0] / D435_SENSORSIZE[1]  # 1.78

_VISUAL_GROUP = 2
# Group 4, not 3. The policy's camera renders groups (0, 3), so a collision box
# sitting in group 3 at the camera's own location is drawn by that camera and
# blocks a strip of its own view. The stock D405 body puts its collision box in
# group 4 for exactly this reason. Collision itself is governed by
# contype/conaffinity, not by the group, so nothing is lost.
_COLLISION_GROUP = 4


def _quat(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
  """URDF fixed-axis rpy -> MuJoCo (w, x, y, z)."""
  x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
  return (float(w), float(x), float(y), float(z))


def get_yam_d435_spec(heavy_visual: bool = False) -> mujoco.MjSpec:
  """YAM spec with the D405 removed and the D435 on its support added."""
  spec = get_spec()

  # Drop the D405 and its camera. Deleting the body takes the camera with it.
  for body in list(spec.bodies):
    if body.name == "camera_d405":
      spec.delete(body)
      break

  link_6 = next(b for b in spec.bodies if b.name == "link_6")

  spec.add_mesh(name="camera_support", file=str(_ASSETS / "camera_support.stl"))
  if heavy_visual:
    for mesh_name in _D435_MESHES:
      spec.add_mesh(
        name=mesh_name,
        file=str(_ASSETS / f"{mesh_name}.obj"),
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
      )

  support = link_6.add_body(
    name="camera_support", pos=_SUPPORT_POS, quat=_quat(_SUPPORT_RPY)
  )
  support.add_geom(
    name="camera_support_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="camera_support",
    mass=_SUPPORT_MASS,
    group=_VISUAL_GROUP,
    contype=0,
    conaffinity=0,
    rgba=(0.25, 0.25, 0.27, 1.0),
  )

  # Mark the holes so the placement is inspectable in the model rather than
  # only in this comment.
  for hole_name, hole_pos in (("mount_hole_a", _HOLE_A), ("mount_hole_b", _HOLE_B)):
    support.add_site(
      name=hole_name,
      pos=hole_pos,
      size=(_HOLE_RADIUS, _HOLE_RADIUS, _HOLE_RADIUS),
      rgba=(1.0, 0.2, 0.2, 0.6),
    )

  d435 = support.add_body(name="d435", pos=_D435_POS, quat=_quat(_D435_RPY))
  if heavy_visual:
    for mesh_name, mesh_rgba in _D435_MESHES.items():
      d435.add_geom(
        name=f"{mesh_name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        # Massless: these are thin surface meshes and MuJoCo cannot infer an
        # inertia from them. The mass lives on the collision box below, the same
        # split the T block uses.
        mass=0.0,
        group=_VISUAL_GROUP,
        contype=0,
        conaffinity=0,
        rgba=mesh_rgba,
      )
  else:
    # A plain box standing in for the 551k-vertex mesh set. The policy's camera
    # renders groups (0, 3) and these visuals are group 2, so it never sees
    # either version -- but the meshes still cost about 1.2 GB of GPU memory
    # whether or not anything renders them, which is enough to push a 4096-env
    # run out of memory. Pass heavy_visual=True when you actually want to look
    # at the camera in a viewer.
    d435.add_geom(
      name="d435_visual_box",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.045, 0.0125, 0.0125),
      pos=(0.0, 0.0, -_D435_DEPTH / 2.0),
      mass=0.0,
      group=_VISUAL_GROUP,
      contype=0,
      conaffinity=0,
      rgba=(0.8, 0.8, 0.82, 1.0),
    )
  # A coarse collision box so the camera cannot pass through the table or the
  # block unnoticed.
  d435.add_geom(
    name="d435_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.0125, 0.045, 0.0125),
    pos=(0.0, 0.0, 0.0),
    mass=_D435_MASS,
    group=_COLLISION_GROUP,
    rgba=(0.1, 0.1, 0.1, 1.0),
  )

  # At the mesh origin, which is the lens face. MuJoCo cameras view along their
  # own -z and the lens looks along the body's +z, so a 180-degree turn about x
  # aligns them.
  # At the lens plane, i.e. the far face of the body, not at the mounting face
  # where the mesh origin sits.
  # quat (0, 0, 1, 0) is 180 deg about y: it aims MuJoCo's -z view axis along the
  # body's +z where the lens looks, AND rolls the image the right way up. The
  # plain 180-about-x turn aims correctly but leaves the picture upside down --
  # in a real frame from this camera the gripper fingers sit in the bottom
  # corners, and with the x-turn they came out along the top edge.
  d435.add_camera(
    name="camera_d435",
    pos=(_COLOR_OFFSET[0], _COLOR_OFFSET[1], 0.0),
    quat=(0.0, 0.0, 1.0, 0.0),
    sensor_size=D435_SENSORSIZE,
    focal_length=D435_FOCAL,
    resolution=list(D435_RESOLUTION),
  )
  return spec


# D435 depth stream FOV, 87 x 58 deg. Wider than the colour stream, and the
# aim measurement below is why it is worth having: mounted as the hardware has
# it, at the arm's home pose the optical axis meets the table at x = 0.206 while
# the task's workspace runs x = 0.28 to 0.42. The block is in frame at 15.8 deg
# off-axis but the goal at x = 0.40 sits 33.8 deg off-axis, outside the colour
# stream's 21.2 deg half-FOV. The camera is on the wrist so the workspace comes
# into view as the arm moves, but at reset the goal is not visible.
D435_DEPTH_FOVY_DEG = 58.0


def get_yam_d435_robot_cfg(heavy_visual: bool = False):
  """EntityCfg for the YAM carrying the D435, otherwise stock.

  The field of view is fixed by D435_SENSORSIZE / D435_FOCAL above, not by a
  parameter. There used to be a fovy_deg argument here; once the camera moved to
  true intrinsics it silently did nothing, so it is gone rather than left as a
  knob that appears to work.
  """
  import dataclasses
  import functools

  from mjlab.asset_zoo.robots import get_yam_robot_cfg

  return dataclasses.replace(
    get_yam_robot_cfg(),
    spec_fn=functools.partial(get_yam_d435_spec, heavy_visual=heavy_visual),
  )
