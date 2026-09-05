"""SO-ARM101 with the printed klip_support bracket and the KWC-500 wrist camera.

The YAM's D435 module takes its poses from the arm's official description. There is no
official description for this bracket -- it is a printed part, and where its camera
actually ends up was established empirically in agent_101, against photographs from the
real camera:

    mount   bolted to the two wrist holes, posed by hand in the viewport
    camera  700 Nelder-Mead evaluations over all nine camera parameters, each one an
            actual render scored on how well the rendered gripper silhouette lands on
            the real frame: mean boundary error 17 px -> 3.1 px, IoU 0.932

Both poses are in the `gripper` body frame, and this MJCF is the same arm as the one
they were fitted on -- FK agrees to 2e-15 -- so they are used verbatim. That is the
reason this file exists rather than a fresh guess at where a camera sits.

The camera pose is EMPIRICAL, not derived. agent_101's note is worth repeating: 20
degrees of total rotation away from the bracket's bore is more than a barrel can rock
in a 19 mm hole, so something upstream -- the hand-posed mount, or the gripper link
frame itself -- carries error this absorbs. It matches the real camera, which is what
sim2real needs; it is not a measurement of where the lens sits in the bracket.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots.so_arm101.so101_constants import get_spec

_ASSETS = Path(__file__).parent / "xmls" / "assets"

# gripper -> klip_support, from agent_101 assets/objects.py. The mount bolts to two
# wrist holes, which is a hard mechanical constraint you can check by eye.
_MOUNT_POS = (-0.01455, 0.08710, -0.01039)
_MOUNT_RPY_DEG = (-179.755, -34.920, -90.061)

# gripper -> camera, the fitted pose. A SIBLING of the mount, not a child: in agent_101
# the camera is built around the measured lens and the modelled webcam body around
# that, so nesting it under a hand-posed bracket would add the bracket's error to a
# number that was fitted without it.
_CAM_POS = (0.00313, 0.06116, -0.01911)
_CAM_RPY_DEG = (-44.989, -1.976, -3.675)

# Klip Xtreme KWC-500 FHD colour stream, from agent_101's config/cameras.json at
# 640x480: fx = fy = 487.1, giving 66.606 x 52.460 degrees.
#
# focal + sensorsize rather than a bare fovy, for the reason the D435 module gives:
# fovy is vertical only and MuJoCo derives the horizontal from the render aspect, so a
# square render would throw away a third of this camera's width. The focal length is
# nominal; only the ratio to sensorsize matters, and the two FOVs fix that.
KWC500_HFOV_DEG = 66.606
KWC500_VFOV_DEG = 52.460
KWC500_FOCAL = (0.0019, 0.0019)
KWC500_SENSORSIZE = (
  2 * KWC500_FOCAL[0] * math.tan(math.radians(KWC500_HFOV_DEG) / 2),
  2 * KWC500_FOCAL[1] * math.tan(math.radians(KWC500_VFOV_DEG) / 2),
)
# 4:3, as the real camera delivers and as the intrinsics were calibrated at.
KWC500_RESOLUTION = (64, 48)

# The principal point is NOT the image centre on this lens: the fit put it at
# (332.3, 287.1) on a 640x480 frame, i.e. 12.3 px right and 47.1 px down of centre.
# The vertical offset is a tenth of the frame height, which is worth carrying rather
# than rounding to centre.
#
# MuJoCo wants it as pixels from the sensor centre, and its sensor +y points UP while
# an image's +y points DOWN, so the vertical term is negated. Scaled to the render
# size, because principal_pixel is in pixels of the image actually produced.
_CALIB_W, _CALIB_H, _CALIB_CX, _CALIB_CY = 640, 480, 332.3, 287.1
KWC500_PRINCIPAL_PIXEL = (
  (_CALIB_CX - _CALIB_W / 2) * KWC500_RESOLUTION[0] / _CALIB_W,
  -(_CALIB_CY - _CALIB_H / 2) * KWC500_RESOLUTION[1] / _CALIB_H,
)

_MOUNT_MASS = 0.018   # printed PLA bracket
_VISUAL_GROUP = 2


def _quat(rpy_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
  """agent_101's Isaac euler triple (degrees, Rx@Ry@Rz) -> MuJoCo (w, x, y, z).

  The same composition objects.py uses, so the numbers mean here what they mean there.
  """
  from scipy.spatial.transform import Rotation

  x, y, z, w = Rotation.from_euler("XYZ", rpy_deg, degrees=True).as_quat()
  return (float(w), float(x), float(y), float(z))


def get_so101_klip_spec() -> mujoco.MjSpec:
  """The arm, plus the printed bracket and the KWC-500 looking out of it."""
  spec = get_spec()
  gripper = next(b for b in spec.bodies if b.name == "gripper")

  spec.add_mesh(name="klip_support", file=str(_ASSETS / "klip_support.stl"))
  mount = gripper.add_body(
    name="klip_support", pos=_MOUNT_POS, quat=_quat(_MOUNT_RPY_DEG)
  )
  mount.add_geom(
    name="klip_support_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="klip_support",
    mass=_MOUNT_MASS,
    group=_VISUAL_GROUP,
    contype=0,
    conaffinity=0,
    rgba=(0.45, 0.45, 0.47, 1.0),
  )

  # MuJoCo cameras look down their own -z with +y up, which is the same convention the
  # fitted pose was expressed in (Isaac's USD cameras), so the rotation transfers with
  # no flip. Attached to the gripper, beside the bracket.
  gripper.add_camera(
    name="camera_klip",
    pos=_CAM_POS,
    quat=_quat(_CAM_RPY_DEG),
    focal_length=KWC500_FOCAL,
    sensor_size=KWC500_SENSORSIZE,
    principal_pixel=KWC500_PRINCIPAL_PIXEL,
    resolution=KWC500_RESOLUTION,
  )
  return spec
