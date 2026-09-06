"""Cube lifting on the SO-ARM101, from the wrist camera.

mjlab's Mjlab-Lift-Cube-Yam-Rgb, transplanted onto an arm a third of the size. The
reward, the command term and the curriculum are taken unchanged; what had to move is
everything that encodes how far the YAM can reach and how wide it opens.

WHAT THE CAMERA IS AND IS NOT FOR. In the vision variant the actor loses ee_to_cube
and cube_to_goal and has to find the CUBE in the image. It keeps goal_position as
state, and that is not the shortcut it was in Push-T: the push goal was a printed
footprint, a physical thing the camera could see and the policy declined to use. The
lift target is a virtual point in the air. No camera can see it, on this bench or any
other, so state is the only place it can come from.

MEASURED ON THIS ARM, because the YAM's numbers are wrong here by a factor of two:

  THE JAW. Tip separation runs 2.5 mm at the -10 degree limit to 137 mm at +100,
  passing 33 mm at 12 degrees. So the jaw does not constrain the cube size at all;
  the camera does, and CUBE_HALF below carries that measurement.

  THE JAW'S ACTION SCALE. 0.5, the same as the arm joints, where Push-T throttled it
  to 0.05. That was right there and would be fatal here: Push-T wanted the jaw shut
  and nothing else, this task cannot be done without opening it. At 0.5 the policy
  can command -0.65 to +0.35 rad about the home pose, which clips to the jaw's own
  limits and spans closed to roughly 60 mm of opening.

  THE WORKSPACE. Sampling 300k configurations inside the URDF's real joint limits and
  keeping those whose gripper points at the table (the body's -z axis, which is where
  GRASP_SITE_POS lies, within 60 degrees of straight down):

      cube spawn   x 0.10-0.20, |y|<0.08, on the table   844 reachable, 67% top-down
      lift target  x 0.10-0.22, |y|<0.08, z 0.10-0.20   1553 reachable, 41% top-down
      YAM default  x 0.30-0.50, |y|<0.20, z 0.20-0.40   8073 reachable,  4% top-down

  The YAM's default target box is not merely far -- 96% of the ways this arm can put
  its grasp site there have the gripper pointing somewhere other than down, which is
  no use for holding a cube. Raising the target above 0.20 m costs the same way: the
  0.20-0.30 band drops to 18%.

  THE TABLE. Black, as in Push-T and as on agent_101's bench, and that constrains the
  cube's colour randomization: mjlab draws the YAM's cube from the full (0, 1) cube
  against a white table, which here would regularly produce a black cube on a black
  table. Drawn from (0.25, 0.85) instead, which stays clear of the table's 0.03 and
  the arm's 0.93.
"""

from __future__ import annotations

from typing import Literal

import mujoco

from mjlab.asset_zoo.robots.so_arm101.klip_camera import (
  KWC500_RESOLUTION,
  get_so101_klip_spec,
)
from mjlab.asset_zoo.robots.so_arm101.so101_constants import (
  SO101_ACTION_SCALE,
  SO101_FINGERTIP_GEOMS,
  get_so101_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

from .env_cfgs import EE_BODY, GRASP_SITE, _black_table_cfg

# 30 mm, and the size is a compromise between the jaw and the camera rather than a
# round number. The jaw allows anything up to about 60 mm, so grasping does not
# constrain it; the 32x24 frame does. Segmentation over 1024 random-action frames:
#
#     25 mm   17.3 px (2.2%)   median 12   absent 21.3% of frames
#     30 mm   27.4 px (3.6%)   median 21   absent 15.1%
#     35 mm   34.2 px (4.4%)   median 23   absent 13.6%
#
# 30 mm buys 58% more pixels than 25 for 0.19 rad of jaw against 0.15, and stops
# short of 35, where the cube approaches the length of the fingers themselves and
# the grasp gets marginal. 27 px is also, for scale, about what the Push-T footprint
# subtends -- and that one was never learned, which is the risk this task carries.
# The mitigation is not pixels: it is that the lift reward is dense in ee-to-cube
# distance, so there is gradient toward the cube before it is resolvable.
#
# Mass is a printed PLA cube at the infill agent_101 prints at, not the YAM's 50 g --
# these are 7.4 kg-cm servos.
CUBE_HALF = 0.015
CUBE_MASS = 0.02
CUBE_RGBA = (0.55, 0.55, 0.57, 1.0)  # the grey the T is printed in, as a default

# Both boxes are the measured top-down-reachable regions, not the YAM's.
SPAWN_X = (0.10, 0.20)
SPAWN_Y = (-0.08, 0.08)
TARGET_X = (0.10, 0.22)
TARGET_Y = (-0.08, 0.08)
TARGET_Z = (0.10, 0.20)

# Clear of the black table (0.03) and the white arm (0.93) at both ends.
CUBE_COLOR_RANGE = (0.25, 0.85)


def get_so101_cube_spec(
  cube_size: float = CUBE_HALF,
  mass: float = CUBE_MASS,
  rgba: tuple[float, float, float, float] = CUBE_RGBA,
) -> mujoco.MjSpec:
  """A cube sized for this jaw. Group 3 so the wrist camera renders it.

  mjlab's YAM cube leaves the geom on the default group 0, which the camera also
  renders, so it is visible either way -- but group 3 is where this scene puts
  everything the camera is meant to see, and matching it keeps one rule.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
    group=3,
  )
  return spec


def so101_lift_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """State-based cube lifting on the SO-ARM101."""
  cfg = make_lift_cube_env_cfg()

  cfg.scene.terrain = _black_table_cfg()
  cfg.scene.entities = {
    "robot": get_so101_robot_cfg(spec_fn=get_so101_klip_spec),
    "cube": EntityCfg(spec_fn=get_so101_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # The jaw is NOT throttled here; see the module docstring.
  joint_pos_action.scale = SO101_ACTION_SCALE

  assert cfg.commands is not None
  lift = cfg.commands["lift_height"]
  assert isinstance(lift, LiftingCommandCfg)
  lift.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=SPAWN_X, y=SPAWN_Y, z=(0.02, 0.04), yaw=(-3.14, 3.14)
  )
  lift.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=TARGET_X, y=TARGET_Y, z=TARGET_Z
  )

  cfg.observations["actor"].terms["ee_to_cube"].params[
    "asset_cfg"
  ].site_names = GRASP_SITE
  cfg.rewards["lift"].params["asset_cfg"].site_names = GRASP_SITE

  for term in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events[term].params["asset_cfg"].geom_names = (SO101_FINGERTIP_GEOMS,)

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = EE_BODY

  cfg.viewer.body_name = EE_BODY

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def so101_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb", play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Cube lifting through the KWC-500 on the wrist.

  The actor loses the cube's pose and reads it off the image. It keeps
  goal_position, which is a virtual point in the air and cannot be seen -- see the
  module docstring for why that is a different situation from Push-T's footprint.
  The critic keeps full state.
  """
  cfg = so101_lift_cube_env_cfg(play=play)

  cam_cfg = CameraSensorCfg(
    name="camera_klip",
    camera_name="robot/camera_klip",
    width=KWC500_RESOLUTION[0],
    height=KWC500_RESOLUTION[1],
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  if cam_type == "depth":
    func = manipulation_mdp.camera_depth
    params: dict = {"sensor_name": cam_cfg.name, "cutoff_distance": 0.5}
  else:
    func = manipulation_mdp.camera_rgb
    params = {"sensor_name": cam_cfg.name}
  cfg.observations["camera"] = ObservationGroupCfg(
    terms={f"camera_klip_{cam_type}": ObservationTermCfg(func=func, params=params)},
    enable_corruption=False,
    concatenate_terms=True,
  )

  if cam_type == "rgb":
    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": CUBE_COLOR_RANGE,
      },
    )

  actor = cfg.observations["actor"]
  actor.terms.pop("ee_to_cube", None)
  actor.terms.pop("cube_to_goal", None)
  actor.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=GRASP_SITE),
    },
    # NOTE: no noise on the goal position, as in mjlab's YAM variant.
  )
  return cfg
