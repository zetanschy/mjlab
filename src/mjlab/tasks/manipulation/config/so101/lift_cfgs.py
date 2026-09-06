"""Lifting the grey T on the SO-ARM101, from the wrist camera.

mjlab's Mjlab-Lift-Cube-Yam-Rgb, transplanted onto an arm a third of the size and
onto the object agent_101 actually owns. The reward, the command term and the
curriculum are taken unchanged; what had to move is everything encoding how far the
YAM reaches, how wide it opens, and what it is holding.

THE OBJECT IS THE PUSH-T BLOCK, not a cube -- the same get_t_object_spec, the same
grey, the same 105 g. This bench has one printed manipuland and a policy trained
against a different one cannot be run on it. It also costs nothing on the vision
side and gains a lot: the T subtends 86.8 px of the 32x24 frame against a 30 mm
cube's 27.4, and is absent from 2.6% of random-action frames against 15.1%.

WHAT IT COSTS is that a T is not a cube in the way that matters to a gripper. A cube
presents the same 30 mm to the jaw from any yaw; the T presents 25 mm across its stem
or crossbar and 100 mm along them, so with yaw drawn from the full +/-pi the policy
has to bring Wrist_Roll into alignment before it can close on anything. That is a
harder problem than the YAM's cube, and it is the thing to watch if this does not
learn.

WHAT THE CAMERA IS AND IS NOT FOR. In the vision variant the actor loses ee_to_cube
and t_to_goal and has to find the T in the image. It keeps goal_position as state,
and that is not the shortcut it was in Push-T: the push goal was a printed footprint,
a physical thing the camera could see and the policy declined to use. The lift target
is a virtual point in the air. No camera can see it, on this bench or any other, so
state is the only place it can come from.

MEASURED ON THIS ARM, because the YAM's numbers are wrong here by a factor of two:

  THE JAW. Tip separation runs 2.5 mm at the -10 degree limit to 137 mm at +100,
  passing 33 mm at 12 degrees. The T's graspable features -- the stem across its
  25 mm width, the crossbar across its 25 mm depth -- sit comfortably inside that,
  needing about 0.15 rad of the 1.92 rad range.

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

  THE TABLE, and no colour randomization. Black table, grey T, white arm -- the same
  scene Push-T uses, for the same reason: mjlab randomizes its YAM cube's colour over
  the full (0, 1) range, and doing that here would train a policy on an object the
  real camera will never show it. The T is printed in one grey and that is what it
  stays.

  THE MASS. 105 g, mjlab's own figure for this block, kept rather than reduced. A
  solid PLA T of these dimensions is about 108 g and a sparse-infill print nearer 40,
  so this is the heavy end of what the real object can be -- which is the safe
  direction to be wrong in for grip force, if not for droop.
"""

from __future__ import annotations

import functools
from typing import Literal

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
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.tasks.manipulation.push_t_scene import get_t_object_spec

from .env_cfgs import EE_BODY, GRASP_SITE, T_BLOCK_RGBA, _black_table_cfg

OBJECT_NAME = "t_object"

# The same spawn band Push-T uses on this arm, which was chosen from the fingertip
# reach density rather than from the gripper body's: 0.13 +/- 0.03, |y| < 0.06.
SPAWN_X = (0.10, 0.16)
SPAWN_Y = (-0.06, 0.06)

# Lower than the cube version's 0.10-0.20 for two measured reasons: the 0.08-0.14
# band has the better top-down density (48% against 41%), and the T is five times
# the mass, so there is no reason to ask for the extra height.
TARGET_X = (0.10, 0.22)
TARGET_Y = (-0.08, 0.08)
TARGET_Z = (0.08, 0.16)


def so101_lift_t_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """State-based lifting of the grey T on the SO-ARM101."""
  cfg = make_lift_cube_env_cfg()

  cfg.scene.terrain = _black_table_cfg()
  cfg.scene.entities = {
    "robot": get_so101_robot_cfg(spec_fn=get_so101_klip_spec),
    OBJECT_NAME: EntityCfg(
      spec_fn=functools.partial(get_t_object_spec, rgba=T_BLOCK_RGBA)
    ),
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

  # The base config names these terms and params after a cube because it is shared
  # with the YAM task. The keys are left alone -- renaming them would reorder the
  # observation for no gain -- but every object_name is retargeted onto the T.
  actor = cfg.observations["actor"]
  actor.terms["ee_to_cube"].params["asset_cfg"].site_names = GRASP_SITE
  actor.terms["ee_to_cube"].params["object_name"] = OBJECT_NAME
  actor.terms["cube_to_goal"].params["object_name"] = OBJECT_NAME
  cfg.rewards["lift"].params["asset_cfg"].site_names = GRASP_SITE
  for term in ("lift", "lift_precise"):
    cfg.rewards[term].params["object_name"] = OBJECT_NAME
  lift.entity_name = OBJECT_NAME

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


def so101_lift_t_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb", play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Cube lifting through the KWC-500 on the wrist.

  The actor loses the T's pose and reads it off the image. It keeps goal_position,
  which is a virtual point in the air and cannot be seen -- see the module docstring
  for why that is a different situation from Push-T's footprint. The critic keeps
  full state.
  """
  cfg = so101_lift_t_env_cfg(play=play)

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

  # No colour randomization: the T is printed in one grey. See the module
  # docstring -- randomizing it trains on an object the real camera cannot show.

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
