"""Push-T on the SO-ARM101, with the printed bracket and the KWC-500 wrist camera.

The YAM's Mjlab-Push-T-Yam-D435-Push, transplanted onto a much smaller arm. The
objective is taken verbatim -- one maniskill_push_t_reward at weight 3.0 with the
tightened tolerances, the precision bonus beside it, a gripper-open penalty to keep it
a pushing problem, and the three penalties -- because that chain is the one with
results behind it and changing two things at once would waste them.

WHAT HAD TO CHANGE, and all of it is the arm being a third of the size:

  THE WORKSPACE. The YAM spawns the block 0.28 m out and puts the goal at 0.40. This
  arm's gripper cannot reach 0.40 m at all. Sampling its joint space at block height
  (400k configurations, inside the URDF's real limits) gives a forward extent of
  0.343 m and a lateral reach that collapses as it extends: +/-0.34 m at 0.10-0.15 m
  out, +/-0.18 m at 0.30-0.35 m. So the block spawns at 0.18 m and the goal sits at
  0.28 m, both with lateral room to spare -- the same 10 cm inward shift mjlab made
  for the D435, for the same reason, arrived at from this arm's own numbers.

  THE ACTION SCALE. 0.3 rad, not the YAM's 0.8. Measured on this arm in agent_101: at
  0.8 the gripper travels 46 mm per control step, 1.4 m/s, and spends 30% of steps
  below the table surface; at 0.3 it is 20 mm, about the width of the T's stem. mjlab
  raised the YAM to 0.8 to make the block's SIDE reachable at all; this arm starts its
  episode 40 mm above the table with the block beside it, so it has the opposite
  problem.

  THE HOME POSE. Solved against this arm's own FK inside its joint limits: gripper 40
  mm above the table, 16 cm to the side of the spawn, 1.01 rad of margin to the
  nearest limit. Clear of the block, because a home directly over it rests the jaws on
  the task object.

  THE CAMERA. A Klip Xtreme KWC-500 in a printed klip_support bracket, at the pose
  fitted against real photographs in agent_101 (3.1 px of silhouette error), with its
  measured intrinsics -- 66.6 x 52.5 degrees and an off-centre principal point. Not a
  D435 and not on a D435's bracket.
"""

from __future__ import annotations

import dataclasses
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
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.push_t_env_cfg import make_push_t_env_cfg, make_push_t_metrics
from mjlab.tasks.velocity import mdp as velocity_mdp

# Measured on this arm, not inherited. See the module docstring.
T_SPAWN_X = 0.18
GOAL_X = 0.28
# Disjoint in x by 5 cm however the draws fall, which is mjlab's own rule: overlap and
# a fraction of episodes start already solved and the reward pays for the reset draw.
SPAWN_X_RANGE = (-0.03, 0.03)
SPAWN_Y_RANGE = (-0.06, 0.06)
GOAL_X_RANGE = (-0.02, 0.02)
# Narrower than the YAM's +/-0.10: past about 6 cm of lateral offset the footprint
# leaves this camera's frame, which is the failure mjlab measured on its own D435
# variant -- the goal rendered at 0 px and coverage above 0.90 fell from 62% to 19%.
GOAL_Y_RANGE = (-0.05, 0.05)

# The block is off the table long before the arm could have followed it here.
WORKSPACE_X = (0.06, 0.40)
WORKSPACE_Y = (-0.25, 0.25)

# The wrist body, for the ground-contact termination and the viewer.
EE_BODY = "gripper"
GRASP_SITE = ("grasp_site",)


def _so101_scene(cfg: ManagerBasedRlEnvCfg) -> None:
  """Swap the YAM out of the base scene and move the task in to reach."""
  cfg.scene.entities = dict(cfg.scene.entities)
  cfg.scene.entities["robot"] = get_so101_robot_cfg(spec_fn=get_so101_klip_spec)
  cfg.scene.entities["t_object"] = dataclasses.replace(
    cfg.scene.entities["t_object"],
    init_state=EntityCfg.InitialStateCfg(pos=(T_SPAWN_X, 0.0, 0.0)),
  )
  cfg.scene.entities["t_goal"] = dataclasses.replace(
    cfg.scene.entities["t_goal"],
    init_state=EntityCfg.InitialStateCfg(pos=(GOAL_X, 0.0, 0.0)),
  )


def so101_push_t_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """State-based Push-T on the SO-ARM101."""
  cfg = make_push_t_env_cfg()
  _so101_scene(cfg)

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      # The jaw gets its own, small: this is a pushing task, and the only thing the
      # policy needs the jaw for is to keep it shut.
      scale={
        "(Rotation|Pitch|Elbow|Wrist_Pitch|Wrist_Roll)": SO101_ACTION_SCALE,
        "Jaw": 0.05,
      },
      use_default_offset=True,
    )
  }

  # The reward set the YAM chain actually ended on: one dense term, the fine-scale
  # bonus, and three penalties. The base config's additive position/orientation/
  # coverage terms are replaced, not extended.
  cfg.rewards = {
    "maniskill": RewardTermCfg(
      func=manipulation_mdp.maniskill_push_t_reward,
      weight=3.0,
      params={
        "object_name": "t_object",
        "goal_name": "t_goal",
        "success_mode": "pose",
        "pos_tol": 0.010,
        "yaw_tol_deg": 8.0,
        "asset_cfg": SceneEntityCfg("robot", site_names=GRASP_SITE),
      },
    ),
    "precision": RewardTermCfg(
      func=manipulation_mdp.push_precision_bonus,
      weight=2.0,
      params={
        "object_name": "t_object",
        "goal_name": "t_goal",
        "pos_scale": 20.0,
        "yaw_scale": 3.0,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=velocity_mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=velocity_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.01,
      params={"max_vel": 0.5, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }
  cfg.metrics = make_push_t_metrics()
  cfg.curriculum = {}

  cfg.observations["actor"].terms["ee_to_t"].params["asset_cfg"].site_names = GRASP_SITE

  cfg.events["reset_t_object"].params["pose_range"] = {
    "x": SPAWN_X_RANGE, "y": SPAWN_Y_RANGE, "yaw": (-3.14, 3.14),
  }
  cfg.events["reset_t_goal"].params["pose_range"] = {
    "x": GOAL_X_RANGE, "y": GOAL_Y_RANGE, "yaw": (-3.14, 3.14),
  }
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = (
    SO101_FINGERTIP_GEOMS,
  )

  cfg.terminations["t_out_of_bounds"].params["x_range"] = WORKSPACE_X
  cfg.terminations["t_out_of_bounds"].params["y_range"] = WORKSPACE_Y

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if isinstance(sensor, ContactSensorCfg) and sensor.name == "ee_ground_collision":
      sensor.primary.pattern = EE_BODY

  cfg.viewer.body_name = EE_BODY

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg


def so101_push_t_camera_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb", play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Push-T through the KWC-500 on the wrist.

  The actor loses the block's pose and reads it off the image; it keeps goal_pose as
  state, because the footprint is 1 mm tall and lives in geom group 2, which the camera
  does not render. The critic keeps full state.

  64x48, the render size agent_101's Isaac task uses, and 4:3 because that is the
  aspect the intrinsics were calibrated at -- a square render against a 4:3 lens throws
  away a quarter of its width.
  """
  cfg = so101_push_t_env_cfg(play=play)

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
  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "camera_klip_rgb": ObservationTermCfg(
        func=manipulation_mdp.camera_rgb, params={"sensor_name": "camera_klip"}
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
  )

  actor = cfg.observations["actor"]
  actor.terms.pop("ee_to_t", None)
  actor.terms.pop("t_to_goal", None)
  actor.terms["goal_pose"] = ObservationTermCfg(
    func=manipulation_mdp.goal_pose_in_base, params={"goal_name": "t_goal"}
  )
  return cfg


def so101_push_t_push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The camera task with the jaw penalized for being open, so it pushes.

  mjlab's reason, on the YAM: left free, the policy holds the gripper open and works
  the block with the two fingers as a fork. It scores better than pushing and
  transfers worse.

  The SO-101 has one jaw joint rather than two linear fingers, so travel_ref is its
  range in radians (-0.175 to 1.745) and closed_tol is measured from shut.
  """
  cfg = so101_push_t_camera_env_cfg(play=play)
  cfg.rewards["gripper_open"] = RewardTermCfg(
    func=manipulation_mdp.gripper_open_penalty,
    weight=-2.0,
    params={
      "closed_tol": 0.05,
      "travel_ref": 1.92,
      "asset_cfg": SceneEntityCfg("robot", joint_names=("Jaw",)),
    },
  )
  return cfg
