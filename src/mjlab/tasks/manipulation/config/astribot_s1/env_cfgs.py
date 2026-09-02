"""Push-T on the Astribot S1 humanoid, seen through its head camera."""

import dataclasses

from mjlab.asset_zoo.robots.astribot_s1 import (
  ARM_JOINTS,
  EE_SITE,
  HEAD_CAMERA,
  TABLE_HEIGHT,
  WORKSPACE_CENTRE,
  get_astribot_s1_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.push_t_env_cfg import (
  make_push_t_env_cfg,
  make_push_t_metrics,
)
from mjlab.tasks.velocity import mdp as velocity_mdp

# The robot faces +x across the table, so +x is "away from the robot": the block
# starts nearer and is pushed outward. Both sit on the line the head camera
# looks at, whose framing sets how much room there is -- see the home-pose notes
# in the robot module.
BLOCK_CENTRE = (WORKSPACE_CENTRE[0] - 0.045, WORKSPACE_CENTRE[1], TABLE_HEIGHT)
GOAL_CENTRE = (WORKSPACE_CENTRE[0] + 0.045, WORKSPACE_CENTRE[1], TABLE_HEIGHT)


def s1_push_t_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Push-T on the S1, rewards carried over from the YAM camera task.

  The reward, the success predicate and the fine precision term are the same ones
  verified at 99.0% / 3.7 mm / 2.2 deg on the YAM, with two things adapted to the
  different embodiment.

  The working surface is the scene's own simpleTable, top face at z = 0.76, so
  everything that tests whether the block is flat takes ``table_height``.
  Without it the flatness conjunct in the success predicate compares the block's
  z against the floor and can never be satisfied.

  Only the right arm and its gripper are actuated. The head camera sits on a
  2-DoF neck above a 4-DoF torso; actuating either moves the camera relative to
  the workspace, which is a materially harder perception problem than the
  wrist-camera tasks this reward comes from. Torso and head hold the home pose.
  """
  cfg = make_push_t_env_cfg()

  # The shared Push-T config runs at dt=0.005 with decimation 4, which is fine
  # for the 6-DoF YAM but blows this model up: at 0.005 the S1's 25 joints and
  # tendon-driven grippers produced NaN in 1660 of 3072 sampled environments,
  # and at 0.002 in none. Decimation rises to 10 to keep the control rate at
  # 50 Hz, so an episode is still 1000 steps.
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10

  # env_spacing 0 so every world coincides. The scene's simpleTable is static
  # geometry, shared across worlds rather than replicated, so a spatial grid
  # would leave all but one environment's robot reaching at empty air. MuJoCo
  # Warp worlds are independent, so the overlap is only a visual one.
  cfg.scene.env_spacing = 0.0
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.env_spacing = 0.0

  cfg.scene.entities = {
    "robot": get_astribot_s1_cfg(),
    "t_object": dataclasses.replace(
      cfg.scene.entities["t_object"],
      init_state=EntityCfg.InitialStateCfg(pos=BLOCK_CENTRE),
    ),
    "t_goal": dataclasses.replace(
      cfg.scene.entities["t_goal"],
      init_state=EntityCfg.InitialStateCfg(pos=GOAL_CENTRE),
    ),
  }
  cfg.scene.sensors = ()

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=tuple(ARM_JOINTS),
      scale=0.3,
      use_default_offset=True,
    )
  }

  # Block and goal sit symmetrically about the point the camera looks at, and
  # the spawn boxes are sized so that a T at any yaw stays inside the frame: a
  # 0.1 m T reaches ~0.07 m from its own centre when turned 45 deg, and the
  # home pose was chosen for a 0.28 x 0.20 m area, so the centres get +/-0.025.
  cfg.events["reset_t_object"].params["pose_range"] = {
    "x": (-0.025, 0.025),
    "y": (-0.025, 0.025),
    "yaw": (-3.14, 3.14),
  }
  cfg.events["reset_t_goal"].params["pose_range"] = {
    "x": (-0.025, 0.025),
    "y": (-0.025, 0.025),
    "yaw": (-3.14, 3.14),
  }
  cfg.events.pop("fingertip_friction_slide", None)

  # Only the right arm is driven by the policy, but every servo in the scene is
  # wrapped, so the untouched ones need a target to hold. Without this the
  # torso and head servos are commanded to zero and unfold the pose the head
  # camera's view depends on.
  cfg.events["hold_posture"] = EventTermCfg(
    func=manipulation_mdp.hold_default_joint_targets,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  ee = SceneEntityCfg("robot", site_names=(EE_SITE,))
  # "Just the top camera as input": the actor keeps proprioception but is told
  # nothing about either the block or the goal in state form. Both are painted
  # into the head camera's view -- the T in yellow, the footprint in green --
  # so the policy has to read them off the image. The critic keeps the
  # privileged terms below, which is the usual asymmetric actor-critic split.
  cfg.observations["actor"].terms.pop("ee_to_t", None)
  cfg.observations["actor"].terms.pop("t_to_goal", None)
  cfg.observations["actor"].terms.pop("goal_pose", None)
  cfg.observations["actor"].terms.pop("object_pose", None)
  cfg.observations["critic"].terms["ee_to_t"] = ObservationTermCfg(
    func=manipulation_mdp.ee_to_object_planar,
    params={"object_name": "t_object", "asset_cfg": ee},
  )
  cfg.observations["critic"].terms["t_to_goal"] = ObservationTermCfg(
    func=manipulation_mdp.object_to_goal_pose_error,
    params={"object_name": "t_object", "goal_name": "t_goal"},
  )

  # Proprioception over the right arm only. The other 28 joints -- chassis,
  # torso, head, left arm, gripper fingers -- are held at their defaults for the
  # whole episode, so reporting them would pad both observations with 56
  # constant channels for the normalizer to divide by a near-zero std.
  arm = SceneEntityCfg("robot", joint_names=tuple(ARM_JOINTS))
  for group in ("actor", "critic"):
    for term in ("joint_pos", "joint_vel"):
      cfg.observations[group].terms[term].params["asset_cfg"] = arm

  cam = CameraSensorCfg(
    name="head_cam",
    camera_name=f"robot/{HEAD_CAMERA}",
    width=96,
    height=72,
    data_types=("rgb",),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cam,)
  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "head_rgb": ObservationTermCfg(
        func=manipulation_mdp.camera_rgb, params={"sensor_name": "head_cam"}
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
  )

  cfg.rewards = {
    "maniskill": RewardTermCfg(
      func=manipulation_mdp.maniskill_push_t_reward,
      weight=3.0,
      params={
        "object_name": "t_object",
        "goal_name": "t_goal",
        "success_mode": "pose",
        "pos_tol": 0.02,
        "yaw_tol_deg": 15.0,
        "table_height": TABLE_HEIGHT,
        "asset_cfg": ee,
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
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=tuple(ARM_JOINTS))},
    ),
    "joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.01,
      params={
        "max_vel": 1.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=tuple(ARM_JOINTS)),
      },
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
    "t_out_of_bounds": TerminationTermCfg(
      func=manipulation_mdp.object_out_of_bounds,
      params={
        "object_name": "t_object",
        "x_range": (WORKSPACE_CENTRE[0] - 0.25, WORKSPACE_CENTRE[0] + 0.25),
        "y_range": (WORKSPACE_CENTRE[1] - 0.25, WORKSPACE_CENTRE[1] + 0.25),
      },
    ),
  }

  metrics = make_push_t_metrics()
  for name in ("success_pose", "block_height_mm"):
    metrics[name].params["table_height"] = TABLE_HEIGHT
  cfg.metrics = metrics
  cfg.curriculum = {}
  cfg.viewer.entity_name = "robot"
  cfg.viewer.body_name = "astribot_torso_link_4"

  if play:
    cfg.observations["actor"].enable_corruption = False
    # Let the viewer run without the episode timer cutting it short; resets
    # then come only from the termination terms.
    cfg.episode_length_s = int(1e9)
  return cfg
