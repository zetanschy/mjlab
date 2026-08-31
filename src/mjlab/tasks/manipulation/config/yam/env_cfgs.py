import colorsys
from typing import Any, Literal

import mujoco

from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  RelativeJointPositionActionCfg,
)
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import MultiCubeLiftingCommandCfg
from mjlab.tasks.manipulation.push_t_env_cfg import (
  make_push_t_env_cfg,
  make_push_t_metrics,
)
from mjlab.tasks.manipulation.push_t_scene import get_yam_gravcomp_robot_cfg
from mjlab.tasks.velocity import mdp as velocity_mdp


def get_cube_spec(
  cube_size: float = 0.02,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def yam_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    # Higher command resampling frequency for more dynamic play.
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def yam_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = yam_lift_cube_env_cfg(play=play)

  camera_names = ["robot/camera_d405"]
  cam_kwargs = {
    "robot/camera_d405": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
      **shared_cam_kwargs,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs
    )

  camera_obs = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )
  cfg.observations["camera"] = camera_obs

  if cam_type == "rgb":
    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.0, 1.0),
      },
    )

  # Pop privileged info from actor observations.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")

  # Add goal_position to actor observations.
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
    # NOTE: No noise for goal position.
  )

  return cfg


def _cube_color(i: int, n: int) -> tuple[float, float, float, float]:
  """Generate a distinct color for cube i of n using HSV hue rotation."""
  h = i / max(n, 1)
  r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
  return (r, g, b, 1.0)


def yam_multi_cube_seg_env_cfg(
  num_cubes: int = 3,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Multi-cube task: depth + segmentation mask for goal conditioning."""
  cfg = make_lift_cube_env_cfg()

  cube_names = [f"cube_{i}" for i in range(num_cubes)]
  entities: dict[str, EntityCfg] = {"robot": get_yam_robot_cfg()}
  for i, name in enumerate(cube_names):
    color = _cube_color(i, num_cubes)
    entities[name] = EntityCfg(
      spec_fn=lambda c=color: get_cube_spec(rgba=c),
    )
  cfg.scene.entities = entities

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.commands = {
    "lift_height": MultiCubeLiftingCommandCfg(
      entity_names=tuple(cube_names),
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
    ),
  }

  cfg.rewards["lift"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_staged_position_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "reaching_std": 0.2,
      "bringing_std": 0.3,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )
  cfg.rewards["lift_precise"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_bring_object_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "std": 0.05,
    },
  )

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 55 + num_cubes * 120)

  cam_cfg = CameraSensorCfg(
    name="camera_d405",
    camera_name="robot/camera_d405",
    height=32,
    width=32,
    data_types=("depth", "segmentation"),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  cam_terms = {
    "depth": ObservationTermCfg(
      func=manipulation_mdp.camera_depth,
      params={
        "sensor_name": "camera_d405",
        "cutoff_distance": 0.5,
      },
    ),
    "target_mask": ObservationTermCfg(
      func=manipulation_mdp.camera_target_cube_mask,
      params={
        "sensor_name": "camera_d405",
        "command_name": "lift_height",
      },
    ),
  }
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms,
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  for group_name in ("actor", "critic"):
    obs = cfg.observations[group_name]
    obs.terms.pop("ee_to_cube", None)
    obs.terms.pop("cube_to_goal", None)
    obs.terms["goal_position"] = ObservationTermCfg(
      func=manipulation_mdp.target_position,
      params={
        "command_name": "lift_height",
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (
      4.0,
      4.0,
    )

  return cfg


def yam_push_t_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """State-based Push-T on the YAM arm."""
  cfg = make_push_t_env_cfg()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.observations["actor"].terms["ee_to_t"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["ee_guidance"].params["asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg


def yam_push_t_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "depth",
  visual_goal: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Push-T from the wrist camera.

  The actor loses the block's pose and must read it off the image. It keeps the
  goal pose as state: the footprint is 1 mm tall and lives in geom group 2,
  which the camera does not render, so it is not visible in depth at all. The
  critic keeps full state (asymmetric actor-critic).
  """
  cfg = yam_push_t_env_cfg(play=play)

  cam_cfg = CameraSensorCfg(
    name="camera_d405",
    camera_name="robot/camera_d405",
    height=32,
    width=32,
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
  if cam_type == "depth":
    param_kwargs["cutoff_distance"] = 0.6
    func = manipulation_mdp.camera_depth
  else:
    func = manipulation_mdp.camera_rgb

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      f"camera_d405_{cam_type}": ObservationTermCfg(func=func, params=param_kwargs)
    },
    enable_corruption=False,
    concatenate_terms=True,
  )

  if cam_type == "rgb":
    # Capped below 1.0 on purpose. The lift-cube task randomizes the full
    # (0, 1) cube, but it sits on the default dark checker floor. This table is
    # white, so a full-range sample can put a near-white block on a near-white
    # background: ~9% of episodes would land above 0.8 luminance and ~1.5%
    # would be all but invisible. 0.7 keeps every sample separable.
    # shared_random keeps the block one colour. Without it every geom draws its
    # own sample and the block renders two-toned: crossbar and stem in
    # unrelated hues. The lift-cube task never hit this because its cube is a
    # single geom.
    #
    # With a visible goal the block must also stay distinguishable from the
    # green footprint, so channels are sampled per-axis rather than over the
    # full colour cube. Separation rides on red: the block never drops below
    # 0.55 while the footprint sits at 0.15, which leaves green free to range
    # high. That keeps the block's original yellow (1.0, 0.85, 0.1) in
    # distribution -- capping green would have excluded it -- while holding the
    # closest reachable colour 0.40 away from the footprint's green.
    color_ranges: Any = (
      {0: (0.55, 1.0), 1: (0.0, 0.9), 2: (0.0, 0.35)} if visual_goal else (0.0, 0.7)
    )
    cfg.events["t_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("t_object", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": color_ranges,
        "shared_random": True,
      },
    )

  # Drop privileged block state from the actor; it comes from the camera now.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_t")
  actor_obs.terms.pop("t_to_goal")

  if visual_goal:
    if cam_type != "rgb":
      raise ValueError(
        "visual_goal requires cam_type='rgb'. The footprint is 1 mm tall, which "
        "is far below what the depth cutoff resolves, so it cannot be seen in "
        "depth even though it is rendered."
      )
    # The goal is read off the image, so it can move: randomizing it is the
    # whole point, otherwise the policy just memorizes one location.
    cfg.events["reset_t_goal"].params["pose_range"] = {
      "x": (-0.02, 0.02),
      "y": (-0.10, 0.10),
      "yaw": (-3.14, 3.14),
    }
  else:
    # The footprint is not legible to the policy, so hand it the target pose.
    actor_obs.terms["goal_pose"] = ObservationTermCfg(
      func=manipulation_mdp.goal_pose_in_base,
      params={"goal_name": "t_goal"},
      # NOTE: No noise on the goal pose.
    )

  return cfg


def yam_push_t_maniskill_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Fixed-goal RGB Push-T scored by ManiSkill's reward, unmodified.

  A control for the tuned variants. The reward here is one term reproducing
  ManiSkill's, with no coverage shaping, no gating, and none of the action or
  joint penalties the other tasks carry. If this learns and the tuned tasks do
  not, the additions are at fault; if neither learns, the difference is the
  robot -- a 7-DoF arm under joint-position control with a parallel-jaw
  gripper, against ManiSkill's end-effector-controlled rod.
  """
  cfg = yam_push_t_vision_env_cfg(cam_type="rgb", visual_goal=False, play=play)

  cfg.rewards = {
    "maniskill": RewardTermCfg(
      func=manipulation_mdp.maniskill_push_t_reward,
      weight=1.0,
      params={
        "object_name": "t_object",
        "goal_name": "t_goal",
        "success_threshold": 0.90,
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
    ),
  }
  # The curriculum ramps joint_vel_hinge, which no longer exists.
  cfg.curriculum = {}
  return cfg


def yam_push_t_hybrid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """ManiSkill's task reward plus the penalties this arm needs to stay stable.

  Run at 3.0 rather than the normalized 1.0 so it keeps ManiSkill's native
  scale, which is what the penalty weights below were tuned against.

  ManiSkill carries no action, joint-limit or joint-velocity penalties: its
  agent is a rod under end-effector control, so there is nothing to stabilize.
  Reproduced verbatim on a 7-DoF arm under joint-position control it flails --
  driving into the table on 58% of episodes and knocking the block out of the
  workspace on 25%, with reward peaking at iteration 1626 and decaying after.
  The three penalties here are the ones that held both at 0% in the tuned runs.

  No curriculum. The tuned tasks ramp joint_vel_hinge to -1.0, which is a
  minority of their ~4.3 reward span but would be a third of this one, and
  pushing is a task that needs the arm to keep moving.
  """
  cfg = yam_push_t_vision_env_cfg(cam_type="rgb", visual_goal=False, play=play)

  cfg.rewards = {
    "maniskill": RewardTermCfg(
      func=manipulation_mdp.maniskill_push_t_reward,
      weight=3.0,
      params={
        "object_name": "t_object",
        "goal_name": "t_goal",
        "success_threshold": 0.90,
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
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
      params={
        "max_vel": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }
  cfg.curriculum = {}
  return cfg


def yam_push_t_replica_env_cfg(
  relative_action: bool = False,
  gravcomp: bool = False,
  episode_length_s: float = 20.0,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """State-based Push-T with the MDP defects fixed. Phase 1 of the redesign.

  Forked from the STATE task, not a vision one, on purpose: no completed run
  has ever trained the state variant (both attempts stopped at 9 and 11
  iterations), so whether a 32x32 wrist camera can even resolve the T's yaw
  sign is untested. Settling that costs a fast run, not a 3.5 hour one.

  Four changes, each measured rather than assumed:

  Nothing about the robot changes by default. Two options exist for ablation and
  are coupled, not independent: a relative joint action has no position holding
  (its zero-action target is wherever the arm already is, so gravity drags the
  target down with it -- measured 288 mm of sag in 2 s against the absolute
  term's 98 mm steady-state), and it is therefore only usable with gravcomp on.
  Gravcomp itself cancels gravity on every link, which is a change to the arm's
  dynamics whose faithfulness depends on whether the real controller compensates
  gravity; the YAM actuators here are a pure PD servo with no gravity
  feedforward. Neither is on by default, because droop is not established as a
  blocker: the lift-cube policy reaches 93.8% within 5 cm on this arm with the
  full 98 mm of it.

  uniform action scale -- the stock scales span 63.8x (joint2 0.16 rad, joint6
    5.52 rad), so joint6 samples routinely breached a -10.0 soft limit penalty
    worth ~70% of the task reward. Policy/mean_std collapsed 0.9995 -> 0.0353 by
    iteration 771 and spent 32.1% of run 7 below 0.10. A uniform 0.15 rad
    removes the disparity while keeping the same authority: measured
    end-effector speed 114 mm/s against the stock config's 115 mm/s.

  reachable success -- coverage >= 0.90 needs |dyaw| <= 4.6 deg AND |d| <= 2.9
    mm and has never once fired. 2 cm / 15 deg, conjoined with flatness so it
    cannot be won by lifting.

  horizon matched to the episode -- 200 steps against gamma=0.995 in the runner,
    a ratio of 1.0 where run 7 was at 0.10.
  """
  cfg = yam_push_t_env_cfg(play=play)

  if gravcomp:
    cfg.scene.entities = dict(cfg.scene.entities)
    cfg.scene.entities["robot"] = get_yam_gravcomp_robot_cfg()

  if relative_action:
    if not gravcomp:
      raise ValueError(
        "relative_action requires gravcomp: the relative term re-anchors its "
        "target to the measured position every substep, so with gravity on and "
        "zero action the arm sags 288 mm in 2 s instead of holding its pose."
      )
    cfg.actions = {
      "joint_pos": RelativeJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale={"joint[1-6]": 0.1, "left_finger": 0.01},
      )
    }
  else:
    pass  # Keep the stock action from yam_push_t_env_cfg.

  cfg.rewards = {
    "staged_pose": RewardTermCfg(
      func=manipulation_mdp.staged_pose_reward,
      weight=2.0,
      params={"object_name": "t_object", "goal_name": "t_goal", "yaw_weight": 1.0},
    ),
    "ee_guidance": RewardTermCfg(
      func=manipulation_mdp.push_ee_guidance_reward,
      weight=1.0,
      params={
        "object_name": "t_object",
        "goal_name": "t_goal",
        "gate_distance": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
    ),
    "success": RewardTermCfg(
      func=manipulation_mdp.push_coverage_success,
      weight=2.0,
      params={"object_name": "t_object", "goal_name": "t_goal", "threshold": 0.70},
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
      params={
        "max_vel": 1.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  cfg.metrics = make_push_t_metrics()
  cfg.curriculum = {}
  # 20 s (1000 steps) by default, matching the hybrid run. The 4 s / 200-step
  # variant was tried alongside the reward change and the block stopped moving
  # entirely -- travel fell from the hybrid's 0.138 m back to the untracked
  # 0.068 m of an untrained policy, with mean_std collapsing to 0.021 by
  # iteration 600. Pushing was a solved skill and 200 steps was not enough
  # exploration per episode to rediscover it.
  cfg.episode_length_s = episode_length_s
  return cfg


def yam_push_t_reachable_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Camera twin of the working state task. Same MDP, RGB instead of state.

  Built from yam_push_t_reachable_state_env_cfg so it inherits the two fixes
  that made the task solvable at all -- the 0.8 rad uniform action scale that
  puts the block's side within reach, and the reachable success predicate.
  Built the other way round it would silently keep the stock scales, under
  which side contact is impossible and every run fails identically.

  The only difference from the state task is the observation: the actor loses
  ee_to_t and t_to_goal and gets a 32x32 RGB wrist frame instead, keeping the
  goal pose as state because the footprint is 1 mm tall. The critic keeps full
  state. With a 96.6% state policy as the control, this is a clean test of
  whether that camera can resolve the T's yaw -- the last untested question.
  """
  cfg = yam_push_t_reachable_state_env_cfg(play=play)

  cam_cfg = CameraSensorCfg(
    name="camera_d405",
    camera_name="robot/camera_d405",
    height=32,
    width=32,
    data_types=("rgb",),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "camera_d405_rgb": ObservationTermCfg(
        func=manipulation_mdp.camera_rgb, params={"sensor_name": "camera_d405"}
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
  )

  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_t", None)
  actor_obs.terms.pop("t_to_goal", None)
  actor_obs.terms["goal_pose"] = ObservationTermCfg(
    func=manipulation_mdp.goal_pose_in_base,
    params={"goal_name": "t_goal"},
  )
  return cfg


def yam_push_t_reachable_state_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """State-observation twin of the reachable task. The iteration harness.

  Same reward as the RGB version, so the pair is a clean one-variable test of
  whether a 32x32 wrist camera can resolve the T's yaw -- which no completed run
  has ever checked, since both state attempts stopped at 9 and 11 iterations.

  It is also roughly four times faster per iteration with no camera to render,
  which is what makes reward iteration practical at all.
  """
  cfg = yam_push_t_env_cfg(play=play)

  # The reach fix. With the stock scales the arm cannot get its gripper down
  # beside the block AT ALL: over 4096 constant actions spanning the action box,
  # the lowest end-effector height reachable is 60.2 mm and zero poses put a
  # gripper geom at the block's side. The block's top is at 20 mm, so every run
  # so far could only graze the block's TOP face -- run 7's trained policy dips
  # to 41.9 mm by dynamic overshoot, which drags the block along (143 mm of
  # travel) but cannot apply controlled yaw torque. That is exactly the
  # nine-run pattern of "position works, rotation never".
  #
  # A uniform 0.8 rad makes side contact reachable: min EE height 10.9 mm and
  # 2.12% of the action box puts a gripper geom beside the block, against
  # 0.00% at the stock scales. 1.2 rad is worse (1.64%) -- too coarse.
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale={"joint[1-6]": 0.8, "left_finger": 0.02},
      use_default_offset=True,
    )
  }

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
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
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
      params={
        "max_vel": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }
  cfg.metrics = make_push_t_metrics()
  cfg.curriculum = {}
  return cfg


def yam_push_t_precise_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Camera Push-T with a second, finer reward scale for tighter alignment.

  Forked from the camera task that works (99.2% pose success at 20 mm / 15 deg)
  and changes only the precision end of the objective.

  Two changes:

  A fine-scale bonus is ADDED alongside the coarse reward. The coarse terms are
  saturated where the working policy sits -- at 17 mm the position kernel is at
  0.419 of its 0.5 ceiling, and at 8.9 deg the rotation kernel is at 0.494,
  leaving 1.2% of headroom -- so there is nothing left to earn by improving. The
  bonus is ~0.01 at 30 mm / 20 deg and 1.0 at the goal, supplying gradient only
  inside about 2 cm. It is added, not substituted: sharpening the coarse kernels
  in place would flatten the far field that makes approach learnable at all.

  The success predicate tightens from 20 mm / 15 deg to 10 mm / 8 deg. Tighter
  than that would repeat the original mistake: coverage >= 0.90 needs 2.9 mm and
  4.6 deg jointly, never fired once in ~590M steps, and contributed no gradient
  at all. 10 mm / 8 deg is roughly half the current error, so it fires on the
  better episodes and stays live rather than dead.
  """
  cfg = yam_push_t_reachable_env_cfg(play=play)

  cfg.rewards["maniskill"].params["pos_tol"] = 0.010
  cfg.rewards["maniskill"].params["yaw_tol_deg"] = 8.0
  cfg.rewards["precision"] = RewardTermCfg(
    func=manipulation_mdp.push_precision_bonus,
    weight=2.0,
    params={
      "object_name": "t_object",
      "goal_name": "t_goal",
      "pos_scale": 20.0,
      "yaw_scale": 3.0,
    },
  )
  return cfg
