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
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
from mjlab.tasks.manipulation.push_t_env_cfg import make_push_t_env_cfg


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
  cfg.rewards["push"].params["asset_cfg"].site_names = ("grasp_site",)

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
    # green footprint, so its channels are sampled per-axis into warm hues
    # instead of the full colour cube.
    color_ranges: Any = (
      {0: (0.45, 0.95), 1: (0.0, 0.40), 2: (0.0, 0.40)} if visual_goal else (0.0, 0.7)
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
