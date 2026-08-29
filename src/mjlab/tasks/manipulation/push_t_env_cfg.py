"""Push-T: shove a T-shaped block onto a matching footprint on the table.

Robot-agnostic base config. Anything marked "Set per-robot" is filled in by a
config under ``config/<robot>/``.

Unlike the lifting tasks, the goal here is a mocap entity in the scene rather
than a command term. That keeps the green footprint and the reward's notion of
the target as the same object, so what the policy is scored against is exactly
what is drawn.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.push_t_scene import make_push_t_scene_cfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

OBJECT_NAME = "t_object"
GOAL_NAME = "t_goal"

# Spawn randomization, as offsets from T_SPAWN_CENTER in push_t_scene.
SPAWN_X_RANGE = (-0.05, 0.05)
SPAWN_Y_RANGE = (-0.12, 0.12)
SPAWN_YAW_RANGE = (-3.14, 3.14)

# Episode ends if the block is pushed outside this box (env-local coordinates).
WORKSPACE_X = (0.12, 0.60)
WORKSPACE_Y = (-0.35, 0.35)

SUCCESS_POS_TOL = 0.02  # 2 cm
SUCCESS_YAW_TOL = 0.26  # ~15 degrees


def make_push_t_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base Push-T task configuration."""

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "ee_to_t": ObservationTermCfg(
      func=manipulation_mdp.ee_to_object_planar,
      params={
        "object_name": OBJECT_NAME,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "t_to_goal": ObservationTermCfg(
      func=manipulation_mdp.object_to_goal_pose_error,
      params={"object_name": OBJECT_NAME, "goal_name": GOAL_NAME},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  events = {
    # Positions the robot base at env_origins.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # The block starts somewhere random; the footprint does not move.
    "reset_t_object": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": SPAWN_X_RANGE,
          "y": SPAWN_Y_RANGE,
          "yaw": SPAWN_YAW_RANGE,
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg(OBJECT_NAME),
      },
    ),
    # Empty ranges, but still required: a mocap entity stays at the world origin
    # until a reset event places it per environment.
    "reset_t_goal": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "asset_cfg": SceneEntityCfg(GOAL_NAME)},
    ),
    "t_friction_slide": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(OBJECT_NAME, geom_names=(".*_collision",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.2, 0.6),
      },
    ),
    "fingertip_friction_slide": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.3, 1.5),
      },
    ),
  }

  ee_ground_collision_cfg = ContactSensorCfg(
    name="ee_ground_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern="",  # Set per-robot.
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  rewards = {
    # Coarse shaping: get the end effector to the block, then the block to the
    # footprint. Gated so approach is learned before placement.
    "push": RewardTermCfg(
      func=manipulation_mdp.push_staged_reward,
      weight=1.0,
      params={
        "object_name": OBJECT_NAME,
        "goal_name": GOAL_NAME,
        "reaching_std": 0.15,
        "placing_std": 0.20,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Fine shaping: position and yaw together.
    "pose_match": RewardTermCfg(
      func=manipulation_mdp.object_goal_pose_reward,
      weight=2.0,
      params={
        "object_name": OBJECT_NAME,
        "goal_name": GOAL_NAME,
        "pos_std": 0.05,
        "yaw_std": 0.50,
      },
    ),
    "success": RewardTermCfg(
      func=manipulation_mdp.push_success_bonus,
      weight=3.0,
      params={
        "object_name": OBJECT_NAME,
        "goal_name": GOAL_NAME,
        "pos_tol": SUCCESS_POS_TOL,
        "yaw_tol": SUCCESS_YAW_TOL,
      },
    ),
    # Keep it a pushing task: no lifting, no flipping the block on its side.
    "object_off_table": RewardTermCfg(
      func=manipulation_mdp.object_displacement_penalty,
      weight=-2.0,
      params={"object_name": OBJECT_NAME},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
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

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "ee_ground_collision": TerminationTermCfg(
      func=manipulation_mdp.illegal_contact,
      params={"sensor_name": "ee_ground_collision", "force_threshold": 10.0},
    ),
    "t_out_of_bounds": TerminationTermCfg(
      func=manipulation_mdp.object_out_of_bounds,
      params={
        "object_name": OBJECT_NAME,
        "x_range": WORKSPACE_X,
        "y_range": WORKSPACE_Y,
      },
    ),
  }

  curriculum = {
    "joint_vel_hinge_weight": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 500 * 24, "weight": -0.1},
          {"step": 1000 * 24, "weight": -1.0},
        ],
      },
    ),
  }

  scene = make_push_t_scene_cfg(num_envs=1, env_spacing=1.0)
  scene.sensors = (ee_ground_collision_cfg,)

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      elevation=-25.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=80,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
