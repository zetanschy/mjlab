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
from mjlab.managers.metrics_manager import MetricsTermCfg
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

# Area overlap required to count as solved. gym-pusht uses 0.95, ManiSkill 0.90.
# Held lower here: at 0.90 the bonus fired only on episodes where the block
# happened to spawn near the goal yaw, so it rewarded luck instead of shaping
# rotation, and vanished once position sharpened.
SUCCESS_COVERAGE = 0.70


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
    # Additive, following ManiSkill's PushT. Position and orientation are
    # separate terms so a wrong yaw cannot zero the translation gradient, which
    # is what stalled the previous multiplicative version.
    "position": RewardTermCfg(
      func=manipulation_mdp.push_position_reward,
      weight=1.0,
      params={
        "object_name": OBJECT_NAME,
        "goal_name": GOAL_NAME,
        "scale": 5.0,
      },
    ),
    "orientation": RewardTermCfg(
      func=manipulation_mdp.push_orientation_reward,
      weight=2.5,
      params={
        "object_name": OBJECT_NAME,
        "goal_name": GOAL_NAME,
        "gate_distance": 0.08,
      },
    ),
    # Area overlap, the quantity gym-pusht actually scores. Zero until the
    # shapes touch, so it refines what position/orientation get close to.
    # The T partially overlaps itself when rotated, so coverage alone is not
    # monotonic in yaw: it peaks near 115 deg. Against orientation at 2.5, the
    # total stays monotonic for any coverage weight up to ~0.71; 0.6 keeps a
    # margin while still paying enough to matter once the block is placed.
    "coverage": RewardTermCfg(
      func=manipulation_mdp.object_goal_coverage,
      weight=0.6,
      params={"object_name": OBJECT_NAME, "goal_name": GOAL_NAME},
    ),
    # Capped at 0.05 internally, an order of magnitude under the task terms.
    # At parity it is farmable: the policy parks on the block and never pushes.
    "ee_guidance": RewardTermCfg(
      func=manipulation_mdp.push_ee_guidance_reward,
      weight=1.0,
      params={
        "object_name": OBJECT_NAME,
        "goal_name": GOAL_NAME,
        "scale": 5.0,
        "gate_distance": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "success": RewardTermCfg(
      func=manipulation_mdp.push_coverage_success,
      weight=2.0,
      params={
        "object_name": OBJECT_NAME,
        "goal_name": GOAL_NAME,
        "threshold": SUCCESS_COVERAGE,
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


def make_push_t_metrics() -> dict[str, MetricsTermCfg]:
  """Per-component metrics, logged under Episode_Metrics/*.

  Run 7 logged only the aggregate reward, so whether the block was ever being
  rotated had to be inferred after the fact -- and inferred wrongly. Every kill
  gate for a Push-T run should read these, never Episode_Reward/*, which can
  move for arithmetic reasons that have nothing to do with skill.
  """

  # A fresh dict per term: these params get specialised per task (a tabletop
  # variant sets table_height on just the two terms that accept it), and a
  # shared dict would leak that keyword into all nine.
  def names() -> dict[str, str]:
    return {"object_name": OBJECT_NAME, "goal_name": GOAL_NAME}

  return {
    # reduce="last": terminal pose is the question; averaging over the episode
    # buries the ending in the approach.
    "yaw_err_deg": MetricsTermCfg(
      func=manipulation_mdp.yaw_err_deg, params=names(), reduce="last"
    ),
    "pos_err_m": MetricsTermCfg(
      func=manipulation_mdp.pos_err_m, params=names(), reduce="last"
    ),
    "coverage": MetricsTermCfg(
      func=manipulation_mdp.coverage, params=names(), reduce="last"
    ),
    "success_pose": MetricsTermCfg(
      func=manipulation_mdp.success_pose, params=names(), reduce="last"
    ),
    "success_cov90": MetricsTermCfg(
      func=manipulation_mdp.success_coverage, params=names(), reduce="last"
    ),
    "block_travel_m": MetricsTermCfg(
      func=manipulation_mdp.block_travel_m, params=names(), reduce="last"
    ),
    # Compare against UNIFORM_YAW_NULL = 0.1875. Mean over the episode is right
    # here: it is a distributional check, not a terminal one.
    "rot_component": MetricsTermCfg(
      func=manipulation_mdp.rot_component, params=names(), reduce="mean"
    ),
    # "max": any lift at all matters, an episode mean would hide a brief one.
    "block_height_mm": MetricsTermCfg(
      func=manipulation_mdp.block_height_mm, params=names(), reduce="max"
    ),
    "block_tilt_dot": MetricsTermCfg(
      func=manipulation_mdp.block_tilt_dot, params=names(), reduce="mean"
    ),
  }
