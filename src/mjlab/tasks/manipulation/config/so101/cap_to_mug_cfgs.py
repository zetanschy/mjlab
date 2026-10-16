"""Cap-to-mug on the agent_101 digital twin: env and PPO configs.

Mjlab-CapToMug-So101-Twin is the STATE TEACHER: actor and critic see privileged state
(the cap and mug poses, finger contacts). A two-camera student distils from it later.

Physics: the twin at the measured RL settings (twin.RL_*): dt 1/900, 30 substeps per
control step, so the policy acts at the robot's 30 Hz.

Action: an integrated joint target per joint (cap_to_mug_mdp.IntegratedJointPosition
ActionCfg), up to 0.05 rad per frame on the arm (1.5 rad/s; the identified servo
slews at 5.3) and 0.1 on the jaw.

Rewards, in stages: reach the cap, close the jaw there, both fingers on it, lift it
over the 12 cm rim, carry it over the mug's axis, lower it in, have it inside (the
twin's cap_in_mug), let go, get clear of the mug, then return to the rest pose and stay
still. Penalties: action rate, joint limits, pushing the mug.
Episodes end on time (20 s; the demos took 10-30 s), a tipped mug or a lost cap.
"""

from __future__ import annotations

from mjlab.asset_zoo.robots.so_arm101 import twin
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import cap_to_mug_mdp as c2m
from .rl_cfg import so101_push_t_ppo_runner_cfg

ARM = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
ACTION_SCALE = {**dict.fromkeys(ARM, 0.05), "Jaw": 0.1}  # rad per 30 Hz frame
ACTION_CLIP = {**dict.fromkeys(ARM, (-0.1, 0.1)), "Jaw": (-0.2, 0.2)}
# Wrist_Roll's target stays where the operator kept it: the 0.5-99.5% range of
# cap_to_mug_real_50 (-2.93..-0.93 rad, rest -1.43) plus a little. Unlimited, the teacher
# grasped rolled ~180 deg from rest, the grip camera under the gripper and at the table.
WRIST_ROLL_RANGE = (-2.95, -0.90)


def _finger_sensor(name: str, pattern: str) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=pattern, entity=c2m.ROBOT),
    secondary=ContactMatch(mode="body", pattern=c2m.CAP, entity=c2m.CAP),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )


def _state_terms(noise: bool) -> dict[str, ObservationTermCfg]:
  def n(x: float):
    return Unoise(n_min=-x, n_max=x) if noise else None

  return {
    "joint_pos": ObservationTermCfg(func=c2m.obs_joint_pos, noise=n(0.01)),
    "joint_vel": ObservationTermCfg(func=c2m.obs_joint_vel, noise=n(0.5)),
    "ee": ObservationTermCfg(func=c2m.obs_ee, noise=n(0.005)),
    "cap_from_ee": ObservationTermCfg(func=c2m.obs_cap_from_ee, noise=n(0.005)),
    "rim_from_ee": ObservationTermCfg(func=c2m.obs_rim_from_ee, noise=n(0.005)),
    "cap_from_rim": ObservationTermCfg(func=c2m.obs_cap_from_rim, noise=n(0.005)),
    "cap_up": ObservationTermCfg(func=c2m.obs_cap_up),
    "handle": ObservationTermCfg(func=c2m.obs_handle),
    "fingers_on_cap": ObservationTermCfg(func=c2m.obs_fingers_on_cap),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def so101_cap_to_mug_env_cfg(
  play: bool = False,
  num_envs: int = 4096,
  p_open: float = 0.0,
  max_cap_mug: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """The state teacher. p_open: share of caps lying open side up (the twin's layouts
  draw 0.5; start at 0). max_cap_mug: cap-mug distance cap, for a curriculum."""
  joints = tuple(twin.twin().units.URDF_JOINTS)
  scene = twin.scene_cfg(
    num_envs=num_envs,
    substeps=twin.RL_SUBSTEPS,
    n_caps=1,
    joint_pos=c2m.rest_joint_pos(),
    sensors=(
      _finger_sensor("fixed_finger_cap", r"fixed_finger_\d+"),
      _finger_sensor("moving_finger_cap", r"moving_finger_\d+"),
    ),
  )
  observations = {
    "actor": ObservationGroupCfg(_state_terms(noise=True), enable_corruption=not play),
    "critic": ObservationGroupCfg(_state_terms(noise=False), enable_corruption=False),
  }
  actions = {
    "joint_pos": c2m.IntegratedJointPositionActionCfg(
      entity_name=c2m.ROBOT,
      actuator_names=joints,
      scale=ACTION_SCALE,
      clip=ACTION_CLIP,
      target_limits={"Wrist_Roll": WRIST_ROLL_RANGE},
    )
  }
  events = {
    # First: it logs the ending episodes' success, and reset_arm would already have put
    # the arm back at rest, making cap_in_mug_and_home read the same as cap_in_mug.
    "reset_layout": EventTermCfg(
      func=c2m.reset_layout,
      mode="reset",
      params={"p_open": p_open, "max_cap_mug": max_cap_mug},
    ),
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg(c2m.ROBOT),
      },
    ),
    "reset_arm": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg(c2m.ROBOT, joint_names=(".*",)),
      },
    ),
  }
  rewards = {
    "reach": RewardTermCfg(func=c2m.reward_reach, weight=1.0),
    "close": RewardTermCfg(func=c2m.reward_close, weight=0.5),
    "grasp": RewardTermCfg(func=c2m.reward_grasp, weight=0.5),
    "lift": RewardTermCfg(func=c2m.reward_lift, weight=2.0),
    "carry": RewardTermCfg(func=c2m.reward_carry, weight=4.0),
    "lower": RewardTermCfg(func=c2m.reward_lower, weight=4.0),
    "in_mug": RewardTermCfg(func=c2m.reward_in_mug, weight=10.0),
    # 5, not 0.5: at 0.5 opening was <3% of the ~18/step a held cap in the mug earns,
    # and 500 iterations never moved the jaw.
    "release": RewardTermCfg(func=c2m.reward_release, weight=5.0),
    "clear": RewardTermCfg(func=c2m.reward_clear, weight=2.0),
    # 10, not 3: at 3 finishing the trip home was ~5% of the ~25/step a placed, released
    # cap earns; the arm stopped 1.5 rad short of a path a scripted move took 93% safely.
    "home": RewardTermCfg(func=c2m.reward_home, weight=10.0),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg(c2m.ROBOT, joint_names=joints)},
    ),
    "mug_moved": RewardTermCfg(func=c2m.penalty_mug_moved, weight=-2.0),
  }
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "mug_tipped": TerminationTermCfg(func=c2m.mug_tipped),
    "cap_lost": TerminationTermCfg(func=c2m.cap_lost),
  }
  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    # reduce="last", computed each step before any reset: logging them from the reset
    # event read a state sim.reset had already restored, so "and_home" was always 0.
    metrics={
      "cap_in_mug": MetricsTermCfg(func=c2m.metric_cap_in_mug, reduce="last"),
      "cap_in_mug_and_home": MetricsTermCfg(
        func=c2m.metric_cap_in_mug_and_home, reduce="last"
      ),
      # reduce="max": the share of episodes in which it ever came that low.
      "grip_cam_low": MetricsTermCfg(func=c2m.metric_grip_cam_low, reduce="max"),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name=c2m.ROBOT,
      body_name="gripper",
      distance=0.9,
      elevation=-35.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=twin.RL_NCONMAX,
      njmax=twin.RL_NJMAX,
      mujoco=MujocoCfg(**twin.mujoco_cfg_kwargs(twin.RL_SUBSTEPS, **twin.RL_SOLVER)),
    ),
    decimation=twin.RL_SUBSTEPS,
    episode_length_s=20.0,
  )


def so101_cap_to_mug_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The state PPO the SO-101 Push-T/Lift-T tasks use: 256-256-128, sigma >= 0.2."""
  cfg = so101_push_t_ppo_runner_cfg()
  cfg.experiment_name = "so101_cap_to_mug_teacher"
  cfg.max_iterations = 3_000
  return cfg
