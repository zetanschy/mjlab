from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

_VISION_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


def yam_lift_cube_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="yam_lift_cube",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )


def yam_lift_cube_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cnn_cfg = _VISION_CNN_CFG
  class_name = _VISION_MODEL_CLS
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="yam_lift_cube_vision",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )


def yam_multi_cube_seg_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=_VISION_CNN_CFG,
      class_name=_VISION_MODEL_CLS,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=_VISION_CNN_CFG,
      class_name=_VISION_MODEL_CLS,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="yam_multi_cube_seg",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )


def yam_push_t_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = yam_lift_cube_ppo_runner_cfg()
  cfg.experiment_name = "yam_push_t"
  cfg.max_iterations = 6_000
  return cfg


def yam_push_t_vision_ppo_runner_cfg(
  cam_type: str = "depth",
) -> RslRlOnPolicyRunnerCfg:
  # Per-modality experiment name so the rgb and depth runs do not write
  # checkpoints into the same log directory.
  cfg = yam_lift_cube_vision_ppo_runner_cfg()
  cfg.experiment_name = f"yam_push_t_{cam_type}"
  cfg.max_iterations = 6_000
  return cfg


def yam_push_t_maniskill_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = yam_lift_cube_vision_ppo_runner_cfg()
  cfg.experiment_name = "yam_push_t_maniskill"
  cfg.max_iterations = 6_000
  return cfg


def yam_push_t_hybrid_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = yam_lift_cube_vision_ppo_runner_cfg()
  cfg.experiment_name = "yam_push_t_hybrid"
  cfg.max_iterations = 6_000
  return cfg


def yam_push_t_replica_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """State MLP runner with the discount matched to the episode.

  gamma=0.995 gives 1/(1-gamma) = 200 = the full 200-step episode, a
  horizon/episode ratio of 1.0. Run 7 used gamma=0.99 over 1000-step episodes,
  a ratio of 0.10, so the value function could not see the end of an episode
  from its start -- and a reorientation payoff arrives at the end.
  """
  cfg = yam_lift_cube_ppo_runner_cfg()
  cfg.experiment_name = "yam_push_t_replica"
  cfg.max_iterations = 4_000
  cfg.num_steps_per_env = 24
  cfg.algorithm.gamma = 0.99
  cfg.algorithm.lam = 0.95
  return cfg


def yam_push_t_replica_gravcomp_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = yam_push_t_replica_ppo_runner_cfg()
  cfg.experiment_name = "yam_push_t_replica_gravcomp"
  return cfg


def yam_push_t_reachable_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Vision runner carrying the same sigma floor as the state task.

  Without it sigma collapsed to 0.026 and the policy never discovered the
  descent to the block's side, even once that descent was reachable.
  """
  cfg = yam_lift_cube_vision_ppo_runner_cfg()
  cfg.experiment_name = "yam_push_t_reachable"
  cfg.max_iterations = 5_000
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
    "std_range": (0.2, 1.0),
  }
  return cfg


def yam_push_t_reachable_state_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """State runner with a floor under the exploration noise.

  Side contact only became geometrically reachable once the action scale was
  widened, but reaching it is a specific descent the policy still has to find.
  At iteration 1000 of the first run with the wider box, the policy's median
  minimum end-effector height was 123 mm -- higher than run 7 managed with the
  NARROWER box -- and only 3.1% of episodes ever reached side-contact geometry,
  with mean_std down to 0.026. There was no exploration left to find it with.

  std_range floors sigma at 0.2 so the descent stays discoverable. Raising
  entropy_coef alone does not bound sigma; this does.
  """
  cfg = yam_lift_cube_ppo_runner_cfg()
  cfg.experiment_name = "yam_push_t_reachable_state"
  cfg.max_iterations = 5_000
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
    "std_range": (0.2, 1.0),
  }
  return cfg
