"""PPO for the SO-ARM101 Push-T tasks.

The YAM D435-Push chain's numbers, unchanged: 256-256-128 actor and critic over the
same spatial-softmax CNN, entropy 0.005, 5 epochs, 4 minibatches, lr 1e-3 adaptive,
24 steps per env. Copied rather than retuned so that a failure here says something
about the arm rather than about the optimizer.

The sigma floor comes with them, and it is the one setting worth understanding before
changing: mjlab's note records sigma collapsing to 0.026 with only 3.1% of episodes
ever reaching contact geometry -- there was no exploration left to find the descent
with -- and that raising entropy_coef alone does not bound sigma. std_range does.
"""

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

_DISTRIBUTION = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
  "std_range": (0.2, 1.0),
}

_ALGORITHM = RslRlPpoAlgorithmCfg(
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
)


def so101_push_t_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """State task: no camera, so no CNN and one observation group per network."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=dict(_DISTRIBUTION),
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=_ALGORITHM,
    experiment_name="so101_push_t",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
  )


def so101_push_t_camera_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Camera task: both networks take the wrist frame alongside the state."""
  cfg = so101_push_t_ppo_runner_cfg()
  cfg.actor = RslRlModelCfg(
    hidden_dims=(256, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=_VISION_CNN_CFG,
    class_name=_VISION_MODEL_CLS,
    distribution_cfg=dict(_DISTRIBUTION),
  )
  cfg.critic = RslRlModelCfg(
    hidden_dims=(256, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=_VISION_CNN_CFG,
    class_name=_VISION_MODEL_CLS,
  )
  cfg.experiment_name = "so101_push_t_camera"
  cfg.obs_groups = {"actor": ("actor", "camera"), "critic": ("critic", "camera")}
  return cfg


def so101_push_t_push_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = so101_push_t_camera_ppo_runner_cfg()
  cfg.experiment_name = "so101_push_t_push"
  return cfg
