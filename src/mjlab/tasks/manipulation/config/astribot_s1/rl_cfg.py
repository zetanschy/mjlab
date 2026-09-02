"""PPO config for Push-T on the Astribot S1."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manipulation.config.yam.rl_cfg import (
  yam_push_t_precise_random_goal_ppo_runner_cfg,
)


def s1_push_t_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same runner as the YAM camera task, including the sigma floor.

  The floor matters more here, not less: the S1 has to discover contact with a
  0.1 m block using a 7-DoF arm it has never moved, and on the YAM an unfloored
  sigma collapsed to 0.026 and the descent to the block was never found.
  """
  cfg = yam_push_t_precise_random_goal_ppo_runner_cfg()
  cfg.experiment_name = "s1_push_t"
  cfg.max_iterations = 6_000
  return cfg
