from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import s1_push_t_env_cfg
from .rl_cfg import s1_push_t_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Push-T-S1",
  env_cfg=s1_push_t_env_cfg(),
  play_env_cfg=s1_push_t_env_cfg(play=True),
  rl_cfg=s1_push_t_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
