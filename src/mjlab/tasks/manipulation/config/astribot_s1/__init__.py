from mjlab.asset_zoo.robots.astribot_s1.s1_constants import astribot_repo_path
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import s1_push_t_env_cfg
from .rl_cfg import s1_push_t_ppo_runner_cfg

# The S1 model is a separate clone, and mjlab imports every task at startup: without
# the model, skip this task instead of failing `train` for every other task.
try:
  astribot_repo_path()
except FileNotFoundError:
  pass
else:
  register_mjlab_task(
    task_id="Mjlab-Push-T-S1",
    env_cfg=s1_push_t_env_cfg(),
    play_env_cfg=s1_push_t_env_cfg(play=True),
    rl_cfg=s1_push_t_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
  )
