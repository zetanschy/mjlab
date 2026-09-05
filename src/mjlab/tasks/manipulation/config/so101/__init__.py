from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  so101_push_t_camera_env_cfg,
  so101_push_t_env_cfg,
  so101_push_t_push_env_cfg,
)
from .rl_cfg import (
  so101_push_t_camera_ppo_runner_cfg,
  so101_push_t_ppo_runner_cfg,
  so101_push_t_push_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-So101",
  env_cfg=so101_push_t_env_cfg(),
  play_env_cfg=so101_push_t_env_cfg(play=True),
  rl_cfg=so101_push_t_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-So101-Camera",
  env_cfg=so101_push_t_camera_env_cfg(),
  play_env_cfg=so101_push_t_camera_env_cfg(play=True),
  rl_cfg=so101_push_t_camera_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-So101-Push",
  env_cfg=so101_push_t_push_env_cfg(),
  play_env_cfg=so101_push_t_push_env_cfg(play=True),
  rl_cfg=so101_push_t_push_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
