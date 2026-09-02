from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  yam_lift_cube_env_cfg,
  yam_lift_cube_vision_env_cfg,
  yam_multi_cube_seg_env_cfg,
  yam_push_t_d435_env_cfg,
  yam_push_t_d435_push_env_cfg,
  yam_push_t_env_cfg,
  yam_push_t_hybrid_env_cfg,
  yam_push_t_maniskill_env_cfg,
  yam_push_t_precise_env_cfg,
  yam_push_t_precise_random_goal_env_cfg,
  yam_push_t_reachable_env_cfg,
  yam_push_t_reachable_state_env_cfg,
  yam_push_t_replica_env_cfg,
  yam_push_t_vision_env_cfg,
)
from .rl_cfg import (
  yam_lift_cube_ppo_runner_cfg,
  yam_lift_cube_vision_ppo_runner_cfg,
  yam_multi_cube_seg_ppo_runner_cfg,
  yam_push_t_d435_ppo_runner_cfg,
  yam_push_t_d435_push_ppo_runner_cfg,
  yam_push_t_hybrid_ppo_runner_cfg,
  yam_push_t_maniskill_ppo_runner_cfg,
  yam_push_t_ppo_runner_cfg,
  yam_push_t_precise_ppo_runner_cfg,
  yam_push_t_precise_random_goal_ppo_runner_cfg,
  yam_push_t_reachable_ppo_runner_cfg,
  yam_push_t_reachable_state_ppo_runner_cfg,
  yam_push_t_replica_gravcomp_ppo_runner_cfg,
  yam_push_t_replica_ppo_runner_cfg,
  yam_push_t_vision_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam",
  env_cfg=yam_lift_cube_env_cfg(),
  play_env_cfg=yam_lift_cube_env_cfg(play=True),
  rl_cfg=yam_lift_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam-Rgb",
  env_cfg=yam_lift_cube_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=yam_lift_cube_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=yam_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam-Depth",
  env_cfg=yam_lift_cube_vision_env_cfg(cam_type="depth"),
  play_env_cfg=yam_lift_cube_vision_env_cfg(cam_type="depth", play=True),
  rl_cfg=yam_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Multi-Cube-Seg-Yam",
  env_cfg=yam_multi_cube_seg_env_cfg(num_cubes=3),
  play_env_cfg=yam_multi_cube_seg_env_cfg(num_cubes=3, play=True),
  rl_cfg=yam_multi_cube_seg_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam",
  env_cfg=yam_push_t_env_cfg(),
  play_env_cfg=yam_push_t_env_cfg(play=True),
  rl_cfg=yam_push_t_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Depth",
  env_cfg=yam_push_t_vision_env_cfg(cam_type="depth"),
  play_env_cfg=yam_push_t_vision_env_cfg(cam_type="depth", play=True),
  rl_cfg=yam_push_t_vision_ppo_runner_cfg(cam_type="depth"),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Rgb",
  env_cfg=yam_push_t_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=yam_push_t_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=yam_push_t_vision_ppo_runner_cfg(cam_type="rgb"),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Rgb-Visual-Goal",
  env_cfg=yam_push_t_vision_env_cfg(cam_type="rgb", visual_goal=True),
  play_env_cfg=yam_push_t_vision_env_cfg(cam_type="rgb", visual_goal=True, play=True),
  rl_cfg=yam_push_t_vision_ppo_runner_cfg(cam_type="rgb_visual_goal"),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Rgb-Maniskill",
  env_cfg=yam_push_t_maniskill_env_cfg(),
  play_env_cfg=yam_push_t_maniskill_env_cfg(play=True),
  rl_cfg=yam_push_t_maniskill_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Rgb-Hybrid",
  env_cfg=yam_push_t_hybrid_env_cfg(),
  play_env_cfg=yam_push_t_hybrid_env_cfg(play=True),
  rl_cfg=yam_push_t_hybrid_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Replica",
  env_cfg=yam_push_t_replica_env_cfg(),
  play_env_cfg=yam_push_t_replica_env_cfg(play=True),
  rl_cfg=yam_push_t_replica_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Replica-Gravcomp",
  env_cfg=yam_push_t_replica_env_cfg(relative_action=True, gravcomp=True),
  play_env_cfg=yam_push_t_replica_env_cfg(
    relative_action=True, gravcomp=True, play=True
  ),
  rl_cfg=yam_push_t_replica_gravcomp_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Reachable",
  env_cfg=yam_push_t_reachable_env_cfg(),
  play_env_cfg=yam_push_t_reachable_env_cfg(play=True),
  rl_cfg=yam_push_t_reachable_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Reachable-State",
  env_cfg=yam_push_t_reachable_state_env_cfg(),
  play_env_cfg=yam_push_t_reachable_state_env_cfg(play=True),
  rl_cfg=yam_push_t_reachable_state_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Precise",
  env_cfg=yam_push_t_precise_env_cfg(),
  play_env_cfg=yam_push_t_precise_env_cfg(play=True),
  rl_cfg=yam_push_t_precise_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-Precise-Random-Goal",
  env_cfg=yam_push_t_precise_random_goal_env_cfg(),
  play_env_cfg=yam_push_t_precise_random_goal_env_cfg(play=True),
  rl_cfg=yam_push_t_precise_random_goal_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-D435",
  env_cfg=yam_push_t_d435_env_cfg(),
  play_env_cfg=yam_push_t_d435_env_cfg(play=True),
  rl_cfg=yam_push_t_d435_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Push-T-Yam-D435-Push",
  env_cfg=yam_push_t_d435_push_env_cfg(),
  play_env_cfg=yam_push_t_d435_push_env_cfg(play=True),
  rl_cfg=yam_push_t_d435_push_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
