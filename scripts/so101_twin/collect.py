"""Collect a cap-to-mug dataset for behaviour cloning and offline RL, from the state teacher.

  env -u PYTHONPATH MUJOCO_GL=egl uv run --frozen --no-sync python scripts/so101_twin/collect.py \
      --checkpoint .../model_2800.pt --out /mnt/storage/datasets/cap_to_mug_teacher --episodes 2000

EVERY STEP (30 Hz) holds what the real robot has, plus the labels:
  front (72, 96, 3), grip (48, 64, 3)  uint8 RGB from the twin's calibrated cameras, as
                                       undistorted pinholes (undistort + resize real frames)
  joint_pos (6) rad, joint_vel (6) rad/s   URDF order Rotation .. Jaw (real: units.to_urdf)
  target (6) rad      the absolute joint goal sent after this step's action
  action (6)          the action executed: the teacher's plus noise (offline RL)
  teacher_action (6)  the teacher's clean action at this state (the BC label, DART-style)
  reward              the task reward for this step
  privileged (35)     the teacher's own state observation (for relabelling / critics)
EVERY EPISODE: length, success (the cap in the mug and the arm home), terminal (mug tipped
  or cap lost), noise, cap_up, cap_xy, mug_xy, handle_yaw (rad).

BALANCE AND VARIETY: caps open/closed side up 50/50 (p_open 0.5); layouts drawn as the
twin's live layouts; action noise cycled per episode over --noise (the label stays the
clean action); visual randomization per episode, drawn as the twin's live/dr.py draws it
(its RANGES): key light gain and tint, mat and mug gain, cap HSV, camera mounts.
Recording stops HOLD_S after the cap is in the mug and the arm home, so the still-at-home
tail is kept but short. Failed episodes are kept and flagged.

OUTPUT: shard_NNNN.npz (np.savez_compressed), each self-contained: the step arrays
concatenated, plus ep_* arrays with one row per episode and ep_start offsets.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.asset_zoo.robots.so_arm101 import twin
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg as E
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation.config.so101 import cap_to_mug_cfgs as cfgs
from mjlab.tasks.manipulation.config.so101 import cap_to_mug_mdp as m
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

TASK = "Mjlab-CapToMug-So101-Twin"
FPS = 30
CAMS = {
  "front": (96, 72),
  "grip": (64, 48),
}  # (width, height), 4:3 like the real 640x480
HOLD_S = 1.0
EPISODE_S = 15.0  # the teacher finishes in ~2-6 s; noisy episodes get the rest
SHARD_FRAMES = 20_000


def _twin_dr():
  twin.twin()  # puts sim/ on sys.path
  return importlib.import_module("real2sim.live.dr")


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  k = (torch.tensor([5.0, 3.0, 1.0], device=h.device) + 6 * h) % 6
  return v - v * s * torch.clamp(torch.minimum(k, 4 - k), 0, 1)


@requires_model_fields("light_diffuse", "mat_rgba", "geom_rgba")
def visual_dr(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, r: dict) -> None:
  """The twin's live/dr.py visual draws: one gain on the key light times a small
  per-channel tint, one gain on the mat and on the mug, the cap's colour from its HSV
  box. (mjlab's dr.light_diffuse/mat_rgba/geom_rgba draw each channel apart, even with
  shared_random: every episode got a strong colour cast.)"""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  ids, n, mj, model = env_ids[:, None], len(env_ids), env.sim.mj_model, env.sim.model
  d = env.sim.get_default_field

  def u(key: str, k: int = 1) -> torch.Tensor:
    lo, hi = r[key] if isinstance(r[key], tuple) else (-r[key], r[key])
    return torch.empty(n, k, device=env.device).uniform_(lo, hi)

  key = [mj.light("robot/key").id]
  tint = 1 + u("light_tint", 3)
  model.light_diffuse[ids, key] = (
    d("light_diffuse")[key] * (u("light_gain") * tint)[:, None]
  ).clamp(0, 2)
  mat = [mj.material("robot/look_table").id]
  model.mat_rgba[ids, mat, :3] = (
    d("mat_rgba")[mat, :3] * u("mat_gain")[:, None]
  ).clamp(0, 1)
  mug = [mj.geom(f"mug/{g}").id for g in ("mug_body_vis", "mug_handle_vis")]
  model.geom_rgba[ids, mug, :3] = (
    d("geom_rgba")[mug, :3] * u("mug_gain")[:, None]
  ).clamp(0, 1)
  rgb = _hsv_to_rgb(u("cap_hue_deg") / 360, u("cap_sat"), u("cap_val"))
  model.geom_rgba[ids, [mj.geom("cap0/cap0_vis").id], :3] = rgb[:, None]


def env_cfg(n: int):
  cfg = cfgs.so101_cap_to_mug_env_cfg(play=True, num_envs=n, p_open=0.5)
  cfg.episode_length_s = EPISODE_S
  cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(
    CameraSensorCfg(
      name=f"{c}_cam",
      camera_name=f"robot/{c}",
      width=w,
      height=h,
      data_types=("rgb",),
      enabled_geom_groups=(twin.VISUAL_GROUP,),
    )
    for c, (w, h) in CAMS.items()
  )
  r, deg = _twin_dr().RANGES, math.radians
  cams = {}
  for c in ("front", "grip"):
    mm, dg = r[f"{c}_cam_mm"] / 1000, deg(r.get(f"{c}_cam_deg", 0.0))
    cams[f"dr_{c}_pos"] = (
      dr.cam_pos,
      {
        "ranges": (-mm, mm),
        "operation": "add",
        "asset_cfg": E("robot", camera_names=(c,)),
      },
    )
    cams[f"dr_{c}_rot"] = (
      dr.cam_quat,
      {
        "roll_range": (-dg, dg),
        "pitch_range": (-dg, dg),
        "yaw_range": (-dg, dg),
        "asset_cfg": E("robot", camera_names=(c,)),
      },
    )
  ev = {"dr_visual": (visual_dr, {"r": r}), **cams}
  for k, (fn, params) in ev.items():
    cfg.events[k] = EventTermCfg(func=fn, mode="reset", params=params)
  return cfg


def episode_meta(env: ManagerBasedRlEnv, ids: torch.Tensor) -> dict[str, torch.Tensor]:
  o = env.scene.env_origins[ids]
  q = env.scene[m.MUG].data.root_link_quat_w[ids]
  return {
    "cap_up": (m.cap_axis_z(env)[ids] < 0),  # True = open side up
    "cap_xy": env.scene[m.CAP].data.root_link_pos_w[ids, :2] - o[:, :2],
    "mug_xy": env.scene[m.MUG].data.root_link_pos_w[ids, :2] - o[:, :2],
    "handle_yaw": 2 * torch.atan2(q[:, 3], q[:, 0]),
  }


def write_shard(path: Path, eps: list[dict]) -> int:
  keys = [k for k in eps[0]["steps"]]
  arr = {k: np.concatenate([e["steps"][k] for e in eps]) for k in keys}
  lens = np.array([len(e["steps"]["reward"]) for e in eps])
  arr["ep_start"] = np.concatenate([[0], np.cumsum(lens)[:-1]])
  arr["ep_length"] = lens
  for k in eps[0]["meta"]:
    arr[f"ep_{k}"] = np.stack([e["meta"][k] for e in eps])
  np.savez_compressed(path, **arr)
  return int(lens.sum())


def main() -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--checkpoint", required=True, type=Path)
  p.add_argument("--out", required=True, type=Path)
  p.add_argument("--episodes", type=int, default=2000)
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument(
    "--noise",
    type=float,
    nargs="+",
    default=[0.0, 0.1, 0.3],
    help="action noise std per episode, cycled for balance",
  )
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  a.out.mkdir(parents=True, exist_ok=True)
  n, dev = a.num_envs, a.device

  env = ManagerBasedRlEnv(cfg=env_cfg(n), device=dev)
  agent_cfg = load_rl_cfg(TASK)
  vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK)
  assert runner_cls is not None
  runner = runner_cls(vec, asdict(agent_cfg), device=dev)
  runner.load(
    str(a.checkpoint), load_cfg={"actor": True}, strict=True, map_location=dev
  )
  policy = runner.get_inference_policy(device=dev)
  act_term = env.action_manager.get_term("joint_pos")
  assert isinstance(act_term, m.IntegratedJointPositionAction)
  robot = env.scene[m.ROBOT]

  T = int(EPISODE_S * FPS) + 2
  shapes = {
    "front": ((CAMS["front"][1], CAMS["front"][0], 3), torch.uint8),
    "grip": ((CAMS["grip"][1], CAMS["grip"][0], 3), torch.uint8),
    "joint_pos": ((6,), torch.float32),
    "joint_vel": ((6,), torch.float32),
    "target": ((6,), torch.float32),
    "action": ((6,), torch.float32),
    "teacher_action": ((6,), torch.float32),
    "reward": ((), torch.float32),
    "privileged": ((35,), torch.float32),
  }
  buf = {
    k: torch.zeros((n, T, *s), dtype=d, device=dev) for k, (s, d) in shapes.items()
  }
  t = torch.zeros(n, dtype=torch.long, device=dev)
  hold = torch.zeros(n, dtype=torch.long, device=dev)
  noise = torch.zeros(n, device=dev)
  meta: dict[str, torch.Tensor] = {}
  levels = torch.tensor(a.noise, device=dev)
  started = 0

  def start(ids: torch.Tensor) -> None:
    nonlocal started
    k = len(ids)
    noise[ids] = levels[(started + torch.arange(k, device=dev)) % len(levels)]
    started += k
    for key, v in episode_meta(env, ids).items():
      if key not in meta:
        meta[key] = torch.zeros((n, *v.shape[1:]), dtype=v.dtype, device=dev)
      meta[key][ids] = v
    t[ids], hold[ids] = 0, 0

  obs, _ = vec.reset()
  start(torch.arange(n, device=dev))
  pending: list[dict] = []
  saved = frames = shard = 0
  rows = torch.arange(n, device=dev)
  with torch.inference_mode():
    while saved < a.episodes:
      clean = policy(obs)
      act = clean + noise[:, None] * torch.randn_like(clean)
      ti = torch.clamp(t, max=T - 1)
      buf["front"][rows, ti] = env.scene["front_cam"].data.rgb
      buf["grip"][rows, ti] = env.scene["grip_cam"].data.rgb
      buf["joint_pos"][rows, ti] = robot.data.joint_pos[:, :6]
      buf["joint_vel"][rows, ti] = robot.data.joint_vel[:, :6]
      buf["privileged"][rows, ti] = torch.as_tensor(obs["actor"])
      buf["teacher_action"][rows, ti], buf["action"][rows, ti] = clean, act
      obs, rew, dones, extras = vec.step(act)
      buf["reward"][rows, ti] = rew
      buf["target"][rows, ti] = (
        act_term.target
      )  # already the next episode's for auto-reset envs
      t += 1
      done = dones.bool()
      won = m.cap_in_mug(env) & (m.arm_to_rest(env) < 0.3) & ~done
      hold = torch.where(won, hold + 1, torch.zeros_like(hold))
      finish = done | (hold >= int(HOLD_S * FPS))
      if not finish.any():
        continue
      ids = finish.nonzero().flatten()
      timeout = extras["time_outs"].bool()
      for i in ids.tolist():
        L = int(t[i])
        steps = {k: v[i, :L].cpu().numpy() for k, v in buf.items()}
        ep = {k: v[i].cpu().numpy() for k, v in meta.items()}
        ep.update(
          success=bool(hold[i] > 0),
          terminal=bool(done[i] and not timeout[i]),
          noise=float(noise[i]),
        )
        pending.append({"steps": steps, "meta": ep})
        saved += 1
      cut = ids[~done[ids]]  # finished by the hold: reset them ourselves
      if len(cut):
        env.reset(env_ids=cut)
        obs = vec.get_observations()
      start(ids)
      if (
        sum(len(e["steps"]["reward"]) for e in pending) >= SHARD_FRAMES
        or saved >= a.episodes
      ):
        frames += write_shard(a.out / f"shard_{shard:04d}.npz", pending)
        print(
          f"shard {shard}: {len(pending)} episodes ({saved}/{a.episodes} saved, {frames} frames)",
          flush=True,
        )
        shard, pending = shard + 1, []

  info = {
    "task": TASK,
    "checkpoint": str(a.checkpoint),
    "fps": FPS,
    "episodes": saved,
    "frames": frames,
    "cameras": {c: {"width": w, "height": h} for c, (w, h) in CAMS.items()},
    "noise_levels": a.noise,
    "p_open": 0.5,
    "hold_s": HOLD_S,
    "episode_s": EPISODE_S,
    "joint_order": list(twin.twin().units.URDF_JOINTS),
    "units": "rad, rad/s; images uint8 RGB",
  }
  (a.out / "meta.json").write_text(json.dumps(info, indent=2))
  check(a.out)
  return 0


def check(out: Path) -> None:
  """Self-check: shards load, offsets add up, images are not blank, and the balance."""
  rows = []
  for f in sorted(out.glob("shard_*.npz")):
    z = np.load(f)
    assert int(z["ep_length"].sum()) == len(z["reward"]) == len(z["front"])
    assert (z["ep_start"] + z["ep_length"])[-1] == len(z["reward"])
    assert z["front"].std() > 5 and z["grip"].std() > 5, f"{f.name}: blank images"
    rows += [
      (bool(o), round(float(s), 3), bool(w))
      for o, s, w in zip(z["ep_cap_up"], z["ep_noise"], z["ep_success"], strict=True)
    ]
  for up in (False, True):
    for s in sorted({r[1] for r in rows}):
      cell = [r for r in rows if r[0] == up and r[1] == s]
      print(
        f"  cap {'open' if up else 'closed'} up, noise {s}: {len(cell)} episodes, "
        f"{sum(r[2] for r in cell)} successes"
      )


if __name__ == "__main__":
  raise SystemExit(main())
