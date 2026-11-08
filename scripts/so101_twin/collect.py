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
  qpos (nq)           the whole sim state, cap and mug relative to their env (re-rendering)
EVERY EPISODE: length, success (the cap in the mug and the arm home), terminal (mug tipped
  or cap lost), noise, cap_up, cap_xy, mug_xy, handle_yaw (rad), visual (the DR draw, JSON).

THE FRAMES ARE THE LIVE TWIN'S LOOK, not mujoco_warp's. The physics runs on the GPU with
no cameras; each finished episode is rendered from its qpos by MuJoCo's own renderer on a
textured copy of the same scene (TwinRenderer): the mat's printed texture, shadows, the
key light. A student trained on warp's flat, shadowless frames scored 13/20 in mjlab and
0/20 in the live twin, which renders like this. Rendered at SS x the input size, then box
averaged, as the real pipeline area-resizes 640x480 frames.

BALANCE AND VARIETY: caps open/closed side up 50/50 (p_open 0.5); layouts drawn as the
twin's live layouts; action noise cycled per episode over --noise (the label stays the
clean action); per episode one draw of the twin's live/dr.py visual randomization, the
same function and ranges the live sim uses: key light gain, tint and direction, ambient,
mat, mug and arm brightness, cap colour, camera mounts. Recording stops HOLD_S after the
cap is in the mug and the arm home, so the still-at-home tail is kept but short. Failed
episodes are kept and flagged.

OUTPUT: shard_NNNN.npz (np.savez_compressed), each self-contained: the step arrays
concatenated, plus ep_* arrays with one row per episode and ep_start offsets.
"""

from __future__ import annotations

import argparse
import copy
import functools
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.so_arm101 import twin
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import Scene
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
SS = 4  # render supersampling


def _twin_dr():
  twin.twin()  # puts sim/ on sys.path
  return importlib.import_module("real2sim.live.dr")


def env_cfg(n: int):
  cfg = cfgs.so101_cap_to_mug_env_cfg(play=True, num_envs=n, p_open=0.5)
  cfg.episode_length_s = EPISODE_S
  return cfg


class TwinRenderer:
  """Episodes rendered as the live twin renders them: MuJoCo's own renderer on a CPU
  copy of the env's scene with the textures kept, and per episode one draw of the
  twin's live/dr.py visual randomization, applied as its Randomizer.apply applies it
  (the names differ here: mjlab prefixes them by entity)."""

  def __init__(self, cfg, nq: int) -> None:
    sc = copy.deepcopy(cfg.scene)
    f = sc.entities["robot"].spec_fn
    sc.entities["robot"].spec_fn = functools.partial(
      f.func, *f.args, **{**f.keywords, "flat": False}
    )
    scene = Scene(sc, "cpu")
    # The look's scene-level globals, which attaching the robot drops: its fitted
    # ambient-only headlight (mjlab's default adds diffuse: brighter, flatter) and
    # shadow map size.
    look, v = sc.entities["robot"].spec_fn().visual, scene.spec.visual
    for k in ("ambient", "diffuse", "specular"):
      setattr(v.headlight, k, getattr(look.headlight, k))
    v.quality.shadowsize = look.quality.shadowsize
    m = self.model = scene.compile()
    assert m.nq == nq, f"render model nq {m.nq} != the env's {nq}"
    self.data = mujoco.MjData(m)
    self.r = {c: mujoco.Renderer(m, h * SS, w * SS) for c, (w, h) in CAMS.items()}
    self.dr = _twin_dr()
    self.key = m.light("robot/key").id
    self.mat = m.material("robot/look_table").id
    self.cap = m.geom("cap0/cap0_vis").id
    self.mug = [m.geom(f"mug/{g}").id for g in ("mug_body_vis", "mug_handle_vis")]
    printed = np.array(twin.twin().model.COLORS["printed"][:3])
    self.arm = [
      i
      for i in range(m.ngeom)
      if m.geom_group[i] == twin.VISUAL_GROUP
      and m.geom_bodyid[i] != m.body("cap0/cap0").id
      and np.allclose(m.geom_rgba[i, :3], printed, atol=1e-3)
    ]
    self.cams = {c: m.camera(f"robot/{c}").id for c in CAMS}
    self.nom = {
      k: getattr(m, k).copy()
      for k in (
        "light_diffuse",
        "light_dir",
        "mat_rgba",
        "geom_rgba",
        "cam_pos",
        "cam_quat",
      )
    }
    self.ambient = m.vis.headlight.ambient.copy()

  def randomize(self, rng: np.random.Generator) -> dict:
    m, n, d = self.model, self.nom, self.dr
    for k, v in n.items():
      getattr(m, k)[:] = v
    v = d.draw("visual", rng)["visual"]
    m.light_diffuse[self.key] = np.clip(
      n["light_diffuse"][self.key]
      * v["light_gain"]
      * (1 + np.asarray(v["light_tint"])),
      0,
      2,
    )
    q = d._small_rot([v["light_tilt"][0], v["light_tilt"][1], 0.0])
    l2 = d._quat_mul(
      d._quat_mul(q, np.r_[0, n["light_dir"][self.key]]), q * [1, -1, -1, -1]
    )[1:]
    m.light_dir[self.key] = l2 / np.linalg.norm(l2)
    m.vis.headlight.ambient[:] = self.ambient * v["ambient_gain"]
    m.mat_rgba[self.mat, :3] = np.clip(
      n["mat_rgba"][self.mat, :3] * v["mat_gain"], 0, 1
    )
    m.geom_rgba[self.cap, :3] = v["cap_rgb"]
    for i in self.mug:
      m.geom_rgba[i, :3] = np.clip(n["geom_rgba"][i, :3] * v["mug_gain"], 0, 1)
    for i in self.arm:
      m.geom_rgba[i, :3] = np.clip(n["geom_rgba"][i, :3] * v["arm_gain"], 0, 1)
    for c, cid in self.cams.items():
      m.cam_pos[cid] = n["cam_pos"][cid] + np.asarray(v[f"{c}_cam_mm"]) / 1000
      m.cam_quat[cid] = d._quat_mul(n["cam_quat"][cid], d._small_rot(v[f"{c}_cam_deg"]))
    return v

  def episode(self, qpos: np.ndarray, rng: np.random.Generator) -> dict:
    """front, grip (L, h, w, 3) uint8 for an episode's (L, nq) local qpos; the draw."""
    v = self.randomize(rng)
    out = {c: np.empty((len(qpos), h, w, 3), np.uint8) for c, (w, h) in CAMS.items()}
    for k, q in enumerate(qpos):
      self.data.qpos[:] = q
      mujoco.mj_kinematics(self.model, self.data)
      mujoco.mj_camlight(self.model, self.data)
      for c, (w, h) in CAMS.items():
        r = self.r[c]
        r.update_scene(self.data, camera=self.cams[c])
        img = r.render().reshape(h, SS, w, SS, 3).mean((1, 3))
        out[c][k] = np.round(img).astype(np.uint8)
    return {**out, "visual": v}


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
  p.add_argument("--seed", type=int, default=0, help="the visual draws' seed")
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  torch.manual_seed(a.seed)
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
  mj = env.sim.mj_model
  free = [  # the free joints' xyz: cap and mug positions, made relative to their env
    int(mj.jnt_qposadr[j]) + k
    for j in range(mj.njnt)
    if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    for k in range(3)
  ]
  renderer = TwinRenderer(env_cfg(1), mj.nq)

  T = int(EPISODE_S * FPS) + 2
  shapes = {
    "qpos": ((env.sim.mj_model.nq,), torch.float32),
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
      qpos = env.sim.data.qpos.clone()
      qpos[:, free] -= env.scene.env_origins.repeat(1, len(free) // 3)
      buf["qpos"][rows, ti] = qpos
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
          index=saved,
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
        for e in pending:
          ep_rng = np.random.default_rng([a.seed, e["meta"]["index"]])
          r = renderer.episode(e["steps"]["qpos"], ep_rng)
          e["steps"].update(front=r["front"], grip=r["grip"])
          e["meta"]["visual"] = json.dumps(r["visual"])
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
    "renderer": f"MuJoCo {mujoco.__version__}, textured, {SS}x supersampled",
    "visual_dr": "real2sim.live.dr.draw('visual'), one draw per episode",
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
