"""Score an mjlab cap-to-mug checkpoint on the benchmark's seeded stages.

  uv run python scripts/so101_twin/eval_teacher.py --checkpoint model_12299.pt \
      --stages ../../outputs/sim_eval/openpi_pi05_lora_cap_to_mug_sim_50/trials.jsonl \
      --out ../../outputs/sim_eval/mjlab_teacher_12299

(env -u PYTHONPATH MUJOCO_GL=egl first.) --student student.pt instead of --checkpoint
scores a student.py student: same stages and grading, in collect.py's env (its two
cameras and the dataset's visual randomization).

THE STAGES are the ones every pi0.5 model was scored on by ./robot sim-eval (seeds
1000-1099): their cap xy, cap orientation (open / closed side up), mug xy and handle
yaw, read from any sim-eval run's trials.jsonl. One env per stage, all at once.

THE GRADING is sim-eval's: success = the twin's cap_in_mug held 1 s, within 45 s. The
teacher's own criterion (the cap in the mug AND the arm back within 0.3 rad of rest) is
reported alongside, as `home`. Failures get a cause: the mug tipped or the cap lost
(the task's terminations), else never picked up (never 2 cm off the table), still
holding at the timeout, or other.

THE OUTPUT is in sim-eval's format (run.json + trials.jsonl), with where="mjlab", so
`./robot sim-eval --compare <out> <a sim-eval run>` gives the paired sign test. This is
the mjlab twin (dt 1/900, the RL settings), not the live twin sim-eval drives (1/1800):
the same model and stages, a coarser step.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import collect
import torch
from student import Student

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.config.so101 import cap_to_mug_cfgs as cfgs
from mjlab.tasks.manipulation.config.so101 import cap_to_mug_mdp as m
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

TASK = "Mjlab-CapToMug-So101-Twin"
FPS = 30


def load_stages(path: Path) -> list[dict]:
  rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  rows.sort(key=lambda r: r["seed"])
  return [
    {k: r[k] for k in ("seed", "cap_xy", "cap_up", "mug_xy", "handle_yaw_deg")}
    for r in rows
  ]


def place(env: ManagerBasedRlEnv, stages: list[dict]) -> None:
  """Each env's cap and mug where its stage puts them, as reset_layout would."""
  dev, n = env.device, len(stages)
  c = m.constants()
  ids = torch.arange(n, device=dev)
  cap_xy = torch.tensor([s["cap_xy"] for s in stages], dtype=torch.float32, device=dev)
  mug_xy = torch.tensor([s["mug_xy"] for s in stages], dtype=torch.float32, device=dev)
  is_open = torch.tensor([s["cap_up"] == "open" for s in stages], device=dev)
  yaw = torch.tensor([math.radians(s["handle_yaw_deg"]) for s in stages], device=dev)
  origin = env.scene.env_origins[ids]
  cap = torch.zeros(n, 13, device=dev)
  cap_z = m._table_z(cap_xy) + torch.where(is_open, c["cap_h"], 0.0)
  cap[:, :3] = origin + torch.cat([cap_xy, cap_z[:, None]], dim=-1)
  cap[:, 3] = 1.0
  cap[is_open, 3:7] = torch.tensor(m.FLIP_X, device=dev)
  mug = torch.zeros(n, 13, device=dev)
  mug[:, :3] = origin + torch.cat([mug_xy, m._table_z(mug_xy)[:, None]], dim=-1)
  mug[:, 3], mug[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
  env.scene[m.CAP].write_root_state_to_sim(cap, env_ids=ids)
  env.scene[m.MUG].write_root_state_to_sim(mug, env_ids=ids)
  m._state(env)["mug_xy0"][ids] = mug_xy
  env.sim.forward()
  env.scene.update(0.0)


def main() -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--checkpoint", type=Path, help="the teacher's model_N.pt")
  p.add_argument(
    "--student", type=Path, help="a student.py .pt instead: scored with its cameras"
  )
  p.add_argument(
    "--stages", required=True, type=Path, help="a sim-eval run's trials.jsonl"
  )
  p.add_argument("--out", required=True, type=Path)
  p.add_argument("--timeout", type=float, default=45.0)
  p.add_argument("--settle", type=float, default=1.0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  if (a.checkpoint is None) == (a.student is None):
    p.error("give one of --checkpoint or --student")

  stages = load_stages(a.stages)
  n = len(stages)
  # The student gets the dataset's env: its cameras and visual DR.
  env_cfg = (
    collect.env_cfg(n)
    if a.student
    else cfgs.so101_cap_to_mug_env_cfg(play=True, num_envs=n)
  )
  env_cfg.episode_length_s = a.timeout + 5.0  # our own timeout decides, not the task's
  env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device)
  agent_cfg = load_rl_cfg(TASK)
  vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if a.student:
    student = Student.load(a.student, a.device)
    policy = lambda _obs: student.act(env)  # noqa: E731
  else:
    runner_cls = load_runner_cls(TASK)
    assert runner_cls is not None
    runner = runner_cls(vec, asdict(agent_cfg), device=a.device)
    runner.load(
      str(a.checkpoint), load_cfg={"actor": True}, strict=True, map_location=a.device
    )
    policy = runner.get_inference_policy(device=a.device)

  vec.reset()
  place(env, stages)
  obs = vec.get_observations()

  dev = env.device
  active = torch.ones(n, dtype=torch.bool, device=dev)
  success = torch.zeros(n, dtype=torch.bool, device=dev)
  home = torch.zeros(n, dtype=torch.bool, device=dev)
  in_run = torch.zeros(n, dtype=torch.long, device=dev)
  t_succ = torch.full((n,), float("nan"), device=dev)
  max_lift = torch.zeros(n, device=dev)
  cause = [""] * n
  t_end = torch.full((n,), a.timeout, device=dev)
  steps, settle = int(a.timeout * FPS), int(a.settle * FPS)
  lat: list[float] = []  # the policy's own forward time, one batched call per step
  with torch.inference_mode():
    for k in range(steps):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      act = policy(obs)
      torch.cuda.synchronize()
      lat.append(time.perf_counter() - t0)
      obs, _, dones, _ = vec.step(act)
      t = (k + 1) / FPS
      inside = m.cap_in_mug(env)
      in_run = torch.where(inside & active, in_run + 1, torch.zeros_like(in_run))
      new = active & ~success & (in_run >= settle)
      t_succ[new] = t - settle / FPS
      success |= new
      home |= success & active & (m.arm_to_rest(env) < 0.3)
      max_lift = torch.where(
        active, torch.maximum(max_lift, m.lift_height(env)), max_lift
      )
      ended = active & dones.bool()
      if ended.any():
        tipped = env.termination_manager.get_term("mug_tipped")
        lost = env.termination_manager.get_term("cap_lost")
        for i in ended.nonzero().flatten().tolist():
          if not success[i]:
            cause[i] = (
              "mug knocked over"
              if tipped[i]
              else "cap fell off the table"
              if lost[i]
              else "other"
            )
          t_end[i] = t
        active &= ~ended
      if not active.any():
        break
    held = m.fingers_on_cap(env).all(dim=-1)

  a.out.mkdir(parents=True, exist_ok=True)
  lat_ms = round(1000 * sorted(lat)[len(lat) // 2], 3)  # median, for all envs at once
  rows = []
  for i, s in enumerate(stages):
    ok = bool(success[i])
    c = (
      "success"
      if ok
      else cause[i]
      or (
        "never picked up"
        if float(max_lift[i]) < 0.02
        else "still holding at timeout"
        if bool(held[i])
        else "other"
      )
    )
    rows.append(
      {
        "trial": i,
        "seed": s["seed"],
        "success": ok,
        "cause": c,
        "t_success": round(float(t_succ[i]), 2) if ok else None,
        "duration": round(float(t_end[i]), 2),
        "cap_xy": s["cap_xy"],
        "cap_up": s["cap_up"],
        "mug_xy": s["mug_xy"],
        "handle_yaw_deg": s["handle_yaw_deg"],
        "home": bool(home[i]),
        "max_lift_cm": round(100 * float(max_lift[i]), 1),
        "latency_ms_mean": lat_ms,
      }
    )
  (a.out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
  run = {
    "policy": str(a.student or a.checkpoint),
    "task": TASK,
    "episodes": n,
    "seed": stages[0]["seed"],
    "timeout": a.timeout,
    "settle": a.settle,
    "where": "mjlab",
    "stages_from": str(a.stages),
    "dt": env_cfg.sim.mujoco.timestep,
    "decimation": env_cfg.decimation,
  }
  (a.out / "run.json").write_text(json.dumps(run, indent=2))
  k = sum(r["success"] for r in rows)
  print(
    f"\nsuccess {k}/{n} (cap in the mug for {a.settle:.0f} s within {a.timeout:.0f} s); "
    f"of those, back home: {sum(r['home'] for r in rows)}"
  )
  for up in ("closed", "open"):
    sub = [r for r in rows if r["cap_up"] == up]
    if sub:
      print(f"  cap {up} side up: {sum(r['success'] for r in sub)}/{len(sub)}")
  causes: dict[str, int] = {}
  for r in rows:
    causes[r["cause"]] = causes.get(r["cause"], 0) + 1
  print("  outcomes:", dict(sorted(causes.items(), key=lambda kv: -kv[1])))
  ts = sorted(r["t_success"] for r in rows if r["success"])
  if ts:
    print(
      f"  time to success: median {ts[len(ts) // 2]:.1f} s, range {ts[0]:.1f}-{ts[-1]:.1f} s"
    )
  print(f"  policy forward: median {lat_ms} ms per batched call ({n} envs)")
  print(f"wrote {a.out}/trials.jsonl and run.json")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
