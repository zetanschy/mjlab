"""Is mjlab's twin the twin? Static equality, then the twin's recorded grasps.

  uv run python scripts/so101_twin/check_twin.py static
  uv run python scripts/so101_twin/check_twin.py replay [--substeps 60 30 20 15 10]
  uv run python scripts/so101_twin/check_twin.py throughput --substeps 20 --envs 1024 4096

(env -u PYTHONPATH first where ROS is sourced.)

static      the mjlab scene (twin.scene_cfg) against the twin's own build(): joints,
            actuators, arm masses/inertias, finger pieces, cap, mug, table, cameras.
replay      the twin's 6 recorded grasps (scene episodes 0-2), open loop, in the twin's
            "action" mode: every servo gets the recorded action through the twin's
            GoalStream (firmware clamp, identified dead time and slew limit), objects
            settle 0.3 s first with the arm held -- replay.run's reset, step for step.
              1. this harness on the twin's model == the twin's replay.run (bitwise)
              2. the mjlab model on CPU, same dt == the twin's model
              3. the mjlab model on the GPU (mujoco_warp) at each --substeps: does each
                 grasp lift and hold the cap as the twin does?
            Per grasp: lift (max cap rise after the close, mm) and hold (longest span
            above +10 mm, s).
throughput  physics only: control steps per second at N envs (random targets near the
            arm's rest pose), at the given substeps.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.so_arm101 import twin as T
from mjlab.scene import Scene
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

T.twin()  # puts agent_101/sim on sys.path for the real2sim imports below

PARK_MUG = (0.6, -0.75)  # on the table, out of reach: the recorded episodes have no mug
SETTLE_S = 0.3  # replay.Config.settle_s
EPISODES = (0, 1, 2)


# --- episodes ------------------------------------------------------------------------


def episode(ep: int) -> dict:
  """Recorded goals and state (URDF rad), the caps as the twin places them, picks."""
  from real2sim import episodes as E

  t, sc = T.twin(), T.scene()
  e = E.load(sc.ds, include_excluded=True)[ep]
  u = sc.units()
  h = sc.cap_dims()["height"]
  caps = []
  for o in t.objects.episode_objects(sc, ep):
    if o["kind"] != "cap":
      continue
    pos = np.array(o["pos"], dtype=float)
    pos[2] = float(t.model.table_top(sc, *pos[:2])) + (h if o["up"] == "open" else 0.0)
    caps.append(
      {"id": o["name"].split("_", 1)[1], "pos": pos, "quat": np.array(o["quat"], float)}
    )
  mug = mug_pose()  # parked when the episode has none
  for o in t.objects.episode_objects(sc, ep):
    if o["kind"] == "mug":
      pos = np.array(o["pos"], dtype=float)
      pos[2] = float(t.model.table_top(sc, *pos[:2]))
      mug = np.r_[pos, np.array(o["quat"], dtype=float)]
  order = {c["id"]: i for i, c in enumerate(caps)}
  picks = [
    (order[p["cap"]], int(p["close"][0])) for p in sc.episode(ep).get("picks", [])
  ]
  return {
    "ep": ep,
    "fps": float(e.fps),
    "state": u.to_urdf(e.state),
    "action": u.to_urdf(e.action),
    "caps": caps,
    "mug": mug,
    "picks": picks,
    "limits": _fw_limits(u),
  }


def _fw_limits(u):
  from real2sim.mujoco.servo import firmware_limits_rad

  return firmware_limits_rad(u)


def mug_pose() -> np.ndarray:
  sc = T.scene()
  return np.r_[PARK_MUG, float(T.twin().model.table_top(sc, *PARK_MUG)), 1.0, 0, 0, 0]


# --- steppers: one interface over the twin, mjlab-on-CPU and mjlab-on-GPU ------------


class Ids:
  """qpos/dof/actuator addresses and cap bodies of a compiled model."""

  def __init__(self, m: mujoco.MjModel, prefix: str, caps: list[str], mug: str | None):
    J = list(T.twin().units.URDF_JOINTS)
    self.qadr = np.array([m.jnt_qposadr[m.joint(prefix + n).id] for n in J])
    self.dadr = np.array([m.jnt_dofadr[m.joint(prefix + n).id] for n in J])
    self.act = np.array([m.actuator(prefix + n).id for n in J])
    fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + T.twin().FLEX_JOINT)
    self.flex = (int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])) if fj >= 0 else None
    self.cap_body = [m.body(c).id for c in caps]
    self.cap_q = [int(m.jnt_qposadr[m.body_jntadr[b]]) for b in self.cap_body]
    self.cap_d = [int(m.jnt_dofadr[m.body_jntadr[b]]) for b in self.cap_body]
    self.mug_q = None
    if mug is not None:
      b = m.body(mug).id
      self.mug_q = int(m.jnt_qposadr[m.body_jntadr[b]])


class CpuStepper:
  def __init__(self, m: mujoco.MjModel, ids: Ids):
    self.m, self.d, self.i = m, mujoco.MjData(m), ids

  def place(self, q0, caps, mug=None) -> None:
    d, i = self.d, self.i
    d.qpos[i.qadr], d.qvel[i.dadr], d.ctrl[i.act] = q0, 0.0, q0
    if i.flex:
      d.qpos[i.flex[0]], d.qvel[i.flex[1]] = 0.0, 0.0
    for c, a, da in zip(caps, i.cap_q, i.cap_d, strict=True):
      d.qpos[a : a + 3], d.qpos[a + 3 : a + 7], d.qvel[da : da + 6] = (
        c["pos"],
        c["quat"],
        0.0,
      )
    if mug is not None and i.mug_q is not None:
      d.qpos[i.mug_q : i.mug_q + 7] = mug
    mujoco.mj_forward(self.m, d)

  def hold_step(self, q0) -> None:
    mujoco.mj_step(self.m, self.d)
    self.d.qpos[self.i.qadr], self.d.qvel[self.i.dadr] = q0, 0.0

  def arm_reset(self, q0) -> None:
    d, i = self.d, self.i
    d.qpos[i.qadr], d.qvel[i.dadr] = q0, 0.0
    if i.flex:
      d.qpos[i.flex[0]], d.qvel[i.flex[1]] = 0.0, 0.0
    d.time = 0.0
    mujoco.mj_forward(self.m, d)

  def step(self, ctrl) -> None:
    self.d.ctrl[self.i.act] = ctrl
    mujoco.mj_step(self.m, self.d)
    self.maxcon = max(getattr(self, "maxcon", 0), self.d.ncon)

  def caps(self) -> np.ndarray:
    return self.d.xpos[self.i.cap_body].copy()

  def arm(self) -> np.ndarray:
    return self.d.qpos[self.i.qadr].copy()


class GpuStepper(CpuStepper):
  """One world of an mjlab Simulation (mujoco_warp), same interface."""

  def __init__(self, sim: Simulation, ids: Ids):
    self.sim, self.i = sim, ids
    self.dev = sim.device
    self._t = lambda a: torch.as_tensor(
      np.asarray(a), dtype=torch.float32, device=self.dev
    )

  def place(self, q0, caps, mug=None) -> None:
    qp, qv, i = self.sim.data.qpos, self.sim.data.qvel, self.i
    qp[0, i.qadr], qv[0, i.dadr] = self._t(q0), 0.0
    self.sim.data.ctrl[0, i.act] = self._t(q0)
    if i.flex:
      qp[0, i.flex[0]], qv[0, i.flex[1]] = 0.0, 0.0
    for c, a, da in zip(caps, i.cap_q, i.cap_d, strict=True):
      qp[0, a : a + 3], qp[0, a + 3 : a + 7] = self._t(c["pos"]), self._t(c["quat"])
      qv[0, da : da + 6] = 0.0
    if mug is not None and i.mug_q is not None:
      qp[0, i.mug_q : i.mug_q + 7] = self._t(mug)
    self.sim.forward()

  def hold_step(self, q0) -> None:
    self.sim.step()
    self.sim.data.qpos[0, self.i.qadr] = self._t(q0)
    self.sim.data.qvel[0, self.i.dadr] = 0.0

  def arm_reset(self, q0) -> None:
    qp, qv, i = self.sim.data.qpos, self.sim.data.qvel, self.i
    qp[0, i.qadr], qv[0, i.dadr] = self._t(q0), 0.0
    if i.flex:
      qp[0, i.flex[0]], qv[0, i.flex[1]] = 0.0, 0.0
    self.sim.forward()

  def step(self, ctrl) -> None:
    self.sim.data.ctrl[0, self.i.act] = self._t(ctrl)
    self.sim.step()

  def caps(self) -> np.ndarray:
    return self.sim.data.xpos[0, self.i.cap_body].double().cpu().numpy()

  def arm(self) -> np.ndarray:
    return self.sim.data.qpos[0, self.i.qadr].double().cpu().numpy()


def replay(st: CpuStepper, e: dict, substeps: int, dt: float, mug=None) -> dict:
  """replay.run's 'action' mode with this stepper. Returns cap (T, C, 3), q (T, 6)."""
  from real2sim.mujoco.servo import GoalStream

  servo = T.twin().ServoModel.from_scene(T.scene())
  state, fps = e["state"], e["fps"]
  st.place(state[0], e["caps"], mug)
  for _ in range(int(round(SETTLE_S / dt))):
    st.hold_step(state[0])
  st.arm_reset(state[0])
  stream = GoalStream(
    e["action"], fps, servo.dead_time, servo.max_velocity, e["limits"], q0=state[0]
  )
  n = len(state)
  cap, q = np.empty((n, len(e["caps"]), 3)), np.empty((n, 6))
  t0 = time.time()
  nan_frame = None
  for k in range(n):
    cap[k], q[k] = st.caps(), st.arm()
    if not (np.isfinite(cap[k]).all() and np.isfinite(q[k]).all()):
      nan_frame = k  # stop: stepping on past a NaN corrupts memory in mujoco_warp
      cap[k:], q[k:] = np.nan, np.nan
      break
    if k == n - 1:
      break
    for s in range(substeps):
      st.step(stream.at((k + s / substeps) / fps))
  return {
    "cap": cap,
    "q": q,
    "wall": time.time() - t0,
    "sim_s": (n - 1) / fps,
    "maxcon": getattr(st, "maxcon", None),
    "nan_frame": nan_frame,
  }


def grasps(r: dict, e: dict) -> list[dict]:
  """Per pick: lift = max rise of its cap after the close starts (mm); hold = longest
  span above +10 mm (s), up to the next pick's close."""
  out = []
  starts = [c0 for _, c0 in e["picks"]] + [len(r["cap"])]
  for j, (ci, c0) in enumerate(e["picks"]):
    z = r["cap"][c0 : starts[j + 1], ci, 2] - r["cap"][c0, ci, 2]
    up = z > 0.010
    run = best = 0
    for v in up:
      run = run + 1 if v else 0
      best = max(best, run)
    k = min(c0 + 30, starts[j + 1] - 1)  # 1 s after the close started
    out.append(
      {
        "cap": e["caps"][ci]["id"],
        "lift_mm": 1000 * float(np.nanmax(z)) if np.isfinite(z).any() else float("nan"),
        "hold_s": best / e["fps"],
        "jaw_deg": float(np.degrees(r["q"][k, 5])),
      }
    )
  return out


# --- the mjlab model ------------------------------------------------------------------


def mj_overrides(a) -> dict:
  return {
    k: v
    for k, v in (
      ("iterations", a.iterations),
      ("ls_iterations", a.ls_iterations),
      ("tolerance", a.tolerance),
      ("impratio", a.impratio),
    )
    if v is not None
  }


def broadphase(a) -> dict:
  kw = {}
  if a.broadphase:
    kw["broadphase"] = a.broadphase
  if a.broadphase_filter:
    kw["broadphase_filter"] = tuple(a.broadphase_filter)
  return kw


def mjlab_model(substeps: int, n_caps: int, **mj) -> mujoco.MjModel:
  cfg = T.scene_cfg(n_caps=n_caps, substeps=substeps)
  m = Scene(cfg, device="cpu").compile()
  MujocoCfg(**T.mujoco_cfg_kwargs(substeps, **mj)).apply(m)
  return m


def mjlab_ids(m, n_caps: int) -> Ids:
  return Ids(m, "robot/", [f"cap{i}/cap{i}" for i in range(n_caps)], "mug/mug")


# --- commands ---------------------------------------------------------------------------


def cmd_static(_a) -> int:
  t, sc = T.twin(), T.scene()
  m = mjlab_model(T.TWIN_SUBSTEPS, 1)
  r = t.model.build(sc, 0, t.ServoModel.from_scene(sc), T.physics()).model
  J = list(t.units.URDF_JOINTS) + [t.FLEX_JOINT]

  def joints(mm, p):
    return np.array(
      [
        np.r_[
          mm.joint(p + n).range,
          mm.dof_damping[mm.joint(p + n).dofadr[0]],
          mm.dof_armature[mm.joint(p + n).dofadr[0]],
          mm.dof_frictionloss[mm.joint(p + n).dofadr[0]],
          mm.dof_solref[mm.joint(p + n).dofadr[0]],
          mm.jnt_stiffness[mm.joint(p + n).id],
        ]
        for n in J
      ]
    )

  def acts(mm, p):
    return np.array(
      [
        np.r_[
          mm.actuator(p + n).gainprm[:3],
          mm.actuator(p + n).biasprm[:3],
          mm.actuator(p + n).forcerange,
        ]
        for n in J[:6]
      ]
    )

  bodies = [
    "shoulder",
    "upper_arm",
    "lower_arm",
    "wrist",
    "gripper",
    "jaw",
    "klip_support",
    "kwc500",
  ]

  def masses(mm, p):
    return np.array([[mm.body(p + b).mass[0], *mm.body(p + b).inertia] for b in bodies])

  def geoms(mm, names):
    return np.array(
      [
        np.r_[
          mm.geom(n).friction,
          mm.geom(n).solref,
          mm.geom(n).solimp[:3],
          mm.geom(n).condim,
          mm.geom(n).size,
        ]
        for n in names
      ]
    )

  fm = [m.geom(i).name for i in range(m.ngeom) if "finger_" in m.geom(i).name]
  fr = [r.geom(i).name for i in range(r.ngeom) if "finger_" in r.geom(i).name]
  rows = [
    (
      "joints (range, damping, armature, frictionloss, solref, flex stiffness)",
      joints(m, "robot/") - joints(r, ""),
    ),
    ("actuators (gain, bias, force range)", acts(m, "robot/") - acts(r, "")),
    ("arm masses and inertias", masses(m, "robot/") - masses(r, "")),
    (
      f"finger pieces ({len(fm)} vs {len(fr)})",
      geoms(m, fm) - geoms(r, fr) if len(fm) == len(fr) else np.inf,
    ),
    (
      "cap collision",
      geoms(m, ["cap0/cap0_collision"]) - geoms(r, ["cap_A_collision"]),
    ),
    (
      "table",
      np.r_[
        m.geom("robot/table").pos - r.geom("table").pos,
        m.geom("robot/table").friction - r.geom("table").friction,
      ],
    ),
  ]
  ok = True
  for name, d in rows:
    v = float(np.max(np.abs(d)))
    ok &= v == 0.0
    print(f"  {'ok ' if v == 0 else 'DIFF'} {name}: max |diff| {v:g}")
  for c in ("front", "grip"):
    i = m.camera("robot/" + c).id
    f = m.cam_intrinsic[i][:2] / m.cam_sensorsize[i] * m.cam_resolution[i]
    k = sc.camera(c)
    print(
      f"  cam {c}: fx fy {f.round(1)} vs calib {k.fx:.1f} {k.fy:.1f}, undistorted (calib dist {k.dist[:2]})"
    )
  return 0 if ok else 1


def cmd_replay(a) -> int:
  from real2sim.mujoco import replay as twin_replay

  t, sc = T.twin(), T.scene()
  servo = t.ServoModel.from_scene(sc)
  results = {}
  for ep in EPISODES:
    e = episode(ep)
    nc = len(e["caps"])
    # 1. the twin, two ways: its own replay.run and this harness on its model
    b = t.model.build(sc, ep, servo, T.physics())
    cache = (
      Path(os.environ.get("TWIN_REF_CACHE", "/tmp")) / f"twin_ref_{sc.hash}_ep{ep}.npz"
    )
    if a.skip_checks and cache.exists():  # the twin's reference is deterministic
      ref = dict(np.load(cache))
    else:
      tw_ids = Ids(
        b.model, "", [o["body_name"] for o in b.objects if o["kind"] == "cap"], None
      )
      ref = replay(
        CpuStepper(b.model, tw_ids), e, T.TWIN_SUBSTEPS, b.model.opt.timestep
      )
      np.savez(cache, cap=ref["cap"], q=ref["q"])
    if not a.skip_checks:
      ref_run = twin_replay.run(
        sc, ep, twin_replay.Config(mode="action"), servo=servo, warn=None
      )
      capcols = [
        i for i, n in enumerate(ref_run.log["objects"]) if str(n).startswith("cap")
      ]
      d_q = np.abs(ref["q"] - ref_run.log["q"]).max()
      d_o = np.abs(ref["cap"] - ref_run.log["obj_pos"][:, capcols]).max()
      print(
        f"ep{ep}: harness vs twin replay.run: arm {d_q:.1e} rad, caps {d_o:.1e} m",
        flush=True,
      )
    results[ep] = {"twin": grasps(ref, e)}
    # 2. mjlab model, CPU, the twin's dt
    cols_cpu = not a.skip_checks
    if not cols_cpu:
      results[ep]["mjlab_cpu"] = results[ep]["twin"]
    m = mjlab_model(T.TWIN_SUBSTEPS, nc) if cols_cpu else None
    cpu = (
      replay(
        CpuStepper(m, mjlab_ids(m, nc)), e, T.TWIN_SUBSTEPS, m.opt.timestep, e["mug"]
      )
      if cols_cpu
      else ref
    )
    d_c = np.abs(cpu["cap"] - ref["cap"]).max()
    cols_cpu and print(
      f"ep{ep}: mjlab CPU vs twin: cap max |diff| {1000 * d_c:.3f} mm, arm {np.degrees(np.abs(cpu['q'] - ref['q']).max()):.3f} deg; peak contacts {cpu['maxcon']}",
      flush=True,
    )
    if cols_cpu:
      results[ep]["mjlab_cpu"] = grasps(cpu, e)
    # 3. mjlab on the GPU at each dt
    for s in a.substeps:
      m = mjlab_model(s, nc, **mj_overrides(a))
      kw = T.mujoco_cfg_kwargs(s, **mj_overrides(a))
      cfg = SimulationCfg(
        nconmax=a.nconmax, njmax=a.njmax, mujoco=MujocoCfg(**kw), **broadphase(a)
      )
      sim = Simulation(num_envs=1, cfg=cfg, model=m, device=a.device)
      g = replay(GpuStepper(sim, mjlab_ids(m, nc)), e, s, m.opt.timestep, e["mug"])
      results[ep][f"gpu_{s}"] = grasps(g, e)
      nan = (
        f"  UNSTABLE: NaN at frame {g['nan_frame']}"
        if g["nan_frame"] is not None
        else ""
      )
      print(
        f"ep{ep}: GPU {s} substeps (dt 1/{30 * s}): {g['wall'] / g['sim_s']:.2f} s wall per sim s, "
        f"cap vs twin max |diff| {1000 * np.nanmax(np.abs(g['cap'] - ref['cap'])):.1f} mm{nan}",
        flush=True,
      )
      del sim
  print("\nper grasp: lift mm / hold s / jaw deg 1 s after the close")
  cols = ["twin", "mjlab_cpu"] + [f"gpu_{s}" for s in a.substeps]
  print("  ep cap  " + "  ".join(f"{c:>14s}" for c in cols))
  for ep, r in results.items():
    for j in range(len(r["twin"])):
      cells = [
        f"{r[c][j]['lift_mm']:5.1f}/{r[c][j]['hold_s']:3.1f}/{r[c][j]['jaw_deg']:5.1f}"
        for c in cols
      ]
      print(
        f"  {ep:2d} {r['twin'][j]['cap']:3s}  " + "  ".join(f"{x:>17s}" for x in cells)
      )
  return 0


def cmd_throughput(a) -> int:
  for s in a.substeps:
    m = mjlab_model(s, 1, **mj_overrides(a))
    ids = mjlab_ids(m, 1)
    for n in a.envs:
      kw = T.mujoco_cfg_kwargs(s, **mj_overrides(a))
      cfg = SimulationCfg(
        nconmax=a.nconmax, njmax=a.njmax, mujoco=MujocoCfg(**kw), **broadphase(a)
      )
      sim = Simulation(num_envs=n, cfg=cfg, model=m, device=a.device)
      sim.create_graph()
      # rest_state is in the robot's units (degrees, gripper %): to URDF radians
      sc = T.scene()
      rest_rad = sc.units().to_urdf(np.asarray(T.twin().layouts.rest_state(sc))[None])[
        0
      ]
      q0 = rest_rad if a.pose == "rest" else episode(0)["state"][150]
      rest = torch.as_tensor(q0, dtype=torch.float32, device=a.device)
      act = torch.as_tensor(ids.act, device=a.device)
      sim.data.qpos[:, torch.as_tensor(ids.qadr, device=a.device)] = rest
      sim.forward()
      if (
        a.motion == "episodes"
      ):  # each env plays recorded commands from a random offset
        A = torch.as_tensor(
          np.concatenate([episode(ep)["action"] for ep in EPISODES]),
          dtype=torch.float32,
          device=a.device,
        )
        start = torch.randint(0, len(A), (n,), device=a.device)
        sim.data.qpos[:, torch.as_tensor(ids.qadr, device=a.device)] = A[start]
        sim.forward()
      steps, t0 = 0, None
      for it in range(a.control_steps + 5):
        if it == 5:
          torch.cuda.synchronize()
          t0 = time.time()
        if a.motion == "episodes":
          sim.data.ctrl[:, act] = A[(start + it) % len(A)]
        else:
          sim.data.ctrl[:, act] = rest + 0.2 * (torch.rand(n, 6, device=a.device) - 0.5)
        for _ in range(s):
          sim.step()
        steps += it >= 5
      torch.cuda.synchronize()
      dt_w = time.time() - t0
      bad = int((~torch.isfinite(sim.data.qpos).all(dim=1)).sum())
      print(
        f"substeps {s} (dt 1/{30 * s}), {n} envs: {steps * n / dt_w:,.0f} env control steps/s "
        f"({steps * n / dt_w / 30:,.0f} x real time), {bad} of {n} worlds non-finite",
        flush=True,
      )
      del sim
      torch.cuda.empty_cache()
  return 0


def main() -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("cmd", choices=["static", "replay", "throughput"])
  p.add_argument("--substeps", type=int, nargs="+", default=[60, 30, 20, 15, 10])
  p.add_argument("--envs", type=int, nargs="+", default=[1024, 4096])
  p.add_argument("--control-steps", type=int, default=30)
  p.add_argument(
    "--motion",
    choices=["rest", "episodes"],
    default="rest",
    help="throughput: random targets near rest, or the recorded commands",
  )
  p.add_argument("--nconmax", type=int, default=128)
  p.add_argument("--njmax", type=int, default=1200)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--iterations", type=int, help="solver iterations (twin: 50)")
  p.add_argument("--ls-iterations", type=int, help="line-search iterations (twin: 50)")
  p.add_argument("--tolerance", type=float, help="solver tolerance (twin: 1e-10)")
  p.add_argument(
    "--impratio", type=float, help="friction/normal impedance ratio (twin: 100)"
  )
  p.add_argument(
    "--pose",
    choices=["rest", "reach"],
    default="rest",
    help="throughput 'rest' motion: around the folded rest pose or a recorded reach",
  )
  p.add_argument(
    "--no-flex", action="store_true", help="drop the Jaw's series flex hinge"
  )
  p.add_argument("--broadphase", choices=["nxn", "sap_tile", "sap_segmented"])
  p.add_argument(
    "--broadphase-filter", nargs="+", choices=["plane", "sphere", "aabb", "obb"]
  )
  p.add_argument(
    "--fingers",
    choices=["coacd", "hull"],
    help="finger collision, the twin's options: coacd (default) or hull",
  )
  p.add_argument(
    "--skip-checks",
    action="store_true",
    help="skip replay.run and mjlab-on-CPU (both proven); twin reference + GPU only",
  )
  a = p.parse_args()
  if a.no_flex:
    T.PHYSICS_OVERRIDES["flex"] = False  # the twin's own option: no series Jaw_flex
  if a.fingers:
    T.PHYSICS_OVERRIDES["fingers"] = a.fingers  # the twin's own options
  return {"static": cmd_static, "replay": cmd_replay, "throughput": cmd_throughput}[
    a.cmd
  ](a)


if __name__ == "__main__":
  raise SystemExit(main())
