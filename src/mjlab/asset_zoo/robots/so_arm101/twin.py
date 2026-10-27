"""The agent_101 digital twin as mjlab entities, built read-only.

The twin (agent_101/sim/real2sim) is the MuJoCo model the imitation-learning policies
were collected and benchmarked in. mjlab's own so101 asset predates it, and differs
where a cap grasp is decided:
  - servo: the identified kp, kd, damping, armature, frictionloss, effort per joint
    (servo.json), not kp 17.8 / kv 2.0;
  - fingers: the CoACD pieces. mjlab's single finger hulls let the jaw close past a
    2.5 g cap (sim/real2sim/mujoco/README.md);
  - joint ranges: the firmware range + 5 deg, Jaw down to its real shut stop (-11.5 deg);
  - the Jaw's series flex hinge, the webcam and mount masses, the measured table height;
  - cameras: the fitted front (world) and grip (gripper) poses and intrinsics.

NOTHING HERE EDITS THE TWIN. The arm is the twin's own MjSpec: build() is called as the
live sim calls it, and its spec_hook (meant for visual additions) hands over the
finished spec, of which this module keeps a COPY. The twin compiles and runs exactly
as before. The cap and the mug are built with the twin's own helpers, dims and contact
settings (real2sim.mujoco.model, real2sim.objects).

ENTITIES
  robot  the twin's arm with its actuators (wrap with XmlActuatorCfg), the table and mat,
         the key light, and two cameras, `front` and `grip`, as UNDISTORTED pinholes
         at the calibrated K and pose. The twin renders oversized pinholes and remaps
         them through the lens distortion. mujoco_warp renders pinholes only, so for
         low-res RL the REAL image is undistorted instead (cv2.undistort with the same
         K and dist) and both sides see the same ideal pinhole.
  cap    a free body: primitive cylinder collision, hollow-cap visual mesh and inertia.
  mug    a free body: floor cylinder + wall boxes + handle capsules, mesh visuals.

The arm keeps the twin's frame: base at the origin, forward along -y (mjlab's so101
yaws it +90 deg instead). Layouts, cameras and the table are all in this frame.

PHYSICS. dt = 1 / (30 * substeps). The twin sets object contacts and the servos' dry
friction to solref (2 dt, 1), so they follow dt: build every entity with the SAME
substeps as the simulation. Solver options do not survive mjlab's attach; set them
with `mujoco_cfg(substeps)`.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import SceneCfg

FPS = 30.0
TWIN_SUBSTEPS = 60  # the twin's own: dt = 1/1800 s

# RL settings, MEASURED against the twin (scripts/so101_twin/check_twin.py, RTX 3060):
#   dt 1/900 (30 substeps per 30 Hz frame). The twin's 6 recorded grasps lift and hold
#     as in the twin (lift within 2 mm, jaw angle within 0.3 deg); its own notes find
#     1/900 "the same outcomes" as 1/1800 with slightly more penetration. 1/600 and
#     coarser go NaN on some grasps with the twin's 2-dt contact time constant.
#   20 solver iterations at tolerance 1e-8 (twin: 50 at 1e-10): same grasps.
#   4096 worlds: ~8-9k env control steps/s, 0 non-finite over 5 s of random targets
#     around the rest pose or a reach pose, or of the recorded commands.
#   Contacts peak at ~30 per world in a grasp; nconmax 96 / njmax 1500 leave margin.
# The twin's fingers are kept: primitive-box proxies were no faster in realistic
# motion and held the cap 3-5 deg more open.
RL_SUBSTEPS = 30
RL_SOLVER = {"iterations": 20, "ls_iterations": 20, "tolerance": 1e-8}
RL_NCONMAX, RL_NJMAX = 96, 1500
AGENT101 = Path(os.environ.get("AGENT101_ROOT", Path(__file__).resolve().parents[7]))
VISUAL_GROUP, COLLISION_GROUP = 2, 3  # the twin's geom groups
# Where objects wait off reach (on the table, ~0.95 m from the base). The compiled
# default pose must not overlap the arm, or mujoco_warp starts with hundreds of
# contacts.
PARK_XY = (0.6, -0.75)


@functools.cache
def twin() -> SimpleNamespace:
  """The twin's modules, imported from agent_101/sim (on sys.path if it is not yet)."""
  sim_dir = str(AGENT101 / "sim")
  if sim_dir not in sys.path:
    sys.path.insert(0, sim_dir)
  # importlib, not import statements: real2sim lives in agent_101, not in this
  # package, and resolves only at runtime.
  mod = importlib.import_module
  servo = mod("real2sim.mujoco.servo")
  return SimpleNamespace(
    objects=mod("real2sim.objects"),
    paths=mod("real2sim.paths"),
    units=mod("real2sim.units"),
    scene_mod=mod("real2sim.scene"),
    to_mujoco=mod("real2sim.camera").to_mujoco,
    layouts=mod("real2sim.live.layouts"),
    model=mod("real2sim.mujoco.model"),
    ServoModel=servo.ServoModel,
    FLEX_JOINT=servo.FLEX_JOINT,
  )


@functools.cache
def scene(ds: str | None = None):
  """The twin's calibrated scene: the same `scene.load(paths.dataset(ds))` the live
  sim's server uses (ds None = its default dataset)."""
  t = twin()
  return t.scene_mod.load(t.paths.dataset(ds))


# Experiments only (scripts/so101_twin): Physics fields to vary from the twin's, e.g.
# {"fingers": "hull"} or {"flex": False}. Empty = the twin as is. Set it before
# building any entity.
PHYSICS_OVERRIDES: dict = {}


def physics(substeps: int = TWIN_SUBSTEPS):
  """The twin's Physics at `substeps` per 30 Hz frame, every other setting as is
  (unless PHYSICS_OVERRIDES says otherwise)."""
  return twin().model.Physics(substeps=int(substeps), **PHYSICS_OVERRIDES)


def mujoco_cfg_kwargs(substeps: int = TWIN_SUBSTEPS, **override) -> dict:
  """MujocoCfg fields equal to the twin's solver settings at this dt."""
  p = physics(substeps)
  kw = {
    "timestep": p.dt,
    "integrator": p.integrator,
    "cone": p.cone,
    "impratio": p.impratio,
    "iterations": p.iterations,
    "ls_iterations": p.ls_iterations,
    "tolerance": p.tolerance,
    "ccd_iterations": 35,  # MuJoCo's default, which the twin keeps
  }
  kw.update(override)
  return kw


def _undistorted_camera(spec: mujoco.MjSpec, sc, name: str) -> None:
  t = twin()
  cam = sc.camera(name)
  pin = replace(cam, dist=(0.0, 0.0, 0.0, 0.0, 0.0))
  mj = t.to_mujoco(pin)
  parent = spec.worldbody if pin.parent == "world" else spec.body(pin.parent)
  c = parent.add_camera(name=name, pos=mj["pos"], quat=mj["quat"])
  c.resolution = mj["resolution"]
  c.sensor_size = mj["sensorsize"]
  c.focal_pixel = mj["focalpixel"]
  c.principal_pixel = mj["principalpixel"]


def robot_spec(
  substeps: int = TWIN_SUBSTEPS,
  ds: str | None = None,
  cameras=("front", "grip"),
  look: bool = True,
) -> mujoco.MjSpec:
  """The twin's arm + table + light (+ the dataset's mat look) as an MjSpec copy, with
  undistorted `cameras`. No objects: the cap and mug are entities of their own."""
  t = twin()
  sc = scene(ds)
  look_hook = None
  if look:
    try:
      look_hook = importlib.import_module("real2sim.live.look_mujoco").hook(sc.ds)
    except FileNotFoundError:
      look_hook = None
  captured: dict = {}

  def hook(spec, s):
    if look_hook is not None:
      look_hook(spec, s)
    captured["spec"] = spec.copy()

  t.model.build(
    sc,
    None,
    t.ServoModel.from_scene(sc),
    physics(substeps),
    cameras=(),
    objects=False,
    spec_hook=hook,
  )
  spec = captured["spec"]
  spec.modelname = "so101_twin"
  _flat_textures(spec)
  for name in cameras:
    _undistorted_camera(spec, sc, name)
  return spec


def _flat_textures(spec: mujoco.MjSpec) -> None:
  """Each textured material becomes its texture's mean colour, untextured.

  mujoco_warp's renderer does not apply these material textures: the look's mat
  (a near-black texture on a white material) rendered white there, dark in MuJoCo's own
  renderer (front camera mean RGB ~196 vs ~32). Flat, both renderers show the same mat;
  only its fine texture is lost, invisible at the cameras' low resolutions anyway.
  """
  m = spec.copy().compile()
  for mat in spec.materials:
    tid = int(max(m.mat_texid[m.material(mat.name).id]))
    if tid < 0:
      continue
    n, a = int(m.tex_nchannel[tid]), int(m.tex_adr[tid])
    px = m.tex_data[a : a + int(m.tex_width[tid] * m.tex_height[tid]) * n]
    mean = px.reshape(-1, n)[:, :3].mean(0) / 255.0
    mat.rgba = [*(mean * np.asarray(mat.rgba[:3])), mat.rgba[3]]
    mat.textures = [""] * len(mat.textures)


def cap_spec(
  substeps: int = TWIN_SUBSTEPS,
  ds: str | None = None,
  name: str = "cap",
  pos=(0.0, 0.0, 0.0),
) -> mujoco.MjSpec:
  """One cap as the twin builds it (_add_objects), as a free body at `pos`. Its
  frame: z up through the cylinder axis, the closed end at z = 0 when standing
  closed side up; open side up is the same body flipped (layouts / episode_objects)."""
  t = twin()
  mdl, phys, pert = t.model, physics(substeps), t.model.Perturbation()
  d = scene(ds).cap_dims()
  spec = mujoco.MjSpec()
  spec.modelname = name
  vis = t.objects.cap_mesh(d)
  com, inertia = mdl._mass_props(vis, d["mass"])
  mdl._add_mesh(spec, f"{name}_visual", vis.vertices, vis.faces)
  b = spec.worldbody.add_body(name=name, pos=list(pos))
  b.add_freejoint(name=f"{name}_free")
  mdl._set_inertial(b, d["mass"], com, inertia)
  g = b.add_geom(
    name=f"{name}_collision",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[d["diameter"] / 2, d["height"] / 2, 0],
    pos=[0, 0, d["height"] / 2],
    group=COLLISION_GROUP,
    contype=1,
    conaffinity=1,
    mass=0,
    rgba=mdl.COLORS["cap"],
  )
  mdl._contact(g, phys, "cap", pert)
  b.add_geom(
    name=f"{name}_vis",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=f"{name}_visual",
    group=VISUAL_GROUP,
    contype=0,
    conaffinity=0,
    mass=0,
    rgba=mdl.COLORS["cap"],
  )
  return spec


def mug_spec(
  substeps: int = TWIN_SUBSTEPS, ds: str | None = None, pos=(0.0, 0.0, 0.0)
) -> mujoco.MjSpec:
  """The mug as the twin builds it: floor + wall boxes + handle capsules, mesh visuals,
  inertia of the mesh. Body frame: z up, the floor's underside at z = 0."""
  t = twin()
  mdl, phys, pert = t.model, physics(substeps), t.model.Perturbation()
  d = scene(ds).mug_dims()
  spec = mujoco.MjSpec()
  spec.modelname = "mug"
  parts = t.objects.mug_parts(d)
  for k in ("body", "handle"):
    mdl._add_mesh(spec, f"mug_{k}_visual", parts[k].vertices, parts[k].faces)
  liner = mdl._mug_liner(d)
  mdl._add_mesh(spec, "mug_liner_visual", liner.vertices, liner.faces)
  com, inertia = mdl._mass_props(t.objects.mug_mesh(d), d["mass"])
  b = spec.worldbody.add_body(name="mug", pos=list(pos))
  b.add_freejoint(name="mug_free")
  mdl._set_inertial(b, d["mass"], com, inertia)
  for g in mdl._mug_collision(b, d, phys.mug_walls):
    mdl._contact(g, phys, "mug", pert)
  for k, col in (("body", "mug"), ("handle", "mug"), ("liner", "steel")):
    b.add_geom(
      name=f"mug_{k}_vis",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"mug_{k}_visual",
      group=VISUAL_GROUP,
      contype=0,
      conaffinity=0,
      mass=0,
      rgba=mdl.COLORS[col],
    )
  return spec


def headlight(spec: mujoco.MjSpec) -> None:
  """Scene-level visual globals the twin sets on its own spec (lost in attach)."""
  spec.visual.headlight.diffuse = np.full(3, 0.25)
  spec.visual.headlight.ambient = np.full(3, 0.15)
  spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 640)
  spec.visual.global_.offheight = max(spec.visual.global_.offheight, 480)


def scene_cfg(
  num_envs: int = 1,
  substeps: int = TWIN_SUBSTEPS,
  n_caps: int = 1,
  cameras=("front", "grip"),
  sensors=(),
  ds: str | None = None,
  look: bool = True,
  joint_pos: dict[str, float] | None = None,
):
  """An mjlab SceneCfg of the twin: robot (XML actuators), `n_caps` caps, the mug.
  joint_pos: the robot's default joint pose (entity reset), e.g. the rest pose."""
  t = twin()
  joints = tuple(t.units.URDF_JOINTS)
  robot = EntityCfg(
    spec_fn=functools.partial(robot_spec, substeps, ds, tuple(cameras), look),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=joints),)
    ),
  )
  if joint_pos is not None:
    robot.init_state = EntityCfg.InitialStateCfg(joint_pos=dict(joint_pos))
  z = float(t.model.table_top(scene(ds), *PARK_XY))
  entities = {
    "robot": robot,
    "mug": EntityCfg(
      spec_fn=functools.partial(mug_spec, substeps, ds, (PARK_XY[0], PARK_XY[1], z)),
      init_state=EntityCfg.InitialStateCfg(pos=(PARK_XY[0], PARK_XY[1], z)),
    ),
  }
  for i in range(n_caps):
    entities[f"cap{i}"] = EntityCfg(
      spec_fn=functools.partial(
        cap_spec, substeps, ds, f"cap{i}", (-PARK_XY[0], PARK_XY[1] + 0.06 * i, z)
      ),
      init_state=EntityCfg.InitialStateCfg(pos=(-PARK_XY[0], PARK_XY[1] + 0.06 * i, z)),
    )
  return SceneCfg(
    num_envs=num_envs,
    env_spacing=1.0,
    terrain=None,
    entities=entities,
    sensors=tuple(sensors),
    spec_fn=headlight,
  )
