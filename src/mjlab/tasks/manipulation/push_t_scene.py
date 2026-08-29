"""Scene definition for the Push-T task.

A yellow T-shaped block rests on a white table. The goal is a green
footprint of the same T outline that the block must be pushed onto.

The T outline comes from ``assets/t_object.stl`` (0.1 x 0.1 x 0.02 m):
a crossbar spanning the full width, and a stem running in -y.

The mesh is used for *visuals only*. MuJoCo convexifies meshes for
collision, which would fill in the T's concave armpits, so contact is
handled by two boxes that reproduce the outline exactly.
"""

from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots import YAM_ACTION_SCALE, get_yam_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils import spec_config as spec_cfg

T_MESH_PATH = Path(__file__).parent / "assets" / "t_object.stl"

# T outline in the STL's own frame, decomposed into two boxes.
# Crossbar: x in [-0.05, 0.05], y in [-0.0125, 0.0125]
# Stem:     x in [-0.0125, 0.0125], y in [-0.0875, -0.0125]
# Both span z in [0, 0.02].
T_THICKNESS = 0.02
_CROSSBAR_HALF = (0.05, 0.0125, T_THICKNESS / 2)
_CROSSBAR_POS = (0.0, 0.0, T_THICKNESS / 2)
_STEM_HALF = (0.0125, 0.0375, T_THICKNESS / 2)
_STEM_POS = (0.0, -0.05, T_THICKNESS / 2)

# The STL's origin sits at the crossbar, not at the T's centroid. Shifting
# every geom by +CENTROID_OFFSET_Y puts the body frame on the area centroid,
# so body position/orientation *is* the T's pose. Derived from the two-box
# decomposition: (A_cross * 0 + A_stem * -0.05) / (A_cross + A_stem).
CENTROID_OFFSET_Y = 0.021428571428571429

YELLOW = (1.0, 0.85, 0.1, 1.0)
GREEN = (0.15, 0.75, 0.3, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)

# Visual mesh is group 2, collision boxes group 3. This matches the robot's own
# convention and means the viewer shows the mesh (groups 0-2 are on by default)
# while the depth camera, which renders groups (0, 3), sees the boxes.
_VISUAL_GROUP = 2
_COLLISION_GROUP = 3
# Group 3 is also what the wrist camera renders (enabled_geom_groups=(0, 3)),
# so anything the policy must see through the camera needs a geom in it.
_CAMERA_GROUP = 3

_FOOTPRINT_HALF_THICKNESS = 0.0005

# Goal sits directly in front of the base (+x is forward). The block starts
# closer in, so the arm pushes it away from itself toward the footprint.
GOAL_POSE = (0.40, 0.0, 0.0)
T_SPAWN_CENTER = (0.28, 0.0, 0.0)


def _shift_y(pos: tuple[float, float, float]) -> tuple[float, float, float]:
  return (pos[0], pos[1] + CENTROID_OFFSET_Y, pos[2])


def get_t_object_spec(
  rgba: tuple[float, float, float, float] = YELLOW,
  crossbar_mass: float = 0.060,
  stem_mass: float = 0.045,
  friction: tuple[float, float, float] = (0.3, 0.005, 0.0001),
) -> mujoco.MjSpec:
  """The pushable T block: mesh for looks, two boxes for contact."""
  spec = mujoco.MjSpec()
  spec.add_mesh(name="t_mesh", file=str(T_MESH_PATH))

  body = spec.worldbody.add_body(name="t_object")
  body.add_freejoint(name="t_joint")

  body.add_geom(
    name="t_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="t_mesh",
    pos=_shift_y((0.0, 0.0, 0.0)),
    rgba=rgba,
    group=_VISUAL_GROUP,
    contype=0,
    conaffinity=0,
    mass=0.0,
  )
  for name, half, pos, mass in (
    ("t_crossbar_collision", _CROSSBAR_HALF, _CROSSBAR_POS, crossbar_mass),
    ("t_stem_collision", _STEM_HALF, _STEM_POS, stem_mass),
  ):
    body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=half,
      pos=_shift_y(pos),
      mass=mass,
      friction=friction,
      group=_COLLISION_GROUP,
      rgba=rgba,
    )

  # Marks the centroid; handy for reward terms and debugging.
  body.add_site(name="t_center", pos=(0.0, 0.0, T_THICKNESS / 2), size=(0.005,) * 3)
  return spec


def get_t_footprint_spec(
  rgba: tuple[float, float, float, float] = GREEN,
) -> mujoco.MjSpec:
  """Flat green target outline. No joint, so mjlab wraps it as a mocap body.

  Two thin boxes rather than a flattened mesh: the outline is then exact,
  and it dodges MuJoCo's mesh recentering when a scale is applied.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="t_goal")
  for name, half, pos in (
    ("goal_crossbar", _CROSSBAR_HALF, _CROSSBAR_POS),
    ("goal_stem", _STEM_HALF, _STEM_POS),
  ):
    size = (half[0], half[1], _FOOTPRINT_HALF_THICKNESS)
    geom_pos = _shift_y((pos[0], pos[1], _FOOTPRINT_HALF_THICKNESS))
    # Two coincident copies, mirroring how the block carries both a group-2
    # mesh and group-3 boxes. The viewer shows groups 0-2 and the camera
    # renders (0, 3), so one copy each keeps the footprint visible to both.
    for suffix, group in ((None, _VISUAL_GROUP), ("cam", _CAMERA_GROUP)):
      body.add_geom(
        name=name if suffix is None else f"{name}_{suffix}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=geom_pos,
        rgba=rgba,
        group=group,
        contype=0,
        conaffinity=0,
        mass=0.0,
      )
  return spec


def get_white_table_cfg() -> TerrainEntityCfg:
  """Ground plane with the default checker swapped for flat white."""
  return TerrainEntityCfg(
    terrain_type="plane",
    textures=(),
    materials=(
      spec_cfg.MaterialCfg(
        name="table",
        rgba=WHITE,
        reflectance=0.0,
        geom_names_expr=("terrain$",),
      ),
    ),
  )


def make_push_t_scene_cfg(num_envs: int = 1, env_spacing: float = 1.0) -> SceneCfg:
  """YAM arm, yellow T block, and green goal footprint on a white table."""
  return SceneCfg(
    terrain=get_white_table_cfg(),
    num_envs=num_envs,
    env_spacing=env_spacing,
    entities={
      "robot": get_yam_robot_cfg(),
      "t_object": EntityCfg(
        spec_fn=get_t_object_spec,
        init_state=EntityCfg.InitialStateCfg(pos=T_SPAWN_CENTER),
      ),
      "t_goal": EntityCfg(
        spec_fn=get_t_footprint_spec,
        init_state=EntityCfg.InitialStateCfg(pos=GOAL_POSE),
      ),
    },
  )


__all__ = [
  "YAM_ACTION_SCALE",
  "CENTROID_OFFSET_Y",
  "T_MESH_PATH",
  "GOAL_POSE",
  "T_SPAWN_CENTER",
  "get_t_object_spec",
  "get_t_footprint_spec",
  "get_white_table_cfg",
  "make_push_t_scene_cfg",
]
