"""Train the two-camera cap-to-mug student by behaviour cloning on collect.py's dataset.

  env -u PYTHONPATH uv run --frozen --no-sync python scripts/so101_twin/student.py \
      --data ../../sim/outputs/mjlab/cap_to_mug_teacher_2800 --out student_bc.pt

THE STUDENT sees only what the real robot has: the front and grip RGB frames (collect.CAMS
sizes) and the 6 joint positions (rad, URDF order). One CNN per camera, then a linear
layer; a linear layer on the joints; the three concatenated into an MLP that outputs the
teacher's action (the same integrated joint-target action, so it drives the same env).
Inputs and outputs are normalized inside the model with the dataset's statistics: it
takes raw uint8 frames and radians, and returns raw actions.

THE LABEL is teacher_action, the teacher's clean action at every recorded state, also
where the executed action was noisy (DART). 5% of episodes are held out for validation.
Images get a random shift of up to SHIFT px (DrQ), on top of the dataset's visual DR.

Closed-loop score on the benchmark stages:
  eval_teacher.py --student student_bc.pt --stages <a sim-eval trials.jsonl> --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from collect import CAMS
from torch import nn

SHIFT = 4  # px


def _encoder(w: int, h: int, out: int = 256) -> nn.Sequential:
  convs: list[nn.Module] = []
  for c_in, c_out in ((3, 32), (32, 64), (64, 64), (64, 64)):
    convs += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.ELU()]
    w, h = (w + 1) // 2, (h + 1) // 2
  return nn.Sequential(
    *convs, nn.Flatten(), nn.Linear(64 * w * h, out), nn.LayerNorm(out), nn.ELU()
  )


class Student(nn.Module):
  def __init__(self, n_joints: int = 6, n_act: int = 6) -> None:
    super().__init__()
    self.cnn = nn.ModuleDict({c: _encoder(w, h) for c, (w, h) in CAMS.items()})
    self.joints = nn.Sequential(nn.Linear(n_joints, 64), nn.ELU())
    self.head = nn.Sequential(
      nn.Linear(256 * len(CAMS) + 64, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, n_act),
    )
    for k, n in (("q", n_joints), ("a", n_act)):
      self.register_buffer(f"{k}_mean", torch.zeros(n))
      self.register_buffer(f"{k}_std", torch.ones(n))

  def forward(
    self, front: torch.Tensor, grip: torch.Tensor, q: torch.Tensor
  ) -> torch.Tensor:
    """uint8 (B, H, W, 3) frames and joint positions (B, 6) rad -> actions (B, 6)."""
    z = [
      self.cnn[c](x.permute(0, 3, 1, 2).float() / 255 - 0.5)
      for c, x in (("front", front), ("grip", grip))
    ]
    z.append(self.joints((q - self.q_mean) / self.q_std))
    return self.head(torch.cat(z, -1)) * self.a_std + self.a_mean

  @torch.no_grad()
  def act(self, env) -> torch.Tensor:
    """The action for every env of a collect.env_cfg env (cameras on)."""
    return self(
      env.scene["front_cam"].data.rgb,
      env.scene["grip_cam"].data.rgb,
      env.scene["robot"].data.joint_pos[:, :6],
    )

  @classmethod
  def load(cls, path: Path, device: str) -> Student:
    s = cls().to(device)
    s.load_state_dict(torch.load(path, map_location=device)["model"])
    return s.eval()


def shift(x: torch.Tensor, pad: int = SHIFT) -> torch.Tensor:
  """Each (B, H, W, C) frame moved by its own random offset, edges repeated."""
  b, h, w, _ = x.shape
  d = torch.randint(-pad, pad + 1, (2, b, 1), device=x.device)
  r = (torch.arange(h, device=x.device) + d[0]).clamp(0, h - 1)
  c = (torch.arange(w, device=x.device) + d[1]).clamp(0, w - 1)
  return x[torch.arange(b, device=x.device)[:, None, None], r[:, :, None], c[:, None]]


KEYS = ("front", "grip", "joint_pos", "teacher_action")


def load(data: Path, device: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
  """Every shard's KEYS on the device, and each frame's episode number.

  Filled shard by shard into tensors allocated once: the frames are ~5.4 GB, and a
  torch.cat of per-shard copies would need twice that on the 12 GB card."""
  shards = [np.load(f) for f in sorted(data.glob("shard_*.npz"))]
  assert shards, f"no shard_*.npz in {data}"
  lens = np.concatenate([z["ep_length"] for z in shards])
  ep = torch.from_numpy(np.repeat(np.arange(len(lens)), lens)).to(device)
  d: dict[str, torch.Tensor] = {}
  at = 0
  for z in shards:
    for k in KEYS:
      x = torch.from_numpy(z[k])
      if k not in d:
        d[k] = torch.empty((len(ep), *x.shape[1:]), dtype=x.dtype, device=device)
      d[k][at : at + len(x)] = x.to(device)
    at += len(x)
  assert at == len(ep)
  return d, ep


def main() -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--data", required=True, type=Path, help="collect.py's output dir")
  p.add_argument("--out", required=True, type=Path, help="the student .pt")
  p.add_argument("--epochs", type=int, default=30)
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--val", type=float, default=0.05, help="share of episodes held out")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  torch.manual_seed(a.seed)
  dev = a.device

  d, ep = load(a.data, dev)
  n_ep = int(ep.max()) + 1
  val_eps = torch.randperm(n_ep, device=dev)[: max(1, round(a.val * n_ep))]
  is_val = torch.isin(ep, val_eps)
  tr, va = (~is_val).nonzero().flatten(), is_val.nonzero().flatten()
  print(f"{len(tr)} train / {len(va)} val frames ({n_ep} episodes)")

  model = Student().to(dev)
  q, act = d["joint_pos"][tr].float(), d["teacher_action"][tr].float()
  model.q_mean.copy_(q.mean(0)), model.q_std.copy_(q.std(0).clamp_min(1e-3))
  model.a_mean.copy_(act.mean(0)), model.a_std.copy_(act.std(0).clamp_min(1e-3))
  opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
  steps = a.epochs * math.ceil(len(tr) / a.batch)
  sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)

  def loss_on(i: torch.Tensor, aug: bool) -> torch.Tensor:
    f, g = d["front"][i], d["grip"][i]
    if aug:
      f, g = shift(f), shift(g)
    pred = model(f, g, d["joint_pos"][i].float())
    # In normalized units, so the jaw counts as much as the arm.
    return F.mse_loss(
      (pred - model.a_mean) / model.a_std,
      (d["teacher_action"][i].float() - model.a_mean) / model.a_std,
    )

  log = []
  t0 = time.time()
  for e in range(a.epochs):
    model.train()
    tot = 0.0
    for i in tr[torch.randperm(len(tr), device=dev)].split(a.batch):
      loss = loss_on(i, aug=True)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      sched.step()
      tot += loss.item() * len(i)
    model.eval()
    with torch.no_grad():
      val = sum(loss_on(i, aug=False).item() * len(i) for i in va.split(1024))
    log.append({"epoch": e + 1, "train": tot / len(tr), "val": val / len(va)})
    print(
      f"epoch {e + 1}/{a.epochs}  train {log[-1]['train']:.4f}  "
      f"val {log[-1]['val']:.4f}  ({time.time() - t0:.0f} s)",
      flush=True,
    )

  a.out.parent.mkdir(parents=True, exist_ok=True)
  info = {"data": str(a.data), "args": {k: str(v) for k, v in vars(a).items()}}
  torch.save({"model": model.state_dict(), "info": info, "log": log}, a.out)
  a.out.with_suffix(".json").write_text(json.dumps({**info, "log": log}, indent=2))
  print(f"wrote {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
