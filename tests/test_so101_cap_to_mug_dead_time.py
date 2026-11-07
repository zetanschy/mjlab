"""The cap-to-mug action's servo dead time picks the goal the live twin's OnlineGoals
picks, at every substep (agent_101's sim/real2sim must be present)."""

import importlib

import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.so_arm101 import twin
from mjlab.tasks.manipulation.config.so101 import cap_to_mug_mdp as c2m

DEC = twin.RL_SUBSTEPS


@pytest.mark.parametrize("dead", [0.0, 0.5, 0.999, 1.0, 1.7, 2.0, 3.0])
def test_dead_time_matches_online_goals(dead: float) -> None:
  twin.twin()  # puts agent_101/sim on sys.path
  goals = importlib.import_module("real2sim.live.goals")
  og = goals.OnlineGoals(dead / 30.0, q0=np.full(6, -1.0))
  sent = [-1.0] * c2m.SENT_FRAMES  # reset: the rest pose, oldest first
  for k in range(10):
    og.push(k / 30.0, np.full(6, float(k)))
    sent = sent[1:] + [float(k)]
    for s in range(DEC):
      j = int(c2m.frames_back(torch.tensor([dead]), s, DEC)[0])
      want = og.at((k + s / DEC) / 30.0)[0]
      assert sent[c2m.SENT_FRAMES - 1 - j] == want, (k, s, j, want)
