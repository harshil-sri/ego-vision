"""Hand-state scorer: are the hands busy with fine work right now?

EXPERIMENTAL. This is the link between the research pilot and the app.

Why: in the HoloAssist recordings, human instructors start routine speech
during fine-motor actions (screw, unscrew, insert) at about a third of the
chance rate (reports/research_gap_map.md, part 5). An assistant that wants
to speak at good moments can use the same signal.

How: the verb model already turns the last 2 s of video into one VideoMAE
vector per tick. A second small head (scripts/train_hand_state_head.py,
trained on HoloAssist) reads that vector. No extra video pass.

    feature (768,) -> head -> 22 state probabilities -> 3 group probabilities
        fine   screw, unscrew, insert
        free   idle, hold, place, inspect
        other  every other action

Limits: the head was trained on assembly tasks filmed with a HoloLens. On
kitchen video it has only an indirect check (higher "fine" for cut, wash,
mix than for take, put). Nothing in the app acts on this signal yet; it is
reported in the engine state under "hands".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class HandStateScorer:
    def __init__(self, run_dir, device: str = "cpu", alpha: float = 0.5) -> None:
        """run_dir: folder with last.pt from scripts.train_hand_state_head.
        alpha: weight of the newest tick in the running average (1 = no smoothing)."""
        import torch
        from src.models.frozen import FeatureHead

        state = torch.load(Path(run_dir) / "last.pt", map_location=device)
        self.head = FeatureHead(**state["head"])
        self.head.load_state_dict(state["model_state_dict"])
        self.head.to(device).eval()
        self.groups: list[str] = list(state["groups"])                  # ["fine", "free", "other"]
        self._gid = np.asarray(state["group_of_state"])                 # (num_states,) group id of each state
        self.device, self.alpha = device, float(alpha)
        self._avg: np.ndarray | None = None

    def reset(self) -> None:
        self._avg = None

    def update(self, feature: np.ndarray) -> dict:
        """feature: (768,) VideoMAE vector of the current window.
        -> {"fine": p, "free": p, "other": p, "state": name of the largest}; the
        probabilities are a running average over ticks and sum to 1."""
        import torch
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(feature, np.float32))[None].to(self.device)
            p = torch.softmax(self.head(x).float(), dim=1)[0].cpu().numpy()   # (num_states,)
        g = np.array([p[self._gid == k].sum() for k in range(len(self.groups))])
        self._avg = g if self._avg is None else self.alpha * g + (1 - self.alpha) * self._avg
        out = {name: round(float(v), 3) for name, v in zip(self.groups, self._avg)}
        out["state"] = self.groups[int(self._avg.argmax())]
        return out
