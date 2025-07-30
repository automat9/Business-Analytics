#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
env.warning
~~~~~~~~~~~~
Early‑Warning Signal generator for capacity disruptions.

For a given disruption schedule (Ts, D), emits a 1 with probability **pd**
during the window                         Ts‑2  ≤ t < Ts + D − 1,
and with probability **pf** otherwise.
"""

import numpy as np
from typing import Any


class EarlyWarningSignal:
    """
    Parameters
    ----------
    cfg : argparse.Namespace or any object
        Must expose:
            pd : float   # true‑positive probability in the window
            pf : float   # false‑positive probability outside the window
    Ts : int
        Disruption start period (sampled by DisruptionProcess)
    D : int
        Disruption duration
    """

    def __init__(self, cfg: Any, Ts: int, D: int):
        self.cfg = cfg
        self.Ts = Ts
        self.D = D

    def is_warning_window(self, t: int) -> bool:
        """Return True if t ∈ [Ts‑2, Ts + D − 2] (upper bound exclusive)."""
        return (self.Ts - 2) <= t < (self.Ts + self.D - 1)

    def get_flag(self, t: int) -> int:
        """
        Emit the warning signal at time t:
          - 1 with probability **pd** if in the window
          - 1 with probability **pf** otherwise
        """
        if self.is_warning_window(t):
            return int(np.random.rand() < self.cfg.pd)
        else:
            return int(np.random.rand() < self.cfg.pf)
