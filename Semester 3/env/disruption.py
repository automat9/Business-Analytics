#!/usr/bin/env python
# coding: utf-8

import numpy as np
from typing import Any


class DisruptionProcess:
    """
    Disruption process with timing constraints to ensure adequate recovery periods.
    
    Parameters
    ----------
    cfg : argparse.Namespace or any object
        Must expose:
            leadRecItem  : int   (inbound lead time for retailer shipments)
            Ttest       : int   (episode length)
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        # Optional deterministic seed
        if hasattr(cfg, "seed_run"):
            np.random.seed(cfg.seed_run + 12345)
        self.lead = cfg.leadRecItem
        self.episode_length = getattr(cfg, 'Ttest', 100)
        self.reset_episode()

    def reset_episode(self) -> None:
        """
        Resample disruption start (Ts) and duration (D) with time constraints.
        Ensures:
        - Sufficient pre-disruption time for baseline behavior
        - Sufficient post-disruption time for recovery learning
        """
        # Duration: 1-3 periods (as per my methodology)
        self.D: int = np.random.randint(1, 4)
        
        # Calculate maximum allowed start time to ensure 20+ recovery periods
        min_recovery = 20
        max_start = self.episode_length - self.D - min_recovery
        
        # Also ensure minimum 10 periods before disruption
        min_start = 10
        
        if max_start < min_start:
            # Episode too short for proper learning, adjust parameters
            # Prioritize having some pre-disruption time
            self.Ts = min_start
            # Reduce duration if needed
            available_periods = self.episode_length - min_start - min_recovery
            self.D = max(1, min(self.D, available_periods))
        else:
            # Sample start time within valid range
            self.Ts: int = np.random.randint(min_start, max_start + 1)

    def in_window(self, t: int) -> bool:
        """
        Check whether period t (after lead correction) belongs
        to the disruption window.
        """
        return self.Ts <= t < self.Ts + self.D

    def apply(self, order_qty: int, t: int) -> int:
        """
        Return the actual shipped quantity given an incoming order_qty
        at simulation time t (retailer's clock).
        
        50% capacity reduction during disruption (as per Dolgui et al., 2019)
        """
        if self.in_window(t - self.lead):
            # Ship half the requested amount (round up)
            return int(np.ceil(order_qty / 2.0))
        return order_qty
    
    def get_info(self) -> dict:
        """Return disruption timing information for logging."""
        return {
            'start': self.Ts,
            'duration': self.D,
            'end': self.Ts + self.D - 1,
            'severity': 0.5  # 50% capacity reduction
        }