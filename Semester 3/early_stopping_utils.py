#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import deque

class PlateauDetector:
    """
    Detects when model performance has plateaued based on cost variance.
    """
    def __init__(self, window_size=20, variance_threshold=0.05, min_episodes=1000):
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.min_episodes = min_episodes
        self.cost_history = deque(maxlen=window_size)
        self.episodes_seen = 0
        
    def update(self, test_cost):
        """Add new test cost to history."""
        self.cost_history.append(test_cost)
        self.episodes_seen += 1
        
    def has_plateaued(self):
        """
        Check if performance has plateaued.
        Returns True if:
        1. We've seen enough episodes
        2. The window is full
        3. The coefficient of variation is below threshold
        """
        if self.episodes_seen < self.min_episodes:
            return False
            
        if len(self.cost_history) < self.window_size:
            return False
            
        costs = np.array(self.cost_history)
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        # Avoid division by zero
        if mean_cost < 1.0:
            return True  # Already at very low cost
            
        # Coefficient of variation
        cv = std_cost / mean_cost
        
        return cv < self.variance_threshold
    
    def get_stats(self):
        """Get current statistics."""
        if len(self.cost_history) == 0:
            return None
            
        costs = np.array(self.cost_history)
        return {
            'mean': np.mean(costs),
            'std': np.std(costs),
            'cv': np.std(costs) / np.mean(costs) if np.mean(costs) > 0 else 0,
            'min': np.min(costs),
            'max': np.max(costs)
        }