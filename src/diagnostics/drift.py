from __future__ import annotations

import math
from typing import Dict, Optional


class RollingStats:
    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self.ema: Optional[float] = None
        self.ema_var: float = 0.0
        self.n_updates: int = 0

    def update(self, value: float) -> tuple[float, float]:
        if not math.isfinite(value):
            return (self.ema or 0.0), math.sqrt(self.ema_var)
        self.n_updates += 1
        if self.ema is None:
            self.ema = value
            self.ema_var = 0.0
        else:
            delta = value - self.ema
            self.ema = self.decay * self.ema + (1 - self.decay) * value
            self.ema_var = self.decay * self.ema_var + (1 - self.decay) * delta * delta
        return self.ema, math.sqrt(self.ema_var)


class DriftDetector:
    def __init__(self, bucket_names: list[str], decay: float = 0.98, k: float = 6.0, min_samples: int = 20):
        self.trackers: Dict[str, RollingStats] = {name: RollingStats(decay) for name in bucket_names}
        self.k = k
        self.min_samples = min_samples

    def update_and_check(self, values: Dict[str, float], explode_threshold: float) -> tuple[bool, Dict[str, str]]:
        triggered = False
        reasons: Dict[str, str] = {}
        for name, value in values.items():
            tracker = self.trackers.setdefault(name, RollingStats())
            ema, std = tracker.update(value)
            if not math.isfinite(value) or value > explode_threshold:
                triggered = True
                reasons[name] = f"value={value:.4g} exceeds absolute threshold={explode_threshold:.4g}"
            elif tracker.n_updates >= self.min_samples and std > 0 and value > ema + self.k * std:
                triggered = True
                reasons[name] = f"value={value:.4g} exceeds drift bound ema={ema:.4g}+{self.k}*std={std:.4g}"
        return triggered, reasons
