"""Shared convergence control for automatic learned-rounding tuning."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return value


def convergence_window(shape: Tuple[int, int], rank: Optional[int] = None) -> int:
    """Return a deterministic observation window for a matrix shape."""
    rows, cols = shape
    small = max(1, min(rows, cols))
    aspect = max(rows, cols) / small
    rank_scale = math.sqrt(max(1, rank or small) / 128.0)
    aspect_scale = math.sqrt(1.0 + math.log2(max(1.0, aspect)))
    return int(_clamp(round(16.0 * rank_scale * aspect_scale), 16, 128))


@dataclass
class AttemptSummary:
    """Serializable measurements for one probe or optimization attempt."""

    kind: str
    lr: float
    budget: int
    iterations: int = 0
    initial_loss: Optional[float] = None
    best_loss: Optional[float] = None
    final_loss: Optional[float] = None
    normalized_best_loss: Optional[float] = None
    stop_reason: str = "budget"
    retry_reason: Optional[str] = None
    lr_events: List[Dict[str, Any]] = field(default_factory=list)
    windows: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "lr": self.lr,
            "budget": self.budget,
            "iterations": self.iterations,
            "initial_loss": _finite_or_none(self.initial_loss),
            "best_loss": _finite_or_none(self.best_loss),
            "final_loss": _finite_or_none(self.final_loss),
            "normalized_best_loss": _finite_or_none(self.normalized_best_loss),
            "stop_reason": self.stop_reason,
            "retry_reason": self.retry_reason,
            "lr_events": self.lr_events,
            "windows": self.windows,
        }


class TuningReportCollector:
    """Collect and optionally persist automatic tuning records."""

    def __init__(self, output_path: Optional[str] = None, flush_on_add: bool = True):
        self.output_path = output_path
        self.flush_on_add = flush_on_add
        self.layers: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any]) -> None:
        self.layers.append(record)
        if self.output_path and self.flush_on_add:
            self.write(self.output_path)

    def as_dict(self) -> Dict[str, Any]:
        reasons: Dict[str, int] = {}
        retries = 0
        for layer in self.layers:
            reason = layer.get("stop_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
            retries += int(layer.get("retried", False))
        return {
            "version": 1,
            "mode": "auto",
            "profile": "balanced",
            "summary": {
                "layers": len(self.layers),
                "retries": retries,
                "stop_reasons": reasons,
            },
            "layers": self.layers,
        }

    def write(self, output_path: Optional[str] = None) -> None:
        path = Path(output_path or self.output_path or "")
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self.as_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise


class AdaptiveConvergenceController:
    """Observe normalized loss trends and choose LR and stopping actions."""

    def __init__(
        self,
        shape: Tuple[int, int],
        rank: Optional[int],
        optimizer: str,
        initial_lr: float,
        budget: int,
        kind: str,
    ):
        self.shape = shape
        self.rank = rank
        self.optimizer = optimizer
        self.window = convergence_window(shape, rank)
        self.warmup = min(budget, max(50 if optimizer == "prodigy" else 0, 2 * self.window))
        self.summary = AttemptSummary(kind=kind, lr=initial_lr, budget=budget)
        self.losses: List[float] = []
        self.best_history: List[float] = []
        self.plateau_windows = 0
        self.stable_windows = 0
        self.lr_reductions = 0
        self.should_stop = False
        self.retry_recommended = False
        self._pending_lr_factor: Optional[float] = None

    @property
    def best_loss(self) -> float:
        return self.summary.best_loss if self.summary.best_loss is not None else math.inf

    def observe(self, current_loss: float, previous_best: float) -> bool:
        """Record a step and return whether it is a meaningful new best."""
        first_observation = self.summary.initial_loss is None
        self.summary.iterations += 1
        self.summary.final_loss = current_loss

        if not math.isfinite(current_loss):
            self.should_stop = True
            self.retry_recommended = True
            self.summary.stop_reason = "nonfinite"
            self.summary.retry_reason = "nonfinite_loss"
            return False

        if self.summary.initial_loss is None:
            self.summary.initial_loss = current_loss
            self.summary.best_loss = min(current_loss, previous_best)

        scale = max(abs(self.summary.initial_loss), 1e-30)
        prior = self.summary.best_loss if self.summary.best_loss is not None else previous_best
        relative_delta = (prior - current_loss) / scale

        recent_deltas = []
        if len(self.losses) >= 2:
            recent = self.losses[-self.window :]
            recent_deltas = [(b - a) / scale for a, b in zip(recent, recent[1:])]
        center = median(recent_deltas) if recent_deltas else 0.0
        noise = median([abs(value - center) for value in recent_deltas]) if recent_deltas else 0.0
        meaningful_floor = max(1e-6, 3.0 * noise)
        improved = first_observation or (current_loss < prior and relative_delta > meaningful_floor)

        self.losses.append(current_loss)
        if current_loss < prior:
            self.summary.best_loss = current_loss
        self.best_history.append(self.best_loss)

        if self.summary.initial_loss is not None:
            self.summary.normalized_best_loss = self.best_loss / scale
            if self.summary.normalized_best_loss <= 1e-6 and self.summary.iterations >= self.warmup:
                self.should_stop = True
                self.summary.stop_reason = "negligible_loss"

        if self.summary.iterations % self.window == 0:
            self._evaluate_window(scale)

        return improved

    def _evaluate_window(self, scale: float) -> None:
        if len(self.best_history) < self.window + 1:
            return
        start_best = self.best_history[-self.window - 1]
        end_best = self.best_history[-1]
        gain = max(0.0, (start_best - end_best) / scale)
        recent = self.losses[-self.window :]
        deltas = [(b - a) / scale for a, b in zip(recent, recent[1:])]
        center = median(deltas) if deltas else 0.0
        mad = median([abs(value - center) for value in deltas]) if deltas else 0.0
        noise_floor = max(1e-6, 3.0 * mad)
        quiet = mad <= max(gain, 1e-12)
        plateau = gain <= noise_floor

        self.summary.windows.append({
            "iteration": self.summary.iterations,
            "relative_gain": gain,
            "noise_mad": mad,
            "plateau": plateau,
        })

        if plateau:
            self.plateau_windows += 1
            self.stable_windows = self.stable_windows + 1 if quiet else 0
            self._pending_lr_factor = 0.8 if quiet else 0.5
        else:
            self.plateau_windows = 0
            self.stable_windows = 0

        initial = self.summary.initial_loss or 0.0
        if recent and initial and max(recent) > abs(initial) * 4.0:
            self.should_stop = True
            self.retry_recommended = True
            self.summary.stop_reason = "unstable"
            self.summary.retry_reason = "sustained_regression"
        elif (
            self.summary.iterations >= self.warmup
            and self.plateau_windows >= 3
            and self.stable_windows >= 2
            and self.lr_reductions >= 1
        ):
            self.should_stop = True
            self.summary.stop_reason = "converged_plateau"

    def update_lr(self, current_lr: float) -> Tuple[float, bool]:
        if self._pending_lr_factor is None:
            return current_lr, False
        factor = self._pending_lr_factor
        self._pending_lr_factor = None
        new_lr = max(current_lr * factor, max(self.summary.lr * 1e-6, 1e-12))
        if new_lr == current_lr:
            return current_lr, False
        self.lr_reductions += 1
        self.summary.lr_events.append({
            "iteration": self.summary.iterations,
            "old_lr": current_lr,
            "new_lr": new_lr,
            "factor": factor,
        })
        return new_lr, True

    def score(self) -> float:
        if self.summary.initial_loss is None or self.summary.best_loss is None:
            return -math.inf
        if not math.isfinite(self.summary.best_loss):
            return -math.inf
        scale = max(abs(self.summary.initial_loss), 1e-30)
        gain = max(0.0, (self.summary.initial_loss - self.summary.best_loss) / scale)
        instability_penalty = 1.0 if self.retry_recommended else 0.0
        return gain / max(1, self.summary.iterations) - instability_penalty
