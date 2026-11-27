"""Boundary handling: what to do with a particle that leaves the search space.

This is an extension to Algorithm 39, which lets particles go anywhere. A
handler is applied straight after the position update (line 26). Each handler
receives the new position and velocity and returns corrected copies.

All handlers share one pair of limits [lower, upper] for every dimension.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BoundaryHandler(ABC):
    """Interface for a boundary handling technique."""

    name: str = "boundary"

    def __init__(self, lower: float = -1.0, upper: float = 1.0):
        if not upper > lower:
            raise ValueError("upper bound must be greater than lower bound.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.rng: Optional[np.random.Generator] = None

    def bind(self, rng: np.random.Generator) -> None:
        """Share the swarm's random generator (used by RandomBoundary)."""
        self.rng = rng

    @abstractmethod
    def apply(self, position: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, velocity) after enforcing the bounds."""
        raise NotImplementedError


class NoBoundary(BoundaryHandler):
    """Plain Algorithm 39: particles may fly anywhere."""

    name = "none"

    def apply(self, position, velocity):
        return position, velocity


class ClampBoundary(BoundaryHandler):
    """Absorbing walls: a particle that crosses a wall is put on the wall and
    its velocity in that dimension is set to zero."""

    name = "clamp"

    def apply(self, position, velocity):
        outside = (position < self.lower) | (position > self.upper)
        if not outside.any():
            return position, velocity
        position = np.clip(position, self.lower, self.upper)
        velocity = np.where(outside, 0.0, velocity)
        return position, velocity


class ReflectBoundary(BoundaryHandler):
    """Reflecting walls: the particle bounces back by the distance it
    overshot and its velocity in that dimension changes sign."""

    name = "reflect"

    def apply(self, position, velocity):
        outside = (position < self.lower) | (position > self.upper)
        if not outside.any():
            return position, velocity
        width = self.upper - self.lower
        # Fold the coordinate back with period 2*width, which gives the right
        # answer even when the overshoot is bigger than the whole range.
        t = np.mod(position - self.lower, 2.0 * width)
        folded = self.lower + np.where(t > width, 2.0 * width - t, t)
        position = np.where(outside, folded, position)
        velocity = np.where(outside, -velocity, velocity)
        return position, velocity


class WrapBoundary(BoundaryHandler):
    """Periodic space: a particle leaving through one wall re-enters through
    the opposite one with its velocity unchanged."""

    name = "wrap"

    def apply(self, position, velocity):
        outside = (position < self.lower) | (position > self.upper)
        if not outside.any():
            return position, velocity
        width = self.upper - self.lower
        wrapped = self.lower + np.mod(position - self.lower, width)
        position = np.where(outside, wrapped, position)
        return position, velocity


class RandomBoundary(BoundaryHandler):
    """Random re-initialisation: each coordinate that left the space is
    replaced by a fresh uniform random value and its velocity is zeroed."""

    name = "random"

    def apply(self, position, velocity):
        outside = (position < self.lower) | (position > self.upper)
        if not outside.any():
            return position, velocity
        if self.rng is None:
            self.rng = np.random.default_rng()
        fresh = self.rng.uniform(self.lower, self.upper, size=position.shape)
        position = np.where(outside, fresh, position)
        velocity = np.where(outside, 0.0, velocity)
        return position, velocity


BOUNDARY_HANDLERS = {
    "none": NoBoundary,
    "clamp": ClampBoundary,
    "reflect": ReflectBoundary,
    "wrap": WrapBoundary,
    "random": RandomBoundary,
}


def make_boundary_handler(name: str, lower: float = -1.0, upper: float = 1.0) -> BoundaryHandler:
    """Create a boundary handler from its configuration name."""
    try:
        return BOUNDARY_HANDLERS[name](lower, upper)
    except KeyError:
        raise ValueError(
            f"Unknown boundary handler '{name}'. Available: {sorted(BOUNDARY_HANDLERS)}."
        ) from None
