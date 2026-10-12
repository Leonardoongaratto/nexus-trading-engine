import math
import random
from typing import Protocol

from xau_bot.domain.models import Direction


class SlippageModel(Protocol):
    def calculate_slippage(
        self,
        price: float,
        direction: Direction,
        volume: float,
        spread: float,
        is_stop_or_market: bool = True,
    ) -> float: ...


class ZeroSlippage:
    def calculate_slippage(
        self,
        price: float,
        direction: Direction,
        volume: float,
        spread: float,
        is_stop_or_market: bool = True,
    ) -> float:
        return 0.0


class FixedSlippage:
    def __init__(self, points: float = 1.0, point_size: float = 0.01) -> None:
        if not math.isfinite(points) or points < 0:
            raise ValueError(f"points must be non-negative and finite, got {points}")
        if not math.isfinite(point_size) or point_size <= 0:
            raise ValueError(f"point_size must be positive and finite, got {point_size}")
        self.points = points
        self.point_size = point_size

    def calculate_slippage(
        self,
        price: float,
        direction: Direction,
        volume: float,
        spread: float,
        is_stop_or_market: bool = True,
    ) -> float:
        if not is_stop_or_market:
            return 0.0
        return self.points * self.point_size


class NormalSlippage:
    def __init__(
        self,
        mean_points: float = 1.0,
        std_points: float = 0.5,
        point_size: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if not math.isfinite(mean_points) or mean_points < 0:
            raise ValueError(f"mean_points must be non-negative and finite, got {mean_points}")
        if not math.isfinite(std_points) or std_points < 0:
            raise ValueError(f"std_points must be non-negative and finite, got {std_points}")
        if not math.isfinite(point_size) or point_size <= 0:
            raise ValueError(f"point_size must be positive and finite, got {point_size}")

        self.mean_points = mean_points
        self.std_points = std_points
        self.point_size = point_size
        self._rng = random.Random(seed)

    def calculate_slippage(
        self,
        price: float,
        direction: Direction,
        volume: float,
        spread: float,
        is_stop_or_market: bool = True,
    ) -> float:
        if not is_stop_or_market:
            return 0.0
        sample = self._rng.gauss(self.mean_points, self.std_points)
        slip_points = max(0.0, sample)
        return round(slip_points * self.point_size, 6)
