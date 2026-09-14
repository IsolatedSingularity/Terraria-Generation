"""Terraria 1.4.5.7 UnifiedRandom, independently implemented integer semantics.

Explicit Int32 wrapping matches the unchecked CLR arithmetic. State is the
actual uint inext and 56 Int32 words, not System.Random's older two-index model.
"""

from __future__ import annotations

MAX_INT = 2147483647
MIN_INT = -2147483648


def int32(value: int) -> int:
    return ((value + 2147483648) & 0xFFFFFFFF) - 2147483648


class UnifiedRandom:
    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int):
        if not MIN_INT <= seed <= MAX_INT:
            raise ValueError("Seed must be Int32")
        magnitude = MAX_INT if seed == MIN_INT else abs(seed)
        previous = int32(161803398 - magnitude)
        self.seed_array = [0] * 56
        self.seed_array[55] = previous
        current = 1
        for index in range(1, 55):
            slot = 21 * index % 55
            self.seed_array[slot] = current
            current = int32(previous - current)
            if current < 0:
                current = int32(current + MAX_INT)
            previous = self.seed_array[slot]
        for _ in range(4):
            for index in range(1, 56):
                value = int32(self.seed_array[index] - self.seed_array[1 + (index + 30) % 55])
                self.seed_array[index] = int32(value + MAX_INT) if value < 0 else value
        self.inext = 0

    def state(self):
        return {"inext": self.inext, "seed_array": self.seed_array.copy()}

    @classmethod
    def from_state(cls, state):
        index = state["inext"]
        words = state["seed_array"]
        if (
            not 0 <= index <= 55
            or len(words) != 56
            or any(not MIN_INT <= value <= MAX_INT for value in words)
        ):
            raise ValueError("Invalid UnifiedRandom state")
        rng = cls.__new__(cls)
        rng.inext = index
        rng.seed_array = list(words)
        return rng

    def _sample_int(self):
        index = self.inext + 1
        if index > 55:
            index = 1
        other = index + 21
        if other > 55:
            other -= 55
        value = int32(self.seed_array[index] - self.seed_array[other])
        if value == MAX_INT:
            value -= 1
        value = int32(value + ((value >> 31) & MAX_INT))
        self.seed_array[index] = value
        self.inext = index
        return value

    def peek(self):
        index = self.inext + 1
        if index > 55:
            index = 1
        other = index + 21
        if other > 55:
            other -= 55
        return int32(self.seed_array[index] - self.seed_array[other])

    def next_double(self):
        return float(self._sample_int()) * 4.656612875245797e-10

    def next(self, *args):
        if not args:
            return self._sample_int()
        if len(args) not in (1, 2) or any(not MIN_INT <= x <= MAX_INT for x in args):
            raise ValueError("Expected zero, one or two Int32 arguments")
        low, high = (0, args[0]) if len(args) == 1 else args
        if low > high:
            raise ValueError("Lower bound exceeds upper bound")
        span = high - low
        if span <= MAX_INT:
            return int(self.next_double() * span) + low
        value = self._sample_int()
        if self._sample_int() % 2 == 0:
            value = -value
        sample = (float(value) + 2147483646.0) / 4294967293.0
        return int(sample * span) + low
