from dataclasses import dataclass


@dataclass(frozen=True)
class IntConstants:
    bits_per_digit: int
    small_neg: int
    small_pos: int
    source: str
    true_bits: int | None = None
    false_bits: int | None = None

    @property
    def compact_bound(self) -> int:
        return 1 << self.bits_per_digit

    @property
    def small_min(self) -> int:
        return -self.small_neg

    @property
    def small_max(self) -> int:
        return self.small_pos - 1

    def bool_bit(self, bit: int, of_true: bool) -> bool:
        bits = self.true_bits if of_true else self.false_bits
        if bits is None:
            raise ValueError("the addresses of True and False are not recorded for this build")
        return (bits >> bit) & 1 == 1


DEFAULTS = IntConstants(30, 5, 1025, "defaults")
