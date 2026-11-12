import json
import re
from pathlib import Path

from ..domain.build import DEFAULTS, IntConstants

BUILD_JSON = Path("Tools/uop_validator/build-constants.json")


def from_checkout(root: Path, build_json: Path | None = None) -> IntConstants:
    header = root / "Include/internal/pycore_runtime_structs.h"
    text = header.read_text()
    neg = re.search(r"#define _PY_NSMALLNEGINTS\s+(\d+)", text)
    pos = re.search(r"#define _PY_NSMALLPOSINTS\s+(\d+)", text)
    if neg is None or pos is None:
        raise ValueError(f"small-int constants not found in {header}")
    bits = DEFAULTS.bits_per_digit
    true_bits: int | None = None
    false_bits: int | None = None
    source = f"{header.name} for the small-int range; bits per digit defaulted to {bits}"
    if build_json is not None and build_json.exists():
        data = json.loads(build_json.read_text())
        bits = int(data["bits_per_digit"])
        source = f"{header.name} for the small-int range; {build_json.name} for bits per digit"
        if "true_id_low" in data and "false_id_low" in data:
            true_bits = int(data["true_id_low"])
            false_bits = int(data["false_id_low"])
            source += " and the address bits of True and False"
    return IntConstants(bits, int(neg.group(1)), int(pos.group(1)), source, true_bits, false_bits)


def default(root: Path) -> IntConstants:
    return from_checkout(root, root / BUILD_JSON)
