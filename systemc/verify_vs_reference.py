#!/usr/bin/env python3
"""Differential test: SystemC model vs the verified Python reference model.

The SystemC testbench (`mxfp8_stfdpa_tb --dump N`) emits one line per random
32-element MXFP8 block:

    FMT=e4m3 A=<64 hex> B=<64 hex> SA=<2 hex> SB=<2 hex> C=<8 hex> D=<8 hex>

This script re-computes D with the Python ST-FDPA reference
(`../mxfp8_stfdpa.py`, the bit-accurate model of the Blackwell/B300 tensor-core
fused dot product) and requires bit-identical results.

    python3 verify_vs_reference.py mxfp8_dump.txt [more dumps...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mxfp8_stfdpa import E4M3, E5M2, st_fdpa  # noqa: E402

LINE = re.compile(
    r"FMT=(?P<fmt>e4m3|e5m2) A=(?P<a>[0-9A-F]{64}) B=(?P<b>[0-9A-F]{64}) "
    r"SA=(?P<sa>[0-9A-F]{2}) SB=(?P<sb>[0-9A-F]{2}) C=(?P<c>[0-9A-F]{8}) D=(?P<d>[0-9A-F]{8})"
)

F_DEFAULT = 25  # Blackwell/B300 MXFP8: fractional bits of the fused summation
L = 32          # one MX block / tcgen05 k = 32


def verify(path: str) -> tuple[int, int, int]:
    """Diff one dump file. The dump carries a self-describing '# F=<n>' header,
    so a non-default fused-adder width can be checked as well."""
    text = Path(path).read_text()
    m_f = re.search(r"^#\s*F=(\d+)", text, re.M)
    F = int(m_f.group(1)) if m_f else F_DEFAULT

    n = ok = bad = 0
    for raw in text.splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        n += 1
        fmt = E4M3 if m["fmt"] == "e4m3" else E5M2
        a = [int(m["a"][2 * i:2 * i + 2], 16) for i in range(L)]
        b = [int(m["b"][2 * i:2 * i + 2], 16) for i in range(L)]
        got = int(m["d"], 16)
        ref = st_fdpa(a, b, int(m["c"], 16), int(m["sa"], 16), int(m["sb"], 16),
                      fmt=fmt, F=F).bits
        if ref == got:
            ok += 1
        else:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH {path}:{n} fmt={m['fmt']} "
                      f"systemc=0x{got:08X} python=0x{ref:08X}\n"
                      f"    A={m['a']}\n    B={m['b']}\n"
                      f"    SA={m['sa']} SB={m['sb']} C={m['c']}")
    print(f"{path}: {n} blocks (F={F}), {ok} identical, {bad} mismatched")
    return n, ok, bad


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    total = bad = 0
    for p in argv[1:]:
        n, _, b = verify(p)
        total += n
        bad += b
    if total == 0:
        print("no stimulus parsed -- did the dump run?")
        return 2
    print(f"\nTOTAL: {total} blocks, {bad} mismatched "
          f"({L=}, RZ-FP32, E8M0, F from each dump header)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
