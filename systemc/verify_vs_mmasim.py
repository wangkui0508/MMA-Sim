#!/usr/bin/env python3
"""Second-opinion differential test: SystemC model vs the MMA-Sim package.

`verify_vs_reference.py` compares against the standalone transcription
(../mxfp8_stfdpa.py).  This script compares the same SystemC output against the
implementation shipped in the repository package itself,

    src/mmasim/mmasim/arithmetic/fdpa.py :: st_fdpa

configured exactly the way nv_ptx/sim.py configures it for the Blackwell
tcgen05 / mma.sync block-scale MXFP8 path:

    MMA_ST_FDPA(F=25, rho="RZ-FP32", L_max=32, e_zero=-133)

NOTE: only the module-level `st_fdpa` is used.  `MMA_ST_FDPA.__call__` is not
usable as a drop-in reference here: its `B.T[None, :, :]` / beta expansion
produces a [m, k, n]-shaped operand, so it only agrees with the documented
[..., K] contract when n == k and it cannot be applied to a single block.

    python3 verify_vs_mmasim.py mxfp8_dump.txt [more dumps...]
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "mmasim"))
from mmasim.arithmetic.fdpa import st_fdpa as mmasim_st_fdpa  # noqa: E402

LINE = re.compile(
    r"FMT=(?P<fmt>e4m3|e5m2) A=(?P<a>[0-9A-F]{64}) B=(?P<b>[0-9A-F]{64}) "
    r"SA=(?P<sa>[0-9A-F]{2}) SB=(?P<sb>[0-9A-F]{2}) C=(?P<c>[0-9A-F]{8}) D=(?P<d>[0-9A-F]{8})"
)

F, L = 25, 32
DTYPE = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}

# exactly the parameters nv_ptx/sim.py picks for Blackwell block-scale MXFP8
PARAMS = dict(F=25, rho="RZ-FP32", e_zero=-133)


def u8(hexstr: str) -> torch.Tensor:
    return torch.tensor([int(hexstr[i:i + 2], 16) for i in range(0, len(hexstr), 2)],
                        dtype=torch.uint8)


def verify(path: str, limit: int) -> tuple[int, int]:
    n = bad = 0
    for raw in Path(path).read_text().splitlines():
        m = LINE.match(raw.strip())
        if not m or n >= limit:
            continue
        n += 1
        dt = DTYPE[m["fmt"]]
        a = u8(m["a"]).view(dt)
        b = u8(m["b"]).view(dt)
        sa = u8(m["sa"]).view(torch.float8_e8m0fnu)          # [1]
        sb = u8(m["sb"]).view(torch.float8_e8m0fnu)
        c = torch.tensor(struct.unpack(">f", bytes.fromhex(m["c"]))[0],
                         dtype=torch.float32)  # 0-dim

        ref = mmasim_st_fdpa(a, b, c, sa, sb, **PARAMS)
        ref_bits = int(ref.view(torch.int32).item()) & 0xFFFFFFFF
        got = int(m["d"], 16)
        if ref_bits == got:
            bad += 0
        else:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH {path}:{n} fmt={m['fmt']} "
                      f"systemc=0x{got:08X} mmasim=0x{ref_bits:08X}")
    print(f"{path}: {n} blocks vs mmasim, {n - bad} identical, {bad} mismatched")
    return n, bad


def main(argv: list[str]) -> int:
    limit = int(argv[0]) if argv and argv[0].isdigit() else 3000
    files = [a for a in argv if not a.isdigit()]
    if not files:
        print(__doc__)
        return 2
    total = bad = 0
    for p in files:
        n, b = verify(p, limit)
        total += n
        bad += b
    print(f"\nTOTAL: {total} blocks vs mmasim, {bad} mismatched ({PARAMS})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
