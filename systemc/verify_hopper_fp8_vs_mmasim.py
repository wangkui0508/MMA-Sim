#!/usr/bin/env python3
"""Differential test: SystemC Hopper (SM90) FP8 wgmma model vs the MMA-Sim package.

Reads the block dumps produced by

    ./hopper_fp8_wgmma_tb --dump N --fmt e4m3|e5m2|mixed --seed S

and re-derives every block with the repository's own model, configured exactly
the way nv_ptx/sim.py configures it for the Hopper FP8 wgmma path:

    WGMMA("Hopper", "m64n8k32.f32.e4m3.e4m3")   -> F=13, L_max=32, rho=RZ-E8M13
    WGMMA("Hopper", "m64n8k32.f16.e4m3.e4m3")   -> F=13, L_max=32, rho=RNE-FP16
    (and the .e5m2.e5m2 variants)

So this is not a transcription check: it is the shipped MMA-Sim model that the
SystemC datapath is measured against.

Dump line format (decimal, whitespace separated, after a `# fmt=...` header):

    a0 .. a31  b0 .. b31  c32_bits  c16_bits  d_fp32_bits  d_fp16_bits

`c32_bits` is the FP32 accumulator for the `.f32` variant and `c16_bits` the
FP16 accumulator for the `.f16` variant, because for wgmma the accumulator type
IS the output type (`isa_wgmma.py`: c_type = d_type). Each variant is checked
against WGMMA("Hopper", ...) using its own accumulator format.

    python3 verify_hopper_fp8_vs_mmasim.py hopper_dump.txt [more dumps...]

An integer first argument caps the number of blocks checked per file
(default 3000, like verify_vs_mmasim.py).

The dump header records the operating point (`F=`, `L=`, `e_zero=`) it was
generated with; those values are parsed and asserted against the parameters
`WGMMA("Hopper", ...)` actually uses, so a dump produced at a different
operating point fails loudly instead of being silently mis-verified.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "mmasim"))

# mmasim.nv_ptx imports arithmetic.fma, which JIT-compiles a C++ extension at
# import time. We only need the pure-PyTorch FDPA path, so stub that module out.
import mmasim.arithmetic as _arith  # noqa: E402

_stub = types.ModuleType("mmasim.arithmetic.fma")
_stub.MMA_FMA = object
sys.modules["mmasim.arithmetic.fma"] = _stub
setattr(_arith, "fma", _stub)

from mmasim.nv_ptx import WGMMA  # noqa: E402

DTYPE = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
CHUNK = 4096

# What nv_ptx/sim.py :: WGMMA configures for the Hopper FP8 wgmma path.
EXPECT_F, EXPECT_L, EXPECT_E_ZERO = 13, 32, -133


def parse_header(line: str) -> dict:
    """`# fmt=e4m3 F=13 L=32 e_zero=-133 ...` -> dict."""
    out = {}
    for tok in line.lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    if "fmt" not in out:
        raise ValueError(f"dump header has no fmt=: {line!r}")
    if out["fmt"] not in DTYPE:
        raise ValueError(f"unknown fp8 format {out['fmt']!r} in dump header")
    for key, want in (("F", EXPECT_F), ("L", EXPECT_L), ("e_zero", EXPECT_E_ZERO)):
        if key in out and int(out[key]) != want:
            raise ValueError(
                f"dump was generated at {key}={out[key]} but WGMMA(\"Hopper\", ...) uses "
                f"{key}={want}; regenerate the dump"
            )
    return out


def load(path: str):
    """-> [(fmt, A, B, C32, C16, D32, D16), ...], one tuple per format section."""
    sections = []
    fmt = None
    A = B = C32 = C16 = D32 = D16 = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if fmt is not None and A:
                sections.append((fmt, A, B, C32, C16, D32, D16))
            fmt = parse_header(line)["fmt"]
            A, B, C32, C16, D32, D16 = [], [], [], [], [], []
            continue
        v = [int(x) for x in line.split()]
        if len(v) != 68:
            raise ValueError(f"{path}: expected 68 fields, got {len(v)}")
        A.append(v[0:32])
        B.append(v[32:64])
        C32.append(v[64])
        C16.append(v[65])
        D32.append(v[66])
        D16.append(v[67])
    if fmt is not None and A:
        sections.append((fmt, A, B, C32, C16, D32, D16))
    if not sections:
        raise ValueError(f"{path}: no `# fmt=...` section found")
    return sections


def check_section(fmt: str, A, B, C32, C16, D32, D16, limit: int, path: str):
    n = min(len(A), limit)
    if n == 0:
        return 0, 0
    dt = DTYPE[fmt]
    w32 = WGMMA("Hopper", f"m64n8k32.f32.{fmt}.{fmt}")
    w16 = WGMMA("Hopper", f"m64n8k32.f16.{fmt}.{fmt}")

    bad = 0
    ta_all = torch.tensor(A[:n], dtype=torch.uint8).view(dt)
    tb_all = torch.tensor(B[:n], dtype=torch.uint8).view(dt)
    # one accumulator per variant, each in its own format
    tc32_all = torch.tensor(C32[:n], dtype=torch.uint32).view(torch.float32)
    tc16_all = torch.tensor(C16[:n], dtype=torch.uint16).view(torch.float16)
    d32_all = torch.tensor(D32[:n], dtype=torch.int64)
    d16_all = torch.tensor(D16[:n], dtype=torch.int64)

    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        r32 = w32.dpa(ta_all[s:e], tb_all[s:e], tc32_all[s:e]).view(torch.int32)
        r32 = r32.to(torch.int64) & 0xFFFFFFFF
        r16 = w16.dpa(ta_all[s:e], tb_all[s:e], tc16_all[s:e]).view(torch.int16)
        r16 = r16.to(torch.int64) & 0xFFFF
        for j in range(e - s):
            i = s + j
            if int(r32[j]) != int(d32_all[i]) or int(r16[j]) != int(d16_all[i]):
                bad += 1
                if bad <= 5:
                    print(
                        f"  MISMATCH {path} [{fmt}] block {i + 1} "
                        f"systemc f32=0x{int(d32_all[i]):08X} f16=0x{int(d16_all[i]):04X} | "
                        f"mmasim f32=0x{int(r32[j]):08X} f16=0x{int(r16[j]):04X}"
                    )
    print(f"  {path} [{fmt}]: {n} blocks vs mmasim, {n - bad} identical, {bad} mismatched")
    return n, bad


def main(argv: list[str]) -> int:
    limit = int(argv[0]) if argv and argv[0].isdigit() else 3000
    files = [a for a in argv if not a.isdigit()]
    if not files:
        print(__doc__)
        return 2
    total = bad = 0
    for p in files:
        sections = load(p)
        for fmt, A, B, C32, C16, D32, D16 in sections:
            n, b = check_section(fmt, A, B, C32, C16, D32, D16, limit, p)
            total += n
            bad += b
    print(f"\nTOTAL: {total} blocks vs mmasim (Hopper FP8 wgmma), {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
