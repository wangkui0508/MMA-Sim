"""
mxfp8_stfdpa.py — bit-accurate reference model for the MXFP8 dot-product-accumulate
performed by NVIDIA Blackwell 5th-gen Tensor Cores.

Models the ST-FDPA (Scaled Truncated Fused Dot-Product-Add) algorithm described in
MMA-Sim (arXiv:2511.10909, Xie et al., Microsoft Research), which is the bit-accurate
model for the SASS instructions QMMA.SF (SM120) and UTCQMMA (SM100), i.e. the PTX
    tcgen05.mma.cta_group::N.kind::mxf8f6f4.block_scale.scale_vec::1X
    mma.sync.aligned.m16n8k32...kind::mxf8f6f4.block_scale.scale_vec::1X
paths with FP8 elements.

Pipeline (one 32-element block == one fused dot product):

  1. MULTIPLY   exact product of the two *signed significands* (fixed point, no
                normalisation), and integer sum of exponents INCLUDING both E8M0
                scale exponents:
                    s_k = SignedSig(a_k) * SignedSig(b_k)
                    e_k = Exp(a_k) + Exp(b_k) + Exp(alpha) + Exp(beta)
                This is where the block scale enters: purely in the exponent
                domain, BEFORE alignment. E8M0 has an implicit significand of
                exactly 1.0, so no significand multiply is needed.
  2. ALIGN      c joins as term L (early addition, not a separate later add).
                All L+1 terms are aligned to e_max = max(e_0..e_L) and truncated
                toward zero to F fractional bits.
  3. ACCUMULATE exact fixed-point sum of the aligned terms (Python ints -> exact).
  4. ROUND      single conversion of S * 2^(e_max-F) to FP32 with round-toward-zero.

Because products are left DENORMALISED, the significand of a product can be >= 2,
which consumes the second integer bit of the (2, 25) alignment format and shifts
the effective resolution by one bit. Two algebraically identical products can
therefore give different results. See test_denormalised_product() below.

Verified parameters (MMA-Sim Table 4): Blackwell / RTX Blackwell, MXFP8/6/4 -> FP32,
L_max = 32, F = 25, rho = RZ-FP32.

KNOWN LIMITATION -- alignment exponent floor for non-zero terms. Khattak & Mikaitis
(arXiv:2512.07004) found that for fp16/bf16/tf19 inputs the exponent used for
alignment is clamped from below (e_min_align = -132 on Ampere/Ada, -133 on
Hopper/Blackwell) when all products fall in the FP32 subnormal range. For the fp8
rows their table lists no such value, and MMA-Sim models no such clamp for
non-zero terms. There IS one zero-specific floor that matters: zero terms
(including c == 0) join the summation at e_zero = -133 and can thereby raise
e_max and truncate tiny products to +0.0; this file models that (see st_fdpa).
This matters MORE for MXFP8 than for plain FP8, because an E8M0 scale can be as
small as 2**-127 and can therefore push product exponents deep into the
subnormal region on its own. If you care about corners below that, probe them on
real hardware before trusting any model.

Everything else here is a faithful transcription of the published algorithm; it has
NOT been validated against a physical B200 by the author of this file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Floating-point format descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FpFormat:
    """Minimal descriptor for a binary floating-point interchange format.

    has_inf=False models OCP FP8-E4M3, which has no infinities: the all-ones
    exponent field is a normal number range, and only S.1111.111 is NaN.
    """

    name: str
    exp_bits: int
    man_bits: int
    has_inf: bool

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def emin(self) -> int:
        """Unbiased exponent of the smallest normal (and of all subnormals)."""
        return 1 - self.bias

    @property
    def width(self) -> int:
        return 1 + self.exp_bits + self.man_bits


E4M3 = FpFormat("e4m3", 4, 3, has_inf=False)
E5M2 = FpFormat("e5m2", 5, 2, has_inf=True)
FP32 = FpFormat("fp32", 8, 23, has_inf=True)

# ---------------------------------------------------------------------------
# Decoded term
# ---------------------------------------------------------------------------

ZERO, FINITE, INF, NAN = "zero", "finite", "inf", "nan"


@dataclass(frozen=True)
class Term:
    """A decoded value.

    For kind == FINITE the value is exactly

        sign * sig * 2**(-P) * 2**exp

    where sig is an unsigned integer, P the number of fractional bits held in
    sig, and exp the format's unbiased exponent (== emin for subnormals). This
    is the SignedSig/Exp split used by the ST-FDPA algorithm: note that for
    subnormals sig < 2**P, i.e. the significand is less than 1.0, and no
    normalisation is performed anywhere.
    """

    kind: str
    sign: int = 1  # +1 or -1, meaningful for ZERO/FINITE/INF
    sig: int = 0
    P: int = 0
    exp: int = 0

    def scaled_exp(self, extra: int) -> "Term":
        if self.kind != FINITE:
            return self
        return Term(FINITE, self.sign, self.sig, self.P, self.exp + extra)


def decode(bits: int, fmt: FpFormat) -> Term:
    """Decode a raw bit pattern into a Term."""
    bits &= (1 << fmt.width) - 1
    sign = -1 if (bits >> (fmt.exp_bits + fmt.man_bits)) & 1 else 1
    E = (bits >> fmt.man_bits) & ((1 << fmt.exp_bits) - 1)
    M = bits & ((1 << fmt.man_bits) - 1)
    E_max_field = (1 << fmt.exp_bits) - 1

    if E == E_max_field:
        if fmt.has_inf:
            return Term(NAN) if M else Term(INF, sign)
        # E4M3: only the all-ones mantissa is NaN; the rest are normals.
        if M == (1 << fmt.man_bits) - 1:
            return Term(NAN)

    if E == 0:
        if M == 0:
            return Term(ZERO, sign)
        return Term(FINITE, sign, M, fmt.man_bits, fmt.emin)  # subnormal

    return Term(FINITE, sign, (1 << fmt.man_bits) | M, fmt.man_bits, E - fmt.bias)


def decode_ue8m0(bits: int) -> Optional[int]:
    """Decode an OCP E8M0 block scale.

    Returns the power-of-two exponent, or None for NaN (encoding 0xFF).
    Encoding 0x00 is 2**-127, not zero: E8M0 cannot represent zero or Inf.
    """
    bits &= 0xFF
    if bits == 0xFF:
        return None
    return bits - 127


# ---------------------------------------------------------------------------
# Exact value -> FP32 conversion
# ---------------------------------------------------------------------------


def _shift_right_rz(v: int, n: int) -> int:
    """Arithmetic right shift that truncates the MAGNITUDE (round toward zero).

    Python's >> rounds toward -inf, which would be round-DOWN, not round-to-zero.
    MMA-Sim models the NVIDIA fused summation with RZ (symmetric); the asymmetric
    RD variant appears only in the AMD CDNA3 models (TR-FDPA / GTR-FDPA), where
    it produces a measurable negative bias.
    """
    if n <= 0:
        return v << (-n)
    if v >= 0:
        return v >> n
    return -((-v) >> n)


def exact_to_fp32_bits(S: int, exp2: int, *, overflow_to_inf: bool = True) -> int:
    """Convert the exact value S * 2**exp2 to FP32 bits with round-toward-zero.

    overflow_to_inf: on magnitude overflow, return +/-Inf (matching the
    finite-accumulation-overflows-to-Inf behaviour reported for these tensor
    cores) rather than the largest finite value that a literal IEEE RZ
    conversion would produce. Flip it if you want strict IEEE RZ semantics.
    """
    if S == 0:
        return 0  # +0.0; exact cancellation is reported as +0 by these units
    sign_bit = 0x80000000 if S < 0 else 0
    M = abs(S)
    E = exp2 + M.bit_length() - 1  # unbiased exponent of the leading 1

    if E < FP32.emin:  # subnormal or underflow to zero
        m = _shift_right_rz(M, -149 - exp2)
        if m == 0:
            return sign_bit  # underflow preserves the sign (trunc(-tiny) = -0.0)
        return sign_bit | m  # exponent field 0

    frac = _shift_right_rz(M, M.bit_length() - (FP32.man_bits + 1))
    if E > FP32.bias:  # overflow
        if overflow_to_inf:
            return sign_bit | 0x7F800000
        return sign_bit | 0x7F7FFFFF
    return sign_bit | ((E + FP32.bias) << FP32.man_bits) | (frac - (1 << FP32.man_bits))


# ---------------------------------------------------------------------------
# ST-FDPA
# ---------------------------------------------------------------------------


@dataclass
class FdpaResult:
    bits: int  # raw FP32 output
    value: float
    e_max: Optional[int]  # None if the result came from a special-value path
    S: Optional[int]  # exact fixed-point accumulator, in units of 2**(e_max-F)
    aligned: Optional[List[int]]  # per-term aligned significands, same units
    note: str = ""


def st_fdpa(
    a_bits: Sequence[int],
    b_bits: Sequence[int],
    c_bits: int,
    sfa_bits: int,
    sfb_bits: int,
    *,
    fmt: FpFormat = E4M3,
    F: int = 25,
    overflow_to_inf: bool = True,
) -> FdpaResult:
    """One fused, scaled dot-product-accumulate over a single MX block.

    a_bits / b_bits : raw FP8 encodings, len(a) == len(b) <= 32
    c_bits          : raw FP32 encoding of the accumulator
    sfa_bits/sfb_bits: raw E8M0 block scales for A and B
    F               : fractional bits of the fused summation (25 on Blackwell)
    """
    if len(a_bits) != len(b_bits):
        raise ValueError("a and b must have the same length")
    if len(a_bits) > 32:
        raise ValueError("one MXFP8 block is at most 32 elements; use mxfp8_dot()")

    ea = decode_ue8m0(sfa_bits)
    eb = decode_ue8m0(sfb_bits)
    scale_nan = ea is None or eb is None
    scale_exp = 0 if scale_nan else ea + eb

    ta = [decode(x, fmt) for x in a_bits]
    tb = [decode(x, fmt) for x in b_bits]
    tc = decode(c_bits, FP32)

    def nan_result(why: str) -> FdpaResult:
        # NVIDIA canonical FP32 NaN
        return FdpaResult(0x7FFFFFFF, float("nan"), None, None, None, why)

    if scale_nan:
        return nan_result("scale factor is NaN")
    if any(t.kind == NAN for t in ta) or any(t.kind == NAN for t in tb):
        return nan_result("NaN in A or B")
    if tc.kind == NAN:
        return nan_result("NaN in C")

    # Products: form terms, detect Inf*0 and collect infinity signs.
    terms: List[Term] = []
    inf_signs = set()
    for x, y in zip(ta, tb):
        if x.kind == INF or y.kind == INF:
            if x.kind == ZERO or y.kind == ZERO:
                return nan_result("Inf * 0")
            inf_signs.add(x.sign * y.sign)
            continue
        if x.kind == ZERO or y.kind == ZERO:
            continue
        terms.append(
            Term(FINITE, x.sign * y.sign, x.sig * y.sig, x.P + y.P,
                 x.exp + y.exp + scale_exp)
        )

    if tc.kind == INF:
        inf_signs.add(tc.sign)
    elif tc.kind == FINITE:
        terms.append(tc)

    if inf_signs:
        if len(inf_signs) > 1:
            return nan_result("+Inf and -Inf in the same dot product")
        s = inf_signs.pop()
        bits = (0x80000000 if s < 0 else 0) | 0x7F800000
        return FdpaResult(bits, float("-inf") if s < 0 else float("inf"),
                          None, None, None, "infinite term dominates")

    if not terms:
        return FdpaResult(0, 0.0, None, 0, [], "all terms zero")

    # Single global alignment to the largest exponent. Non-zero c joins as an
    # ordinary term; a ZERO c is dropped here but contributes an alignment
    # floor at e_zero = -133 (see below), which is what "c == 0 is a special
    # case, not a subnormal" refers to.
    e_max = max(t.exp for t in terms)
    if tc.kind == ZERO:
        # MMA-Sim keeps zero terms in the summation at e = e_zero = -133
        # (fdpa.py st_fdpa: "e[s == 0.0] = e_zero"). With c == 0 that floor
        # can become e_max and kill (to +0.0) products more than 25 bins
        # below it, which the exact-sum model below would otherwise carry
        # into the result. Zero *products* sit at the same -133 and only
        # bind when c is also zero, so the floor on c alone is equivalent.
        e_max = max(e_max, -133)

    aligned = []
    for t in terms:
        shift = (t.exp - e_max) + (F - t.P)
        aligned.append(_shift_right_rz(t.sign * t.sig, -shift))

    S = sum(aligned)  # exact: Python ints
    bits = exact_to_fp32_bits(S, e_max - F, overflow_to_inf=overflow_to_inf)
    return FdpaResult(bits, fp32_bits_to_float(bits), e_max, S, aligned)


def st_fdpa_n(
    a_bits: Sequence[int],
    b_bits: Sequence[int],
    c_bits: int,
    sfa_bits: Sequence[int],
    sfb_bits: Sequence[int],
    *,
    fmt: FpFormat = E4M3,
    F: int = 25,
    block: int = 32,
    overflow_to_inf: bool = True,
) -> FdpaResult:
    """Generalized N-term fused scaled dot-product-accumulate: K products + c.

    Identical arithmetic to st_fdpa -- one global alignment to e_max, per-term
    truncation to F fractional bits, exact integer sum, a single RZ-FP32
    conversion -- but the fused summation spans K products, and each
    `block`-element chunk of the K elements carries its own E8M0 scale pair
    (MX semantics). K=32 with one scale pair is exactly st_fdpa.

    SEMANTICS WARNING: for K > 32 this models NO real TensorCore instruction.
    The hardware path for K > 32 is the per-block chain (mxfp8_dot), whose
    intermediate FP32 roundings make it a different arithmetic: a 65/129-term
    single node disagrees with the chain on a substantial fraction of inputs.
    Use this to DEFINE a wider fused adder (e.g. N=65 or N=129 tree nodes),
    not to claim hardware bit-exactness. The scale-NaN and Inf*0 paths return
    the same canonical NaN the chain would end at.
    """
    K = len(a_bits)
    if K == 0 or len(b_bits) != K:
        raise ValueError("a and b must be non-empty and have the same length")
    nblocks = (K + block - 1) // block
    if len(sfa_bits) != nblocks or len(sfb_bits) != nblocks:
        raise ValueError(f"expected {nblocks} scale pairs for K={K}, block={block}")

    scale_exps: List[Optional[int]] = []
    for j in range(nblocks):
        ea = decode_ue8m0(sfa_bits[j])
        eb = decode_ue8m0(sfb_bits[j])
        scale_exps.append(None if ea is None or eb is None else ea + eb)

    ta = [decode(x, fmt) for x in a_bits]
    tb = [decode(x, fmt) for x in b_bits]
    tc = decode(c_bits, FP32)

    def nan_result(why: str) -> FdpaResult:
        # NVIDIA canonical FP32 NaN
        return FdpaResult(0x7FFFFFFF, float("nan"), None, None, None, why)

    if any(e is None for e in scale_exps):
        # the per-block chain would poison its running accumulator and end NaN
        return nan_result("scale factor is NaN")
    if any(t.kind == NAN for t in ta) or any(t.kind == NAN for t in tb):
        return nan_result("NaN in A or B")
    if tc.kind == NAN:
        return nan_result("NaN in C")

    # Products: form terms, detect Inf*0 and collect infinity signs.
    terms: List[Term] = []
    inf_signs = set()
    for k, (x, y) in enumerate(zip(ta, tb)):
        if x.kind == INF or y.kind == INF:
            if x.kind == ZERO or y.kind == ZERO:
                return nan_result("Inf * 0")
            inf_signs.add(x.sign * y.sign)
            continue
        if x.kind == ZERO or y.kind == ZERO:
            continue
        terms.append(
            Term(FINITE, x.sign * y.sign, x.sig * y.sig, x.P + y.P,
                 x.exp + y.exp + scale_exps[k // block])
        )

    if tc.kind == INF:
        inf_signs.add(tc.sign)
    elif tc.kind == FINITE:
        terms.append(tc)

    if inf_signs:
        if len(inf_signs) > 1:
            return nan_result("+Inf and -Inf in the same dot product")
        s = inf_signs.pop()
        bits = (0x80000000 if s < 0 else 0) | 0x7F800000
        return FdpaResult(bits, float("-inf") if s < 0 else float("inf"),
                          None, None, None, "infinite term dominates")

    if not terms:
        return FdpaResult(0, 0.0, None, 0, [], "all terms zero")

    e_max = max(t.exp for t in terms)
    if tc.kind == ZERO:
        e_max = max(e_max, -133)  # zero terms sit at e_zero = -133

    aligned = []
    for t in terms:
        shift = (t.exp - e_max) + (F - t.P)
        aligned.append(_shift_right_rz(t.sign * t.sig, -shift))

    S = sum(aligned)  # exact: Python ints
    bits = exact_to_fp32_bits(S, e_max - F, overflow_to_inf=overflow_to_inf)
    return FdpaResult(bits, fp32_bits_to_float(bits), e_max, S, aligned)


def mxfp8_dot(
    a_bits: Sequence[int],
    b_bits: Sequence[int],
    sfa_bits: Sequence[int],
    sfb_bits: Sequence[int],
    c_bits: int = 0,
    *,
    fmt: FpFormat = E4M3,
    F: int = 25,
    block: int = 32,
) -> int:
    """Chain fused dot products over K > 32, one MX block at a time.

    Each block of 32 K-elements has its own scale pair and its own fused
    summation; the FP32 result of one block becomes the c of the next. This
    reproduces what a real GEMM main loop does, and it means the per-block
    rounding is part of the answer -- accumulating over K is NOT one big exact
    sum. Returns raw FP32 bits.
    """
    K = len(a_bits)
    if len(b_bits) != K:
        raise ValueError("a and b must have the same length")
    nblocks = (K + block - 1) // block
    if len(sfa_bits) != nblocks or len(sfb_bits) != nblocks:
        raise ValueError(f"expected {nblocks} scale factors per operand")

    d = c_bits
    for i in range(nblocks):
        lo, hi = i * block, min((i + 1) * block, K)
        r = st_fdpa(a_bits[lo:hi], b_bits[lo:hi], d, sfa_bits[i], sfb_bits[i],
                    fmt=fmt, F=F)
        d = r.bits
    return d


# ---------------------------------------------------------------------------
# Conversion helpers (convenience only -- not part of the hardware model)
# ---------------------------------------------------------------------------


def fp32_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def float_to_fp32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _fp8_table(fmt: FpFormat):
    out = []
    for bits in range(1 << fmt.width):
        t = decode(bits, fmt)
        if t.kind == FINITE:
            out.append((t.sign * t.sig * 2.0 ** (t.exp - t.P), bits))
        elif t.kind == ZERO:
            out.append((0.0, bits))
    return out


_TABLES = {f.name: _fp8_table(f) for f in (E4M3, E5M2)}


def float_to_fp8_bits(x: float, fmt: FpFormat = E4M3) -> int:
    """Nearest-even quantisation of a float to an FP8 encoding, saturating.

    Brute-force over all encodings: slow but unambiguous. Only for building
    test inputs -- the hardware model itself never converts from float.
    """
    table = _TABLES[fmt.name]
    best = min(table, key=lambda vb: (abs(vb[0] - x), vb[1] & 1))
    return best[1]


def fp8_bits_to_float(bits: int, fmt: FpFormat = E4M3) -> float:
    t = decode(bits, fmt)
    if t.kind == ZERO:
        return -0.0 if t.sign < 0 else 0.0
    if t.kind == INF:
        return float("-inf") if t.sign < 0 else float("inf")
    if t.kind == NAN:
        return float("nan")
    return t.sign * t.sig * 2.0 ** (t.exp - t.P)


def e8m0_bits(exp: int) -> int:
    """Encode the scale 2**exp as an E8M0 byte."""
    if not -127 <= exp <= 127:
        raise ValueError("E8M0 exponent out of range [-127, 127]")
    return exp + 127


def encode_fp8_vector(values: Sequence[float], fmt: FpFormat = E4M3) -> List[int]:
    return [float_to_fp8_bits(v, fmt) for v in values]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _hexf(bits: int) -> str:
    return f"0x{bits:08X} ({fp32_bits_to_float(bits)!r})"


def test_basic():
    a = encode_fp8_vector([1.0, 2.0, 3.0])
    b = encode_fp8_vector([1.0, 2.0, 3.0])
    r = st_fdpa(a, b, float_to_fp32_bits(0.5), e8m0_bits(0), e8m0_bits(0))
    assert r.value == 14.5, r
    print(f"basic dot product           : {r.value}  (expected 14.5)")


def test_scale_is_a_pure_exponent_shift():
    """Scaling by 2**p must shift the result by exactly 2**p, bit for bit."""
    a = encode_fp8_vector([1.5, -0.75, 3.5, 0.125] * 8)
    b = encode_fp8_vector([2.0, 1.25, -1.5, 6.0] * 8)
    base = st_fdpa(a, b, 0, e8m0_bits(0), e8m0_bits(0))
    for p, q in [(10, 0), (0, 10), (5, 5), (-20, 3), (30, -30)]:
        r = st_fdpa(a, b, 0, e8m0_bits(p), e8m0_bits(q))
        assert r.value == base.value * 2.0 ** (p + q), (p, q, r.value)
    print("scale == exponent shift     : ok (5 scale pairs, exact)")


def test_scale_moves_e_max_not_the_relative_alignment():
    """The scale is shared by the whole block, so it cannot change which terms
    get truncated -- only the absolute exponent. S must be identical."""
    a = encode_fp8_vector([1.5, 0.03125, 0.001953125] + [0.0] * 29)
    b = encode_fp8_vector([1.5, 0.0625, 0.001953125] + [0.0] * 29)
    r0 = st_fdpa(a, b, 0, e8m0_bits(0), e8m0_bits(0))
    r1 = st_fdpa(a, b, 0, e8m0_bits(40), e8m0_bits(-11))
    assert r0.S == r1.S and r1.e_max == r0.e_max + 29
    print(f"scale shifts e_max only     : S={r0.S}, e_max {r0.e_max} -> {r1.e_max}")


def test_c_is_added_early():
    """c participates in e_max and in the same fused summation, so a huge c
    swamps the products entirely rather than being added afterwards."""
    a = encode_fp8_vector([1.0] * 32)
    b = encode_fp8_vector([2.0 ** -9] * 32)  # 32 * 2**-9 = 2**-4
    c = float_to_fp32_bits(2.0 ** 24)
    r = st_fdpa(a, b, c, e8m0_bits(0), e8m0_bits(0))
    assert r.value == 2.0 ** 24, r
    print(f"early c swamps products     : {r.value} (products truncated away)")


def test_truncation_at_25_fractional_bits():
    """A term far enough below e_max is discarded during alignment.

    Same tiny product 2**-18 in both runs; only the dominant term moves. With
    F=25 the survival threshold sits between the two cases.
    """
    tiny = 2.0 ** -9

    def run(big):
        a = encode_fp8_vector([big, tiny] + [0.0] * 30)
        b = encode_fp8_vector([big, tiny] + [0.0] * 30)
        return st_fdpa(a, b, 0, e8m0_bits(0), e8m0_bits(0))

    near = run(1.0)  # e_max = 0,  resolution 2**-25
    far = run(256.0)  # e_max = 16, resolution 2**-9
    assert near.aligned[1] == 128 and near.value == 1.0 + 2.0 ** -18
    assert far.aligned[1] == 0 and far.value == 65536.0
    print(f"alignment truncation        : e_max=0  keeps 2**-18 -> {near.value!r}")
    print(f"                              e_max=16 drops it     -> {far.value!r}")


def test_denormalised_product():
    """The headline non-IEEE behaviour.

    Two ways to obtain the same product 288:
        A: 12 * 24   -> SignedSig = 2.25 (>= 2, uses the 2nd integer bit), Exp = 7
        B: 18 * 16   -> SignedSig = 1.125,                                 Exp = 8
    Products are never normalised, so case B aligns one bit coarser and loses
    the low bits of the small addends. Two extra terms sum to exactly one ulp
    of 288, so the difference survives the final RZ to FP32.
    """
    tiny1 = (2.0 ** -7 + 2.0 ** -8 + 2.0 ** -9, 2.0 ** -9)  # 1.75*2**-16
    tiny2 = (2.0 ** -9, 2.0 ** -9)  # 2**-18

    def run(a0, b0):
        a = encode_fp8_vector([a0, tiny1[0], tiny2[0]] + [0.0] * 29)
        b = encode_fp8_vector([b0, tiny1[1], tiny2[1]] + [0.0] * 29)
        return st_fdpa(a, b, 0, e8m0_bits(0), e8m0_bits(0))

    A = run(12.0, 24.0)
    B = run(18.0, 16.0)
    assert A.e_max == 7 and B.e_max == 8, (A.e_max, B.e_max)
    assert A.value == 288.0 + 2.0 ** -15, A.value
    assert B.value == 288.0, B.value
    print(f"denormalised product        : A(1.5*1.5 form) = {A.value!r}")
    print(f"                              B(1.0*2.25 form) = {B.value!r}  <- one bit lost")


def test_rz_not_rne():
    """Output rounding is truncation, so the result never grows in magnitude.

    The accumulator holds 1 + 2**-25 exactly, but 1 ulp of 1.0 is 2**-23, so
    RZ throws the remainder away. RNE would too here; the point of the check is
    that the accumulator is strictly larger than the reported result.
    """
    # E5M2 so that a product of exactly 2**-25 is reachable: 2**-13 * 2**-12.
    a = encode_fp8_vector([1.0, 2.0 ** -13] + [0.0] * 30, E5M2)
    b = encode_fp8_vector([1.0, 2.0 ** -12] + [0.0] * 30, E5M2)
    r = st_fdpa(a, b, 0, e8m0_bits(0), e8m0_bits(0), fmt=E5M2)
    assert r.S == (1 << 25) + 1 and r.value == 1.0
    print(f"RZ output rounding          : S={r.S} = 2**25+1, result still {r.value}")


def test_special_values():
    ok = 0
    r = st_fdpa(encode_fp8_vector([1.0]), encode_fp8_vector([1.0]), 0, 0xFF, e8m0_bits(0))
    assert r.bits == 0x7FFFFFFF; ok += 1  # NaN scale poisons the block
    r = st_fdpa([0x7F], encode_fp8_vector([1.0]), 0, e8m0_bits(0), e8m0_bits(0))
    assert r.bits == 0x7FFFFFFF; ok += 1  # E4M3 NaN encoding
    r = st_fdpa([0x7C], [0x00], 0, e8m0_bits(0), e8m0_bits(0), fmt=E5M2)
    assert r.bits == 0x7FFFFFFF; ok += 1  # Inf * 0
    r = st_fdpa([0x7C, 0xFC], [0x3C, 0x3C], 0, e8m0_bits(0), e8m0_bits(0), fmt=E5M2)
    assert r.bits == 0x7FFFFFFF; ok += 1  # +Inf + -Inf
    r = st_fdpa([0x7C], [0x3C], 0, e8m0_bits(0), e8m0_bits(0), fmt=E5M2)
    assert r.bits == 0x7F800000; ok += 1  # +Inf survives
    print(f"special values              : {ok}/5 paths ok (NaN = 0x7FFFFFFF)")


def test_subnormal_input_and_output():
    a = encode_fp8_vector([2.0 ** -9])  # E4M3 min subnormal
    b = encode_fp8_vector([1.0])
    r = st_fdpa(a, b, 0, e8m0_bits(-120), e8m0_bits(0))
    assert r.value == 2.0 ** -129 and (r.bits >> 23) & 0xFF == 0
    print(f"subnormal in and out        : {r.value!r} (FP32 subnormal, exp field 0)")


def test_multi_block_chaining():
    """K=64 is two fused summations with a rounding in between, not one."""
    a = encode_fp8_vector([1.0] + [2.0 ** -9] * 31 + [1.0] + [2.0 ** -9] * 31)
    b = encode_fp8_vector([2.0 ** 8] + [2.0 ** -9] * 31 + [2.0 ** 8] + [2.0 ** -9] * 31)
    sfa = [e8m0_bits(0), e8m0_bits(0)]
    bits = mxfp8_dot(a, b, sfa, sfa)
    chained = fp32_bits_to_float(bits)
    single = st_fdpa(a[:32], b[:32], 0, sfa[0], sfa[0]).value
    assert chained == 2 * single == 512.0
    print(f"K=64 chained blocks         : {chained} (= 2 x {single})")


def test_zero_c_sets_alignment_floor():
    """c == 0 joins the summation at e_zero = -133 and can raise e_max.

    The product -2**-8 * 2**-8 = -2**-16 lands at e = -16 - 127 - 17 = -160
    with scale 2**-127 * 2**-17. The -133 floor makes e_max = -133, so the
    term sits 27 bins low and is truncated away: +0. All three outputs below
    are diff-tested against MMA-Sim.
    """
    r = st_fdpa([0x82], [0x02], 0, 0x00, e8m0_bits(-17))
    assert r.bits == 0, r  # killed by the floor -> +0
    # scale 2**-127 * 2**-10 puts the same product at e = -153: it survives
    # alignment (8 bins of 2**-158) but underflows FP32, keeping its sign
    r = st_fdpa([0x82], [0x02], 0, 0x00, e8m0_bits(-10))
    assert r.bits == 0x80000000 and r.value == 0.0, r  # -0.0
    # scale 2**-100 * 2**-10 -> -2**-126, the smallest normal
    r = st_fdpa([0x82], [0x02], 0, e8m0_bits(-100), e8m0_bits(-10))
    assert r.bits == 0x80800000, r
    # all-zero inputs with c = -0.0 also report +0 (the +/-0.0 sum is +0)
    r = st_fdpa([0x00], [0x80], 0x80000000, 0x7F, 0x7F)
    assert r.bits == 0, r
    print("zero-c alignment floor      : killed -> +0, underflow keeps -0")


def test_st_fdpa_n_generalization():
    """K=32 must reproduce st_fdpa bit-for-bit; K=64/128 keep the
    scale-is-a-pure-exponent-shift property."""
    import random as _r

    _r.seed(11)
    for _ in range(2000):
        a = [_r.randrange(256) for _ in range(32)]
        b = [_r.randrange(256) for _ in range(32)]
        sfa, sfb = _r.randrange(256), _r.randrange(256)
        c = _r.randrange(1 << 32)
        r1 = st_fdpa(a, b, c, sfa, sfb)
        r2 = st_fdpa_n(a, b, c, [sfa], [sfb])
        assert r1.bits == r2.bits, (r1.bits, r2.bits)
    for K in (64, 128):
        nb = K // 32

        def _finite_vec(n):
            out = []
            for _ in range(n):
                v = _r.randrange(0x7F)  # 0x00..0x7E: finite E4M3
                out.append(v | 0x80 if _r.random() < 0.5 else v)
            return out

        a = _finite_vec(K)
        b = _finite_vec(K)
        sfa = [_r.randrange(255) for _ in range(nb)]
        sfb = [_r.randrange(255) for _ in range(nb)]
        base = st_fdpa_n(a, b, 0, sfa, sfb).value
        assert base != 0.0
        for p, q in [(5, 0), (0, 5), (-7, 7)]:
            r = st_fdpa_n(a, b, 0, [x + p for x in sfa], [x + q for x in sfb])
            assert -126 < _r_log2(base) + p + q < 126  # stay in normal range
            assert r.value == base * 2.0 ** (p + q), (K, p, q, r.value)
    print("st_fdpa_n (65/129-term)      : K=32 identical to st_fdpa; shift property ok")


def _r_log2(x: float) -> int:
    import math
    return int(math.floor(math.log2(abs(x))))


def run_all():
    for t in (
        test_basic,
        test_scale_is_a_pure_exponent_shift,
        test_scale_moves_e_max_not_the_relative_alignment,
        test_c_is_added_early,
        test_truncation_at_25_fractional_bits,
        test_denormalised_product,
        test_rz_not_rne,
        test_special_values,
        test_subnormal_input_and_output,
        test_multi_block_chaining,
        test_zero_c_sets_alignment_floor,
        test_st_fdpa_n_generalization,
    ):
        t()
    print("\nall self-checks passed")


if __name__ == "__main__":
    run_all()
