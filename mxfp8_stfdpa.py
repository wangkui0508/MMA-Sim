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

An OPT-IN simplified normalise stage (RNORM) is also provided for area/power
studies: it replaces the final FP32 conversion with a cheap approximate
normalise that keeps (sign, mantissa, exponent) in a reduced form. It trades a
little precision for a much smaller CLZ/barrel-shifter. It is NOT bit-accurate
to any real hardware (except at the man_bits = W - scan_bits operating point,
which is bit-identical to this reference model); see the "Reduced-precision
normalisation" section below.
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


def exact_to_fp32_bits(
    S: int,
    exp2: int,
    *,
    overflow_to_inf: bool = True,
    rounding: str = "RTZ",
) -> int:
    """Convert the exact value S * 2**exp2 to FP32 bits.

    rounding: "RTZ" (truncate toward zero -- the hardware behaviour), "RNE"
    (nearest, ties to even), or "RNA" (nearest, ties away from zero). The
    mode applies to the 24-bit-significand rounding and to subnormal
    quantisation; underflow to zero keeps the sign in every mode.

    overflow_to_inf: on magnitude overflow, return +/-Inf (matching the
    finite-accumulation-overflows-to-Inf behaviour reported for these tensor
    cores) rather than the largest finite value that a literal IEEE RZ
    conversion would produce. Flip it if you want strict IEEE RZ semantics.
    """
    if rounding not in ("RTZ", "RNE", "RNA"):
        raise ValueError(f"rounding must be RTZ, RNE or RNA, got {rounding!r}")
    if S == 0:
        return 0  # +0.0; exact cancellation is reported as +0 by these units
    sign_bit = 0x80000000 if S < 0 else 0
    M = abs(S)
    E = exp2 + M.bit_length() - 1  # unbiased exponent of the leading 1

    if E < FP32.emin:  # subnormal or underflow to zero
        shift = -149 - exp2
        if shift <= 0:
            m = M << (-shift)  # exact: at least min-subnormal granularity
        else:
            m, rem = divmod(M, 1 << shift)
            half = 1 << (shift - 1)
            if rounding == "RNE" and (rem > half or (rem == half and m & 1)):
                m += 1
            elif rounding == "RNA" and rem >= half:
                m += 1
        if m == 0:
            return sign_bit
        if m > (1 << FP32.man_bits) - 1:  # rounded up into the min normal
            return sign_bit | (1 << FP32.man_bits)
        return sign_bit | m  # exponent field 0

    shift = M.bit_length() - (FP32.man_bits + 1)
    if shift > 0:
        m, rem = divmod(M, 1 << shift)
        half = 1 << (shift - 1)
        if rounding == "RNE" and (rem > half or (rem == half and m & 1)):
            m += 1
        elif rounding == "RNA" and rem >= half:
            m += 1
        if m > (1 << (FP32.man_bits + 1)) - 1:  # carry: 1.11..1 + 1 ulp -> 2**(E+1)
            m >>= 1
            E += 1
    else:
        m = M << (-shift)
    if E > FP32.bias:  # overflow
        if overflow_to_inf:
            return sign_bit | 0x7F800000
        return sign_bit | 0x7F7FFFFF
    return sign_bit | ((E + FP32.bias) << FP32.man_bits) | (m - (1 << FP32.man_bits))


# ---------------------------------------------------------------------------
# Reduced-precision normalisation (RNORM) -- area/power vs precision tradeoff
# ---------------------------------------------------------------------------
# OPT-IN design sketch. NOT part of the bit-accurate Blackwell model; nothing
# here has been validated against hardware. Use st_fdpa_fast / mxfp8_dot_fast
# instead of st_fdpa / mxfp8_dot when you want to explore the tradeoff.
#
# Reference behaviour (exact_to_fp32_bits): the exact fixed-point accumulator
# S * 2**q is converted to FP32 by (a) a full-width CLZ (M.bit_length() - 1),
# (b) a bidirectional barrel shift to 24 bits, (c) truncation (RTZ). In
# hardware (a) is a ~W-input priority tree and (b) a ~log2(W)-level
# bidirectional barrel, with W = F + 1 + ceil(log2(L + 1)) (~32 for the
# default 32-term design).
#
# Three observations make this cheaper without changing the arithmetic that
# surrounds it:
#
#   1. The result only needs to feed the NEXT block's alignment stage, which
#      reads a (sign, mantissa, exponent) triple. It does not need to be an
#      IEEE FP32. The mantissa may carry its leading one EXPLICITLY (no
#      hidden-bit bookkeeping), and it may be narrower than 24 bits.
#   2. The output rounding is RTZ, i.e. pure truncation. There is no rounder
#      and -- crucially -- no round-up carry, so the mantissa is literally the
#      top man_bits bits of the accumulator and can never need a
#      renormalise-after-round step.
#   3. The CLZ does not need to locate the leading one exactly. Its only job
#      is to choose a man_bits-wide window that does NOT cut the top of the
#      value. Any window whose top bit is at or below the true leading one
#      yields an EXACT (m, e) pair for the bits it keeps; the pair simply may
#      show leading zeros in m (under-normalised but not wrong).
#
# Scheme (reduce_normalize):
#
#   * scan_bits-input priority encoder examines the top scan_bits bits of the
#     fixed W-bit register.
#       - HIT (common): the leading one is inside the scanned window, so the
#         exponent is EXACT and m is fully normalised (leading one at bit
#         man_bits-1). Identical to the reference except for the low-bit
#         truncation implied by the narrower mantissa. The barrel shifter only
#         ever right-shifts by scan_bits distinct amounts (a 3-bit control).
#       - MISS (cancellation: the sum fell below the scanned window): emit the
#         FIXED window just below the scan (a wire slice -- no shifter at
#         all). Rule: never over-estimate the exponent, so the pair stays
#         exact; precision degrades smoothly as the sum cancels (fewer
#         significant bits kept). If even that window is empty
#         (|S| < 2**(W - scan_bits - man_bits)) the value is flushed to +/-0
#         -- the only truly lossy corner. fallback="exact" instead pays a
#         full-width CLZ (slow path / second cycle) and never flushes.
#
# Area (order of magnitude, default W=32, man_bits=16, scan_bits=8):
#   CLZ         32-input tree        -> 8-input encoder (~3x smaller, shallower)
#   barrel      bidirectional 32->24 -> right-only 32->16, 3-bit control
#               (~5-6 mux levels     (~3 levels; the miss path is free)
#                both directions)
#   rounder     none either way (RTZ truncation, no carry)
#   c feedback  32-bit FP32 (24-bit  -> 1 + man_bits + ~10 = 27-bit (sign,m,e)
#               significand)         (exponent range ~[-160, 280] incl. E8M0)
#   precision   fast-path truncation < 2**-(man_bits-1) relative per block
#               (2**-15 at man_bits=16 vs 2**-23 for FP32); miss region keeps
#               fewer significant bits; flush is the only hard loss.
#
# FREE OPERATING POINT: man_bits = W - scan_bits (24 at W=32, scan_bits=8)
# makes the scheme BIT-IDENTICAL to the reference in every case (the fast path
# truncates to 24 bits exactly like FP32-RTZ; the miss window holds the whole
# value; the flush cannot fire), while still shrinking the CLZ and shifter.
# See test_rnorm_zero_precision_loss().
#
# The signed-mantissa alternative: the model keeps sign + magnitude (matching
# the Term convention). Hardware on a two's-complement datapath can equally
# run CLS on S and keep the window signed -- same arithmetic, it just saves
# the |S| negation. The leading-one/explicit-window logic is unchanged.


@dataclass(frozen=True)
class Reduced:
    """A reduced-precision floating value: sign * m * 2**(exp - P).

    m is an unsigned integer of up to man_bits bits (P == man_bits - 1). In
    the HIT ("fast") path m is fully normalised with the leading one at bit
    man_bits - 1; in the MISS ("fallback") path m may carry leading zeros
    (under-normalised but exact for the bits it keeps). m == 0 is a flushed
    zero that keeps the sign. kept is the number of significant mantissa bits
    actually carried by m.
    """

    sign: int
    m: int
    P: int
    exp: int
    note: str  # "zero" | "fast" | "fallback" | "flush" | "exact"
    kept: int


def reduce_normalize(
    S: int,
    exp2: int,
    *,
    man_bits: int = 16,
    scan_bits: int = 8,
    W: int = 32,
    fallback: str = "flush",
) -> Reduced:
    """Cheap approximate normalise of the exact value S * 2**exp2.

    Returns a Reduced pair that the next block's alignment can consume
    directly (see reduced_to_term). See the RNORM section header for the
    design rationale; the returned pair is exact for the bits it keeps, and
    only the (bounded) low-bit truncation plus the rare flush are lossy.

    man_bits : mantissa width, explicit leading one in the fast path.
    scan_bits: how many top bits of the fixed-width register the cheap CLZ
               examines (the user's "only the top 8 bits").
    W        : fixed datapath width of the accumulator register. Pass the
               design width; the default 32 matches the 32-term E4M3 F=25
               datapath that st_fdpa_fast derives. W=None disables the scan
               entirely and uses a full-width CLZ (reference mode: reduced
               mantissa width only, exact exponent, never flushes).
    fallback : "flush" -- empty miss window reports +/-0 (fast, fixed
               latency); "exact" -- empty miss window pays a full CLZ (slow
               path / second cycle) and never flushes.
    """
    if man_bits < 1 or scan_bits < 1 or fallback not in ("flush", "exact"):
        raise ValueError(f"bad reduce_normalize args: man_bits={man_bits}, "
                         f"scan_bits={scan_bits}, fallback={fallback!r}")
    if S == 0:
        return Reduced(1, 0, man_bits - 1, exp2, "zero", 0)
    sign = 1 if S > 0 else -1
    A = abs(S)
    p = A.bit_length() - 1

    if W is None:  # reference mode: exact exponent, reduced mantissa only
        shift = p - (man_bits - 1)
        m = (A >> shift) & ((1 << man_bits) - 1) if shift >= 0 else A << (-shift)
        return Reduced(sign, m, man_bits - 1, exp2 + p, "exact", man_bits)

    if scan_bits > W or W < man_bits + scan_bits:
        raise ValueError(
            f"need scan_bits <= W and W >= man_bits + scan_bits "
            f"(got W={W}, man_bits={man_bits}, scan_bits={scan_bits})")
    if A.bit_length() > W:  # datapath overflow: saturate at the register top
        A = (1 << W) - 1
        p = W - 1

    # HIT: leading one inside the top scan_bits bits -> exact exponent.
    top = A >> (W - scan_bits)
    if top:
        c8 = top.bit_length()  # 1..scan_bits
        p_est = W - scan_bits + c8 - 1
        assert p_est == p, (p_est, p)  # the scan is exact when it fires
        shift = p - (man_bits - 1)
        m = (A >> shift) & ((1 << man_bits) - 1)
        return Reduced(sign, m, man_bits - 1, exp2 + p, "fast", man_bits)

    # MISS: value shrank below the scanned window (cancellation). Fixed
    # window just below the scan, never above the true leading one.
    j = W - 1 - scan_bits          # top bit of the fixed window
    bottom = j - (man_bits - 1)
    if p < bottom:
        if fallback == "flush":
            return Reduced(sign, 0, man_bits - 1, exp2 + j, "flush", 0)
        shift = p - (man_bits - 1)
        m = (A >> shift) & ((1 << man_bits) - 1) if shift >= 0 else A << (-shift)
        return Reduced(sign, m, man_bits - 1, exp2 + p, "exact", man_bits)
    m = (A >> bottom) & ((1 << man_bits) - 1)
    return Reduced(sign, m, man_bits - 1, exp2 + j, "fallback", m.bit_length())


def reduced_to_fp32_bits(r: Reduced, *, rounding: str = "RTZ",
                         overflow_to_inf: bool = True) -> int:
    """Exact FP32 encode of a reduced value (for man_bits <= 24 nothing is
    lost here: the only precision loss happened inside reduce_normalize)."""
    if r.m == 0:
        return 0x80000000 if r.sign < 0 else 0  # flushed/zero, sign kept
    return exact_to_fp32_bits(r.sign * r.m, r.exp - r.P, rounding=rounding,
                              overflow_to_inf=overflow_to_inf)


def reduced_to_float(r: Reduced) -> float:
    if r.m == 0:
        return -0.0 if r.sign < 0 else 0.0
    return r.sign * r.m * 2.0 ** (r.exp - r.P)


def reduced_to_term(r: Reduced) -> Term:
    """Turn a reduced pair into the Term the next block's alignment consumes."""
    if r.m == 0:
        return Term(ZERO, r.sign)
    return Term(FINITE, r.sign, r.m, r.P, r.exp)


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


def st_fdpa_fast(
    a_bits: Sequence[int],
    b_bits: Sequence[int],
    c_bits: int,
    sfa_bits: int,
    sfb_bits: int,
    *,
    fmt: FpFormat = E4M3,
    F: int = 25,
    man_bits: int = 16,
    scan_bits: int = 8,
    fallback: str = "flush",
    overflow_to_inf: bool = True,
) -> FdpaResult:
    """The fused block with the cheap RNORM normalise (NOT bit-accurate).

    Identical pipeline to st_fdpa up to the exact sum; only the final
    conversion differs: reduce_normalize produces a (sign, m, e) reduced pair
    instead of a full FP32. The block's result bits are the exact FP32 encode
    of that pair (m < 2**24 is lossless), so chaining the bits is identical to
    chaining the reduced pair directly -- the next block's c carries a
    man_bits-bit significand instead of 24 bits.

    Special-value results (NaN / Inf / all-zero) are unchanged from st_fdpa.
    The datapath width W is derived from the design: W = F + 1 +
    ceil(log2(L + 1)) (32 for the default 32-term E4M3 F=25 block).
    """
    r = st_fdpa(a_bits, b_bits, c_bits, sfa_bits, sfb_bits, fmt=fmt, F=F,
                overflow_to_inf=overflow_to_inf)
    if r.S is None:
        return r  # NaN / Inf special paths are identical
    if r.S == 0:
        return FdpaResult(0, 0.0, r.e_max, 0, r.aligned, "rnorm:zero")
    L = len(a_bits)
    W = F + 1 + (L + 1).bit_length()  # fixed register width of this datapath
    red = reduce_normalize(r.S, r.e_max - F, man_bits=man_bits,
                           scan_bits=scan_bits, W=W, fallback=fallback)
    bits = reduced_to_fp32_bits(red, overflow_to_inf=overflow_to_inf)
    return FdpaResult(bits, fp32_bits_to_float(bits), r.e_max, r.S, r.aligned,
                      f"rnorm:{red.note}")


def _round_shift_int(v: int, shift: int, rounding: str) -> int:
    """Round the exact value v * 2**shift to an integer.

    shift >= 0 is an exact left shift. shift < 0 quantises: RTZ truncates
    toward zero, RNE rounds to nearest with ties to even, RNA to nearest with
    ties away from zero. For rounding="RTZ" this is exactly _shift_right_rz.
    """
    if shift >= 0:
        return v << shift
    q, rem = divmod(abs(v), 1 << -shift)
    half = 1 << (-shift - 1)
    if rounding == "RNE" and (rem > half or (rem == half and q & 1)):
        q += 1
    elif rounding == "RNA" and rem >= half:
        q += 1
    return -q if v < 0 else q


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
    rounding: str = "RTZ",
    align_rounding: Optional[str] = None,
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

    rounding: mode of the final FP32 conversion -- "RTZ" (truncate, the
    hardware behaviour), "RNE" (nearest, ties to even), or "RNA" (nearest,
    ties away).
    align_rounding: mode of the per-term alignment on entry to the adder tree
    ("per-term": RTZ wire-truncation, RNE/RNA need a per-input incrementer,
    which real CSA trees avoid). None (default) follows `rounding`.
    "RTZ" at both points reproduces the hardware behaviour. Note the per-term
    choice dominates accuracy for small F (alignment truncation is the main
    error source there), while the output choice matters for F >= 25.
    """
    if rounding not in ("RTZ", "RNE", "RNA"):
        raise ValueError(f"rounding must be RTZ, RNE or RNA, got {rounding!r}")
    align_mode = rounding if align_rounding is None else align_rounding
    if align_mode not in ("RTZ", "RNE", "RNA"):
        raise ValueError(f"align_rounding must be RTZ, RNE or RNA, got {align_mode!r}")
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
        aligned.append(_round_shift_int(t.sign * t.sig, shift, align_mode))

    S = sum(aligned)  # exact: Python ints
    bits = exact_to_fp32_bits(S, e_max - F, overflow_to_inf=overflow_to_inf,
                              rounding=rounding)
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


def mxfp8_dot_fast(
    a_bits: Sequence[int],
    b_bits: Sequence[int],
    sfa_bits: Sequence[int],
    sfb_bits: Sequence[int],
    c_bits: int = 0,
    *,
    fmt: FpFormat = E4M3,
    F: int = 25,
    block: int = 32,
    man_bits: int = 16,
    scan_bits: int = 8,
    fallback: str = "flush",
) -> int:
    """Chain fused dot products with the RNORM normalise between blocks.

    Identical structure to mxfp8_dot, but each block runs st_fdpa_fast, so the
    c fed forward carries a man_bits-bit significand instead of FP32's 24.
    Returns raw FP32 bits. NOT bit-accurate (except at man_bits = W -
    scan_bits, see test_rnorm_zero_precision_loss).
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
        r = st_fdpa_fast(a_bits[lo:hi], b_bits[lo:hi], d, sfa_bits[i],
                         sfb_bits[i], fmt=fmt, F=F, man_bits=man_bits,
                         scan_bits=scan_bits, fallback=fallback)
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


def test_st_fdpa_n_rounding():
    """RNE / RNA / RTZ conversion semantics on exact_to_fp32_bits, plus
    end-to-end tie cases through st_fdpa_n. w=25 RTZ must stay identical to
    the hardware-verified default."""
    import random as _r

    _r.seed(5)
    for _ in range(500):
        K = _r.choice([32, 64, 128])
        nb = K // 32
        a = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(K)]
        b = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(K)]
        sfa = [_r.randrange(255) for _ in range(nb)]
        sfb = [_r.randrange(255) for _ in range(nb)]
        c = _r.randrange(1 << 32)
        assert st_fdpa_n(a, b, c, sfa, sfb).bits == st_fdpa_n(
            a, b, c, sfa, sfb, rounding="RTZ").bits

    to_fp32 = float_to_fp32_bits
    # halfway between 1.0 and 1+2**-23: RNE keeps the even side
    assert exact_to_fp32_bits(2**24 + 1, -24, rounding="RNE") == to_fp32(1.0)
    assert exact_to_fp32_bits(2**24 + 1, -24, rounding="RNA") == to_fp32(1.0 + 2.0**-23)
    assert exact_to_fp32_bits(2**24 + 1, -24, rounding="RTZ") == to_fp32(1.0)
    # halfway just below 2.0: RNE rounds up to the even 2.0 (carry)
    assert exact_to_fp32_bits(2**25 - 1, -24, rounding="RNE") == to_fp32(2.0)
    assert exact_to_fp32_bits(2**25 - 1, -24, rounding="RNA") == to_fp32(2.0)
    assert exact_to_fp32_bits(2**25 - 1, -24, rounding="RTZ") == to_fp32(2.0 - 2.0**-23)
    # negative halfway: RNE keeps even, RNA goes away from zero, RTZ toward
    assert exact_to_fp32_bits(-(2**24 + 1), -24, rounding="RNE") == to_fp32(-1.0)
    assert exact_to_fp32_bits(-(2**24 + 1), -24, rounding="RNA") == to_fp32(-(1.0 + 2.0**-23))
    # subnormal tie at 2.5 * 2**-149
    assert exact_to_fp32_bits(5, -150, rounding="RNE") == 2
    assert exact_to_fp32_bits(5, -150, rounding="RNA") == 3
    # subnormal rounding up into the min normal
    assert exact_to_fp32_bits(2**24 - 1, -150, rounding="RNE") == 0x00800000
    assert exact_to_fp32_bits(2**24 - 1, -150, rounding="RTZ") == 0x007FFFFF
    # end-to-end: shared scale 2**-3 per operand; products 64.0 and 2**-18
    # become 1.0 and 2**-24 -- sum lands exactly halfway at 1+2**-24
    a, b = [0x50, 0x01], [0x50, 0x01]
    sfa = sfb = [e8m0_bits(-3)]
    assert st_fdpa_n(a, b, 0, sfa, sfb, rounding="RNE").bits == to_fp32(1.0)
    assert st_fdpa_n(a, b, 0, sfa, sfb, rounding="RNA").bits == to_fp32(1.0 + 2.0**-23)
    assert st_fdpa_n(a, b, 0, sfa, sfb, rounding="RTZ").bits == to_fp32(1.0)
    # end-to-end: product 1.0 + c = 1+2**-23 sums to 2+2**-23, half an ulp of 2
    c = to_fp32(1.0 + 2.0**-23)
    assert st_fdpa_n([0x38], [0x38], c, [0x7F], [0x7F], rounding="RNE").bits == to_fp32(2.0)
    assert st_fdpa_n([0x38], [0x38], c, [0x7F], [0x7F], rounding="RNA").bits == to_fp32(2.0 + 2.0**-22)
    # per-term alignment rounding (F=17 grid, 64x coarser than the output ulp):
    # the small product 2**-18 sits exactly half a grid unit below 1.0 --
    # RTZ/RNE drop it (tie -> even 0), RNA's tie-away revives it as 1 unit
    a, b = [0x38, 0x01], [0x38, 0x01]
    assert st_fdpa_n(a, b, 0, [0x7F], [0x7F], F=17, rounding="RTZ").bits == to_fp32(1.0)
    assert st_fdpa_n(a, b, 0, [0x7F], [0x7F], F=17, rounding="RNE").bits == to_fp32(1.0)
    assert st_fdpa_n(a, b, 0, [0x7F], [0x7F], F=17, rounding="RNA").bits == to_fp32(1.0 + 2.0**-17)
    print("st_fdpa_n rounding           : RTZ == hardware default; RNE/RNA ties, carry, subnormal, per-term ok")


def test_rnorm_reduce_normalize_units():
    """The three paths of the cheap normalise, by hand.

    W=32, man_bits=16, scan_bits=8: the scan covers bits 31..24, the fixed
    miss window sits at bits 23..8, and values with a leading one below bit 7
    flush.
    """
    # HIT: S = 2**25 + 12345 (leading one at bit 25, inside bits 31..24).
    r = reduce_normalize(2**25 + 12345, -25)
    assert r.note == "fast" and r.kept == 16
    assert r.m == ((2**25 + 12345) >> (25 - 15)) & 0xFFFF == ((2**25 + 12345) >> 10)
    assert r.m.bit_length() == 16  # explicit leading one at bit 15
    assert r.exp == 0  # exp2 + p = -25 + 25
    # value kept: 32780 * 2**-15 = 1.0003662109..., true 1 + 12345*2**-25
    assert reduced_to_float(r) == 32780 * 2.0 ** -15
    # the dropped low bits are exactly the truncation (RTZ): 12345 mod 1024
    assert abs(reduced_to_float(r) - (2**25 + 12345) * 2.0 ** -25) == 57 * 2.0 ** -25

    # MISS (cancellation): leading one at bit 20, below the scan.
    r = reduce_normalize(2**20 + 7, -20)
    assert r.note == "fallback" and r.m == 4096 and r.exp == 3  # 2**12, window top j=23
    assert reduced_to_float(r) == 1.0  # exact for the kept bits: 2**20 * 2**-20

    # FLUSH: leading one below the fixed window (p = 7 < bottom = 8).
    r = reduce_normalize(2**7, -7)
    assert r.note == "flush" and r.m == 0 and reduced_to_float(r) == 0.0
    # fallback="exact" pays a full CLZ instead and keeps the value
    r = reduce_normalize(2**7 + 3, -7, fallback="exact")
    assert r.note == "exact" and r.m == (2**7 + 3) << (15 - 7)
    assert r.exp == 0 and reduced_to_float(r) == (2**7 + 3) * 2.0 ** -7

    # sign and zero handling; reduced pair feeds the next block as a Term
    r = reduce_normalize(-(2**30), 0)
    assert r.sign == -1 and r.note == "fast"
    t = reduced_to_term(r)
    assert t.kind == FINITE and t.sign == -1 and t.P == 15
    t0 = reduced_to_term(reduce_normalize(-(2**7), -7))
    assert t0.kind == ZERO and t0.sign == -1  # flushed zero keeps the sign
    # flush keeps the sign in the FP32 encode too (matches underflow-keeps-sign)
    assert reduced_to_fp32_bits(reduce_normalize(-(2**7), -7)) == 0x80000000

    # W=None reference mode: exact exponent, reduced mantissa width, no flush
    r = reduce_normalize(2**7 + 3, -7, W=None)
    assert r.note == "exact" and r.exp == 0 and reduced_to_float(r) == (2**7 + 3) * 2.0 ** -7
    print("rnorm reduce_normalize       : hit / miss / flush / exact paths ok")


def test_rnorm_block_paths():
    """End-to-end: the miss path is exact, the flush is the one lossy corner,
    and fallback='exact' removes it."""
    # Cancellation: products +24 (12*2) and -24 (8*-3) at e_max=4 cancel to
    # leave only 1*1 = 2**21 in accumulator units -> p=21, a MISS (with the
    # full 32-term datapath W=32 the scan covers bits 31..24), but the fixed
    # window keeps the whole significant part, so the result is exact.
    a = encode_fp8_vector([12.0, 8.0, 1.0] + [0.0] * 29)
    b = encode_fp8_vector([2.0, -3.0, 1.0] + [0.0] * 29)
    exact = st_fdpa(a, b, 0, e8m0_bits(0), e8m0_bits(0))
    fast = st_fdpa_fast(a, b, 0, e8m0_bits(0), e8m0_bits(0))
    assert exact.S == 1 << 21 and exact.e_max == 4, (exact.S, exact.e_max)
    assert fast.note == "rnorm:fallback"
    assert fast.bits == exact.bits and fast.value == 1.0  # miss kept everything

    # Deeper cancellation (E5M2): +24 and -24 cancel, leaving 1*2**-16
    # aligned to 2**5 -> p=5 < 8, so the flush fires and the real 2**-16
    # result is lost. fallback='exact' restores it.
    a5 = encode_fp8_vector([12.0, 8.0, 1.0] + [0.0] * 29, E5M2)
    b5 = encode_fp8_vector([2.0, -3.0, 2.0 ** -16] + [0.0] * 29, E5M2)
    exact5 = st_fdpa(a5, b5, 0, e8m0_bits(0), e8m0_bits(0), fmt=E5M2)
    assert exact5.S == 1 << 5 and exact5.e_max == 4, (exact5.S, exact5.e_max)
    assert exact5.value == 2.0 ** -16
    flushed = st_fdpa_fast(a5, b5, 0, e8m0_bits(0), e8m0_bits(0), fmt=E5M2)
    assert flushed.note == "rnorm:flush" and flushed.value == 0.0
    kept = st_fdpa_fast(a5, b5, 0, e8m0_bits(0), e8m0_bits(0), fmt=E5M2,
                        fallback="exact")
    assert kept.value == 2.0 ** -16 and kept.bits == exact5.bits
    print("rnorm block paths            : miss exact, flush lossy, exact-fallback restores")


def test_rnorm_zero_precision_loss():
    """The free operating point: man_bits = W - scan_bits (24 at W=32) makes
    the cheap normalise bit-identical to the reference model, always."""
    import random as _r

    _r.seed(7)
    for _ in range(3000):
        a = [_r.randrange(256) for _ in range(32)]
        b = [_r.randrange(256) for _ in range(32)]
        sfa, sfb, c = _r.randrange(256), _r.randrange(256), _r.randrange(1 << 32)
        r1 = st_fdpa(a, b, c, sfa, sfb)
        r2 = st_fdpa_fast(a, b, c, sfa, sfb, man_bits=24, scan_bits=8)
        assert r1.bits == r2.bits, (r1.bits, r2.bits, r2.note)
    for _ in range(300):
        K = _r.choice([64, 128])
        nb = K // 32
        a = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(K)]
        b = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(K)]
        sfa = [_r.randrange(255) for _ in range(nb)]
        sfb = [_r.randrange(255) for _ in range(nb)]
        assert mxfp8_dot(a, b, sfa, sfb) == mxfp8_dot_fast(
            a, b, sfa, sfb, man_bits=24, scan_bits=8), (K,)
    print("rnorm zero-precision-loss    : man_bits=24 == reference bit-for-bit")


def test_rnorm_precision_tradeoff():
    """man_bits=16: chained results stay within ~2**-12 relative (i.e. a few
    2**11 FP32 ulps) of the exact model -- per-block RTZ truncation is
    < 2**-15 -- and widening the mantissa shrinks the error monotonically.
    Subnormal results (which FP32 itself quantises on a coarse 2**-149 grid)
    are measured in ulps, where the scheme is within ~2 ulps."""
    import math
    import random as _r

    def ulp_of(x):
        if x == 0.0:
            return 2.0 ** -149
        e = _r_log2(x)
        return 2.0 ** (e - 23) if e >= -126 else 2.0 ** -149

    def chain_stats(man_bits, fallback):
        _r.seed(3)
        worst, mean, n = 0.0, 0.0, 0
        normal_worst = 0.0  # relative error restricted to normal-range results
        for _ in range(1200):
            K = _r.choice([32, 64, 128])
            nb = K // 32
            a = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(K)]
            b = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(K)]
            sfa = [_r.randrange(255) for _ in range(nb)]
            sfb = [_r.randrange(255) for _ in range(nb)]
            ex = fp32_bits_to_float(mxfp8_dot(a, b, sfa, sfb))
            fa = fp32_bits_to_float(mxfp8_dot_fast(a, b, sfa, sfb, man_bits=man_bits,
                                                   fallback=fallback))
            if ex == 0.0 and fa == 0.0:
                continue
            if not math.isfinite(ex) or not math.isfinite(fa):
                continue  # overflow-to-Inf / NaN paths are identical in both models
            err = abs(fa - ex) / ulp_of(ex)
            worst = max(worst, err)
            mean += err
            n += 1
            if abs(ex) >= 2.0 ** -126:
                normal_worst = max(normal_worst, abs(fa - ex) / abs(ex))
        return worst, mean / max(n, 1), normal_worst

    # fallback="exact" never flushes: the error is pure truncation.
    w16, m16, r16 = chain_stats(16, "exact")
    w20, m20, r20 = chain_stats(20, "exact")
    print(f"rnorm precision (16b)        : worst {w16:.0f} ulp, mean {m16:.2f} ulp; "
          f"normal-range rel worst {r16:.3e}")
    print(f"rnorm precision (20b)        : worst {w20:.0f} ulp, mean {m20:.2f} ulp; "
          f"normal-range rel worst {r20:.3e}")
    assert w16 < 2.0 ** 12, w16  # ~2**10 ulps expected across 4-block chains
    assert r16 < 2.0 ** -11, r16  # normal-range relative error ~2**-13
    assert w20 < w16, (w20, w16)  # wider mantissa is never worse
    assert m20 < m16, (m20, m16)

    # fallback="flush" adds only the (rare) cancellation flush: count how
    # often it deviates from the never-flush mode.
    _r.seed(3)
    dev = tot = 0
    for _ in range(1200):
        K = _r.choice([32, 64, 128])
        nb = K // 32
        a = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(K)]
        b = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(K)]
        sfa = [_r.randrange(255) for _ in range(nb)]
        sfb = [_r.randrange(255) for _ in range(nb)]
        f1 = mxfp8_dot_fast(a, b, sfa, sfb, man_bits=16, fallback="flush")
        f2 = mxfp8_dot_fast(a, b, sfa, sfb, man_bits=16, fallback="exact")
        dev += f1 != f2
        tot += 1
    print(f"rnorm flush corner           : {dev}/{tot} chains differ between flush and exact fallback")
    assert dev / tot < 0.01  # random data almost never cancels into the flush zone


def test_rnorm_hit_rate():
    """The 8-bit scan resolves the exponent on the overwhelming majority of
    blocks; the miss path covers cancellation exactly (fewer kept bits)."""
    import random as _r

    _r.seed(9)
    stats = {"fast": 0, "fallback": 0, "flush": 0, "zero": 0}
    for _ in range(5000):
        a = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(32)]
        b = [_r.randrange(0x7F) | (0x80 if _r.random() < 0.5 else 0) for _ in range(32)]
        r = st_fdpa_fast(a, b, _r.randrange(1 << 32), _r.randrange(256), _r.randrange(256))
        if r.note.startswith("rnorm:"):
            stats[r.note[6:]] += 1
    tot = sum(stats.values())
    hit = stats["fast"] / tot
    print(f"rnorm hit rate (8-bit scan)  : fast {stats['fast']} "
          f"({hit:.1%}), fallback {stats['fallback']}, flush {stats['flush']}")
    assert stats["flush"] == 0  # random data never cancels below the window
    assert hit > 0.9  # the dominant term keeps the sum in the scanned window


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
        test_st_fdpa_n_rounding,
        test_rnorm_reduce_normalize_units,
        test_rnorm_block_paths,
        test_rnorm_zero_precision_loss,
        test_rnorm_precision_tradeoff,
        test_rnorm_hit_rate,
    ):
        t()
    print("\nall self-checks passed")


if __name__ == "__main__":
    run_all()
