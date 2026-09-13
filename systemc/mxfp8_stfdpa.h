// =============================================================================
//  mxfp8_stfdpa.h -- bit-accurate SystemC model of the MXFP8 fused dot product
//  of ONE 32-element block, as executed by NVIDIA Blackwell 5th-gen tensor cores
//  (B200 / SM100 and B300 / GB300 / SM103):
//
//      tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X   (UTCQMMA)
//      mma.sync.aligned.m16n8k32...kind::mxf8f6f4.block_scale.scale_vec::1X (QMMA.SF)
//
//  No accumulate:  D = A * B        (the FP32 accumulator C is wired to 0)
//
//  Arithmetic model = ST-FDPA (Scaled Truncated Fused Dot Product Add) from
//  MMA-Sim, arXiv:2511.10909, which is the model that reproduces these units
//  bit-exactly.  Verified parameter row (MMA-Sim Table 4 / nv_ptx/sim.py):
//
//      MXFP8 (e4m3/e5m2) -> FP32,  L_max = 32,  F = 25,  rho = RZ-FP32
//      alignment floor for zero terms:  e_zero = -133
//
//  ---------------------------------------------------------------------------
//  THE ADDER WIDTH (the question this file is written to answer)
//  ---------------------------------------------------------------------------
//  F (= 25, the template parameter below, default 25) is the number of
//  FRACTIONAL bits the fused adder keeps below the alignment point.  The adder
//  itself is wider, because the products are left DENORMALISED (a product
//  significand can be in [2,4) and therefore needs a second integer bit) and
//  because L+1 = 33 terms have to be summed without losing a carry:
//
//      MAG = F + 1 + ceil(log2(L + 1))
//          = 25 + 1 + 6                  (L = 32: 32 products + C)
//          = 32 bits                     <-- datapath register width
//
//  Layout of the 32-bit accumulator register (units of 2**(e_max - F)):
//
//      bit 31 .. 27 : carry headroom for 33 accumulated terms (5..6 bits)
//      bit 26 .. 25 : the 2 integer bits of a denormalised product (s < 3.516)
//      bit 24 ..  0 : the F = 25 fractional bits kept below the alignment point
//
//  Tightness proof (worst case, all-terms-max):
//      max product significand = 15/8 * 15/8 = 225/64 = 3.515625   (e4m3)
//      32 products + |C| < 32*3.515625 + 2 = 114.5 < 128 = 2**7
//      => |S| < 2**7 * 2**F = 2**32  => 32 magnitude bits are exactly enough,
//      and a 33-bit two's-complement register would be needed if you prefer a
//      signed accumulator (|S| keeps 32 significant bits, so bit 31 is data,
//      not sign).  This model therefore carries sign + 32-bit magnitude, which
//      is also what the reference implementation does.
//
//  ---------------------------------------------------------------------------
//  SIGNAL WIDTH RULE used throughout
//  ---------------------------------------------------------------------------
//  * every VALUE-carrying signal is an explicitly sized sc_uint / sc_int:
//        significands  sc_uint<4> / sc_uint<8> / sc_uint<24>
//        exponents     sc_int<10>   (range -282 .. +284 incl. both E8M0 scales)
//        align shift   sc_int<8>    (range -21 .. +39)
//        aligned term  sc_uint<27>  (= F+2, exact bound, see align_term())
//        accumulator   sc_uint<32>  (= MAG) magnitude + 1 sign bit
//        result        sc_uint<32>  (FP32 bit pattern)
//  * control signals (valid / nan / inf / sign) are plain bool, i.e. exactly
//    1 bit; they are not arithmetic signals.
//
//  ---------------------------------------------------------------------------
//  PIPELINE
//  ---------------------------------------------------------------------------
//  stage 1  DECODE + MULTIPLY : exact integer product of the two signed
//                               significands, integer sum of exponents INCLUDING
//                               both E8M0 block-scale exponents (no significand
//                               multiply: E8M0 is exactly 2**e)
//  stage 2  ALIGN             : e_max = max(exp of all terms, E_ZERO if C == 0);
//                               every term is truncated toward zero to F
//                               fractional bits below e_max
//  stage 3  ACCUMULATE        : exact sign-magnitude sum of the aligned terms
//  stage 4  ROUND             : one single conversion of S * 2**(e_max-F) to
//                               FP32 with round-toward-zero (subnormals kept,
//                               overflow -> Inf, exact cancellation -> +0.0)
//
//  Because products are never renormalised, two algebraically identical
//  products can give different results: 12*24 = 288 (significand 2.25, 2nd
//  integer bit used, exp 7) aligns one bit coarser than 18*16 = 288
//  (significand 1.125, exp 8).  See test a7/a8 in the testbench.
// =============================================================================

#ifndef MXFP8_STFDPA_H
#define MXFP8_STFDPA_H

#include <systemc.h>

namespace mxfp8 {

// ---------------------------------------------------------------------------
// FP8 format descriptors (OCP FP8 / MX element formats)
// ---------------------------------------------------------------------------
struct e4m3 {                                     // E4M3: no Inf, S.1111.111 = NaN
    static constexpr int  EXP_BITS = 4;
    static constexpr int  MAN_BITS = 3;
    static constexpr bool HAS_INF  = false;
    static constexpr const char* name = "e4m3";
};
struct e5m2 {                                     // E5M2: IEEE-like Inf/NaN
    static constexpr int  EXP_BITS = 5;
    static constexpr int  MAN_BITS = 2;
    static constexpr bool HAS_INF  = true;
    static constexpr const char* name = "e5m2";
};

constexpr int ceil_log2(unsigned n) { int b = 0; while ((1u << b) < n) ++b; return b; }

// ---------------------------------------------------------------------------
// Datapath widths -- the one place where every bit length is derived
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, int E_ZERO>
struct widths {
    // -- one decoded FP8 operands ------------------------------------------
    static constexpr int SIG_A    = 1 + FA::MAN_BITS;                   // 4
    static constexpr int SIG_B    = 1 + FB::MAN_BITS;                   // 4
    // -- exact product of the two significands ------------------------------
    static constexpr int PROD_SIG = SIG_A + SIG_B;                      // 8   (<= 225)
    static constexpr int PROD_P   = FA::MAN_BITS + FB::MAN_BITS;        // 6
    // -- exponents: exp_a + exp_b + ea8m0 + eb8m0 ---------------------------
    static constexpr int EXP      = 10;                                 // -282..284
    // -- alignment shift count ---------------------------------------------
    //    n = (P - F) - (exp - e_max);  exp and e_max both span -282..+284
    //    => |exp - e_max| <= 566  =>  n in [-587, +547]  => 11 bits suffice,
    //    12 bits are used for headroom.  An 8-bit counter WRAPS here (e.g. a
    //    tiny C far below a large e_max), which silently turns the right shift
    //    that should flush the term into a huge left shift.
    static constexpr int SHIFT    = 12;
    // -- fused adder --------------------------------------------------------
    static constexpr int CARRY    = ceil_log2(L + 1);                   // 6
    static constexpr int MAG      = F + 1 + CARRY;                      // 32  <== ADDER
    static constexpr int ALIGNED  = F + 2;                              // 27  (exact bound)
    // -- constants ----------------------------------------------------------
    static constexpr int ZERO_EXP = E_ZERO;                             // -133
    static constexpr int LATENCY  = 4;
};

// ---------------------------------------------------------------------------
// decoded FP8
// ---------------------------------------------------------------------------
template <int EXP_BITS, int MAN_BITS, bool HAS_INF, int EXP_W>
struct fp8_decoded {
    bool                  nan  = false;
    bool                  inf  = false;
    bool                  zero = false;
    bool                  sign = false;                 // 1 bit
    sc_uint<MAN_BITS + 1> sig;                          // |significand| * 2**-MAN_BITS
    sc_int<EXP_W>         exp;                          // unbiased exponent
};

// ---------------------------------------------------------------------------
// stage 1 output: the exact (significand, exponent) terms of one block
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, int E_ZERO>
struct stage1_t {
    using W = widths<F, L, FA, FB, E_ZERO>;

    bool nan_flag  = false;      // 1 : NaN anywhere -> canonical NaN
    bool inf_pos   = false;      // 1 : +Inf term seen
    bool inf_neg   = false;      // 1 : -Inf term seen

    bool has_term[L];            // 1 : product is finite and non-zero -> is a term
    bool tsign[L];               // 1 : sign of the product
    sc_uint<W::PROD_SIG> tsig[L];// 8 : |sig_a * sig_b|, exact, NOT normalised
    sc_int<W::EXP>       texp[L];// 10: exp_a + exp_b + scale_a + scale_b
    sc_uint<4>           tP[L];  // 4 : fractional bits of tsig

    bool c_valid = false;        // 1 : C is finite and non-zero -> joins as term L
    bool c_zero  = false;        // 1 : C == +/-0  -> contributes the E_ZERO floor
    bool csign   = false;        // 1
    sc_uint<24> csig;            // 24: |significand of C| * 2**-23
    sc_int<W::EXP> cexp;         // 10: unbiased exponent of C
};

// ---------------------------------------------------------------------------
// stage 2 output: everything aligned to e_max
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, int E_ZERO>
struct stage2_t {
    using W = widths<F, L, FA, FB, E_ZERO>;

    bool nan_flag = false;
    bool inf_pos  = false;
    bool inf_neg  = false;

    sc_int<W::EXP> e_max;                 // 10: alignment exponent
    bool           sgn[L + 1];            // 1 : sign of each term (L products + C)
    sc_uint<W::MAG> mag[L + 1];           // 32: |term| in units of 2**(e_max - F)
    bool            present[L + 1];       // 1 : term exists (non-zero product / C)
};

// ---------------------------------------------------------------------------
// stage 3 output: the exact fixed-point accumulator
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, int E_ZERO>
struct stage3_t {
    using W = widths<F, L, FA, FB, E_ZERO>;

    bool nan_flag = false;
    bool inf_pos  = false;
    bool inf_neg  = false;

    bool           sgn = false;           // 1 : sign of S
    sc_uint<W::MAG> mag;                  // 32: |S| in units of 2**(e_max - F)
    sc_int<W::EXP>  e_max;                // 10
};

// ===========================================================================
//  stage 1 : DECODE + MULTIPLY
// ===========================================================================
template <int EXP_BITS, int MAN_BITS, bool HAS_INF, int EXP_W>
inline fp8_decoded<EXP_BITS, MAN_BITS, HAS_INF, EXP_W>
decode_fp8(const sc_uint<1 + EXP_BITS + MAN_BITS>& bits)
{
    constexpr int      BIAS       = (1 << (EXP_BITS - 1)) - 1;
    constexpr int      EMIN       = 1 - BIAS;
    constexpr unsigned EFIELD_MAX = (1u << EXP_BITS) - 1;

    fp8_decoded<EXP_BITS, MAN_BITS, HAS_INF, EXP_W> d;

    d.sign = bits[EXP_BITS + MAN_BITS];

    const unsigned E = bits.range(EXP_BITS + MAN_BITS - 1, MAN_BITS).to_uint();
    const unsigned M = bits.range(MAN_BITS - 1, 0).to_uint();

    if (E == EFIELD_MAX) {
        if (HAS_INF) {                      // E5M2: all-ones exponent is Inf/NaN
            if (M != 0) { d.nan = true; return d; }
            d.inf = true; return d;
        }
        if (M == ((1u << MAN_BITS) - 1)) {  // E4M3: only S.1111.111 is NaN
            d.nan = true; return d;
        }
    }
    if (E == 0) {                           // zero / subnormal (exp = emin)
        if (M == 0) { d.zero = true; return d; }
        d.sig = M; d.exp = EMIN; return d;
    }
    d.sig = (1u << MAN_BITS) | M;            // normal
    d.exp = int(E) - BIAS;
    return d;
}

// E8M0 block scale: value = 2**e, e = bits - 127.  0x00 -> 2**-127 (NOT zero),
// 0xFF -> NaN.  E8M0 has no zero and no Inf.
inline sc_int<10> decode_e8m0(const sc_uint<8>& bits, bool& is_nan)
{
    is_nan = (bits.to_uint() == 0xFFu);
    return sc_int<10>(int(bits.to_uint()) - 127);
}

template <int F, int L, class FA, class FB, int E_ZERO>
inline void stage1_decode_multiply(const sc_uint<8>* a,
                                   const sc_uint<8>* b,
                                   const sc_uint<8>&  scale_a,
                                   const sc_uint<8>&  scale_b,
                                   const sc_uint<32>& c_bits,
                                   stage1_t<F, L, FA, FB, E_ZERO>& out)
{
    using W = widths<F, L, FA, FB, E_ZERO>;

    out.nan_flag = false;
    out.inf_pos  = false;
    out.inf_neg  = false;
    out.c_valid  = false;
    out.c_zero   = false;

    // ---- E8M0 block scales (one pair per 32-element block) ----------------
    bool sa_nan = false, sb_nan = false;
    const sc_int<W::EXP> scale_exp =
        sc_int<W::EXP>(decode_e8m0(scale_a, sa_nan)) + sc_int<W::EXP>(decode_e8m0(scale_b, sb_nan));
    if (sa_nan || sb_nan) out.nan_flag = true;   // NaN scale poisons the block

    // ---- 32 exact products ------------------------------------------------
    for (int k = 0; k < L; ++k) {
        const fp8_decoded<FA::EXP_BITS, FA::MAN_BITS, FA::HAS_INF, W::EXP> da =
            decode_fp8<FA::EXP_BITS, FA::MAN_BITS, FA::HAS_INF, W::EXP>(a[k]);
        const fp8_decoded<FB::EXP_BITS, FB::MAN_BITS, FB::HAS_INF, W::EXP> db =
            decode_fp8<FB::EXP_BITS, FB::MAN_BITS, FB::HAS_INF, W::EXP>(b[k]);

        out.has_term[k] = false;
        out.tsig[k]     = 0;
        out.texp[k]     = 0;
        out.tP[k]       = W::PROD_P;

        if (da.nan || db.nan) { out.nan_flag = true; continue; }

        if (da.inf || db.inf) {                  // Inf handling
            if (da.zero || db.zero) { out.nan_flag = true; continue; }  // Inf * 0
            if (da.sign ^ db.sign) out.inf_neg = true; else out.inf_pos = true;
            continue;
        }
        if (da.zero || db.zero) continue;        // zero product: no term

        // MULTIPLY: significands multiply EXACTLY (uint64 arithmetic, the
        // declared 8-bit result width is exact: 15*15 = 225 < 256) ...
        out.tsig[k] = sc_uint<W::PROD_SIG>(da.sig.to_uint() * db.sig.to_uint());
        // ... and exponents are summed as integers, block scale included.
        out.texp[k] = sc_int<W::EXP>(da.exp) + sc_int<W::EXP>(db.exp) + scale_exp;
        out.tsign[k] = da.sign ^ db.sign;
        out.has_term[k] = true;
    }

    // ---- the FP32 accumulator C (term L) ----------------------------------
    // C = 0 is the "no accumulate" / D = A*B case.  It is NOT a normal zero
    // term: it joins the summation at the floor exponent E_ZERO = -133 (see
    // stage 2), which is what the hardware does and is what makes corners such
    // as scale = 2**-127 behave as measured on B200.
    const bool csign = c_bits[31];
    const unsigned cexp_field = c_bits.range(30, 23).to_uint();
    const unsigned cman       = c_bits.range(22, 0).to_uint();

    if (cexp_field == 0xFFu) {                   // Inf or NaN
        if (cman != 0)      out.nan_flag = true;
        else if (csign)     out.inf_neg  = true;
        else                out.inf_pos  = true;
    } else if (cexp_field == 0) {
        if (cman == 0) {                         // C == +/-0 -> alignment floor
            out.c_zero = true;
        } else {                                 // FP32 subnormal, exp = -126
            out.c_valid = true; out.csign = csign;
            out.csig = cman; out.cexp = sc_int<W::EXP>(-126);
        }
    } else {
        out.c_valid = true; out.csign = csign;
        out.csig = sc_uint<24>((1u << 23) | cman);
        out.cexp = sc_int<W::EXP>(int(cexp_field) - 127);
    }
}

// ===========================================================================
//  stage 2 : ALIGN  (truncate toward zero to F fractional bits below e_max)
// ===========================================================================
//  Exact range of the result: |term|< 3.515625, so the aligned magnitude is
//  < 3.515625 * 2**F < 2**(F+2) = 2**27 -> sc_uint<W::ALIGNED>.  The right
//  shifts below truncate the MAGNITUDE (round toward zero), which is the
//  behaviour of the fused adder;  Python/arithmetic >> would round down
//  instead, hence the explicit magnitude handling.
template <int F, int L, class FA, class FB, int E_ZERO>
inline sc_uint<widths<F, L, FA, FB, E_ZERO>::ALIGNED>
align_term(const sc_uint<widths<F, L, FA, FB, E_ZERO>::PROD_SIG>& sig, // <= 225
           const sc_uint<4>& P,                                        // frac bits
           const sc_int<widths<F, L, FA, FB, E_ZERO>::EXP>& exp,       // term exponent
           const sc_int<widths<F, L, FA, FB, E_ZERO>::EXP>& e_max)     // alignment exp
{
    using W = widths<F, L, FA, FB, E_ZERO>;

    // n = number of bits the significand must move RIGHT
    //   aligned = sig * 2**(exp - e_max + F - P)
    const sc_int<W::SHIFT> n =
        sc_int<W::SHIFT>(int(P.to_uint()) - F) - sc_int<W::SHIFT>(exp - e_max);

    if (n.to_int() > 0) {                       // right shift: truncate (RZ)
        if (n.to_int() >= W::PROD_SIG) return sc_uint<W::ALIGNED>(0);   // shifted out
        return sc_uint<W::ALIGNED>(sc_uint<W::PROD_SIG>(sig) >> n.to_int());
    }
    // left shift, exact.  Unreachable for huge k: we only get here when
    // exp - e_max + F - P > 0, and exp <= e_max, so k = -n <= F - P <= 21.
    const int k = -n.to_int();
    if (k > F - W::PROD_P) return sc_uint<W::ALIGNED>(~0u);             // saturate (never taken)
    return sc_uint<W::ALIGNED>(sc_uint<W::PROD_SIG + W::ALIGNED>(sig) << k);
}

template <int F, int L, class FA, class FB, int E_ZERO>
inline void stage2_align(const stage1_t<F, L, FA, FB, E_ZERO>& in,
                         stage2_t<F, L, FA, FB, E_ZERO>& out)
{
    using W = widths<F, L, FA, FB, E_ZERO>;

    out.nan_flag = in.nan_flag;
    out.inf_pos  = in.inf_pos;
    out.inf_neg  = in.inf_neg;

    // ---- e_max: one single global alignment point for the whole block -----
    bool any = false;
    sc_int<W::EXP> e_max = sc_int<W::EXP>(W::ZERO_EXP);
    for (int k = 0; k < L; ++k) {
        if (!in.has_term[k]) continue;
        e_max = any ? (in.texp[k] > e_max ? in.texp[k] : e_max) : in.texp[k];
        any = true;
    }
    if (in.c_valid) {
        e_max = any ? (in.cexp > e_max ? in.cexp : e_max) : in.cexp;
        any = true;
    }
    if (in.c_zero) {                            // C == 0 -> floor at E_ZERO
        if (e_max < sc_int<W::EXP>(W::ZERO_EXP)) e_max = sc_int<W::EXP>(W::ZERO_EXP);
    }
    out.e_max = e_max;

    // ---- align every term (products) --------------------------------------
    for (int k = 0; k < L; ++k) {
        out.present[k] = in.has_term[k];
        out.sgn[k]     = in.tsign[k];
        out.mag[k]     = 0;
        if (in.has_term[k]) {
            out.mag[k] = sc_uint<W::MAG>(
                align_term<F, L, FA, FB, E_ZERO>(in.tsig[k], in.tP[k], in.texp[k], e_max));
        }
    }

    // ---- align C ----------------------------------------------------------
    out.present[L] = in.c_valid;
    out.sgn[L]     = in.csign;
    out.mag[L]     = 0;
    if (in.c_valid) {
        // C carries a 24-bit significand, so it does not fit the 8-bit product
        // path of align_term(); the identical rule is applied directly.  The
        // shift count MUST be wide: cexp - e_max spans -566..+566, and an
        // 8-bit counter wraps a large right shift into a left shift.
        const sc_int<W::SHIFT> n =
            sc_int<W::SHIFT>(23 - F) - sc_int<W::SHIFT>(in.cexp - e_max);
        if (n.to_int() > 0) {                       // right shift: truncate (RZ)
            out.mag[L] = (n.to_int() >= 24) ? sc_uint<W::MAG>(0)
                                            : sc_uint<W::MAG>(in.csig >> n.to_int());
        } else {                                    // left shift: k <= F - 23 = 2
            const int k = -n.to_int();
            out.mag[L] = (k > F - 23) ? sc_uint<W::MAG>(0)
                                      : sc_uint<W::MAG>(sc_uint<W::MAG + 24>(in.csig) << k);
        }
    }
}

// ===========================================================================
//  stage 3 : ACCUMULATE  (exact sign-magnitude sum)
// ===========================================================================
template <int F, int L, class FA, class FB, int E_ZERO>
inline void stage3_accumulate(const stage2_t<F, L, FA, FB, E_ZERO>& in,
                              stage3_t<F, L, FA, FB, E_ZERO>& out)
{
    using W = widths<F, L, FA, FB, E_ZERO>;

    out.nan_flag = in.nan_flag;
    out.inf_pos  = in.inf_pos;
    out.inf_neg  = in.inf_neg;
    out.e_max    = in.e_max;

    bool           sgn = false;                 // +0 accumulator
    sc_uint<W::MAG> mag = 0;
    for (int k = 0; k <= L; ++k) {
        if (!in.present[k] || in.mag[k] == 0) continue;
        if (sgn == in.sgn[k]) {
            mag = mag + in.mag[k];              // same sign: add
        } else if (mag >= in.mag[k]) {
            mag = mag - in.mag[k];              // opposite: subtract
        } else {
            mag = in.mag[k] - mag;              // opposite, result flips sign
            sgn = in.sgn[k];
        }
    }
    out.sgn = sgn;
    out.mag = mag;
}

// ===========================================================================
//  stage 4 : ROUND  (single RZ-FP32 conversion of S * 2**(e_max - F))
// ===========================================================================
//  Mirrors exact_to_fp32_bits(S, e_max - F) of the reference: round toward
//  zero, FP32 subnormals kept (down to 2**-149), underflow keeps the sign,
//  S == 0 (exact cancellation) is reported as +0.0, finite overflow -> Inf.
template <int MAG>
inline sc_uint<32> round_fp32_rz(bool sign, const sc_uint<MAG>& M, int exp2,
                                 bool overflow_to_inf = true)
{
    static_assert(MAG <= 40, "keep the shift arithmetic in plain int");
    const sc_uint<32> sbit = sign ? sc_uint<32>(0x80000000u) : sc_uint<32>(0u);

    if (M == 0) return sc_uint<32>(0);          // +0.0, sign dropped on cancellation

    int p = -1;                                 // leading-one index (CLZ)
    for (int i = MAG - 1; i >= 0; --i) if (M[i]) { p = i; break; }

    const int E = exp2 + p;                     // unbiased exponent of the value

    if (E < -126) {                             // FP32 subnormal / underflow
        const int shift = -149 - exp2;          // quantise to 2**-149
        sc_uint<24> m = 0;
        if (shift <= 0) {
            const int k = -shift;
            m = (k >= 24) ? sc_uint<24>(0) : sc_uint<24>(M << k);   // exact
        } else {
            m = (shift >= MAG) ? sc_uint<24>(0) : sc_uint<24>(M >> shift);  // RZ
        }
        if (m == 0)                    return sbit;               // underflow, sign kept
        if (m > sc_uint<24>(0x7FFFFFu)) return sbit | 0x00800000u; // rounded into normal
        return sbit | sc_uint<32>(m.to_uint());
    }

    const int shift = p - 23;                   // 24-bit significand (1 implicit + 23)
    const sc_uint<24> m = (shift > 0) ? sc_uint<24>(M >> shift) : sc_uint<24>(M << (-shift));

    if (E > 127)                                // accumulation overflow
        return overflow_to_inf ? (sbit | sc_uint<32>(0x7F800000u))
                               : (sbit | sc_uint<32>(0x7F7FFFFFu));

    return sbit | sc_uint<32>((E + 127) << 23) | sc_uint<32>(m.to_uint() - 0x800000u);
}

template <int F, int L, class FA, class FB, int E_ZERO>
inline sc_uint<32> stage4_round(const stage3_t<F, L, FA, FB, E_ZERO>& in)
{
    if (in.nan_flag)                   return sc_uint<32>(0x7FFFFFFFu);  // canonical NaN
    if (in.inf_pos && in.inf_neg)      return sc_uint<32>(0x7FFFFFFFu);  // +Inf + -Inf
    if (in.inf_pos)                    return sc_uint<32>(0x7F800000u);
    if (in.inf_neg)                    return sc_uint<32>(0xFF800000u);

    return round_fp32_rz(in.sgn, in.mag, int(in.e_max) - F);
}

// ===========================================================================
//  combinational reference entry point (used by the testbench / diff test)
// ===========================================================================
template <int F = 25, int L = 32, class FA = e4m3, class FB = e4m3, int E_ZERO = -133>
inline sc_uint<32> mxfp8_dot(const sc_uint<8>* a,
                             const sc_uint<8>* b,
                             const sc_uint<8>&  scale_a,
                             const sc_uint<8>&  scale_b,
                             const sc_uint<32>& c_bits = sc_uint<32>(0))
{
    stage1_t<F, L, FA, FB, E_ZERO> s1;
    stage2_t<F, L, FA, FB, E_ZERO> s2;
    stage3_t<F, L, FA, FB, E_ZERO> s3;

    stage1_decode_multiply<F, L, FA, FB, E_ZERO>(a, b, scale_a, scale_b, c_bits, s1);
    stage2_align<F, L, FA, FB, E_ZERO>(s1, s2);
    stage3_accumulate<F, L, FA, FB, E_ZERO>(s2, s3);
    return stage4_round<F, L, FA, FB, E_ZERO>(s3);
}

// ===========================================================================
//  the 4-stage pipeline module
// ===========================================================================
template <int F = 25, int L = 32, class FA = e4m3, class FB = e4m3, int E_ZERO = -133>
SC_MODULE(mxfp8_dot_block) {
    using W = widths<F, L, FA, FB, E_ZERO>;

    static_assert(F >= 16 && F <= 32, "sane fused-adder fractional width");
    static_assert(L >= 1 && L <= 32, "one MX block / tcgen05 k <= 32");
    static_assert(W::MAG == F + 1 + ceil_log2(L + 1), "adder width definition");
    static_assert(W::MAG >= W::ALIGNED, "adder must hold one aligned term");
    static_assert(W::PROD_SIG <= W::MAG, "product significand must fit the adder");

    // widths exposed for the testbench / documentation
    static constexpr int ADDER_FRAC_BITS = F;        // 25
    static constexpr int ADDER_WIDTH     = W::MAG;   // 32 (magnitude), +1 sign
    static constexpr int LATENCY         = W::LATENCY;

    // ---- ports ------------------------------------------------------------
    sc_in<bool> clk;
    sc_in<bool> rst_n;

    sc_in<bool>       valid_in;
    sc_in<sc_uint<8>> a_in[L];            // 32 x FP8 element of A (K = 0..31)
    sc_in<sc_uint<8>> b_in[L];            // 32 x FP8 element of B (K = 0..31)
    sc_in<sc_uint<8>> scale_a;            // E8M0 block scale for A (scale_vec::1X)
    sc_in<sc_uint<8>> scale_b;            // E8M0 block scale for B
    sc_in<sc_uint<32>> c_in;              // FP32 accumulator; 0 => D = A*B

    sc_out<bool>       valid_out;
    sc_out<sc_uint<32>> d_out;            // FP32 result of the fused dot product

    // ---- pipeline registers ----------------------------------------------
    stage1_t<F, L, FA, FB, E_ZERO> r1;    bool v1 = false;   // decode + multiply
    stage2_t<F, L, FA, FB, E_ZERO> r2;    bool v2 = false;   // aligned to e_max
    stage3_t<F, L, FA, FB, E_ZERO> r3;    bool v3 = false;   // accumulator S
    // stage 4 (RZ-FP32) is combinational after r3 and drives the outputs

    SC_CTOR(mxfp8_dot_block) {
        // ONE clocked process for the whole pipeline: all four stage registers
        // are updated from local temporaries in a single edge evaluation, so
        // the stage-to-stage order is unambiguous (four separate SC_METHODs
        // writing plain member variables would race at the same edge).
        SC_METHOD(cycle);
            sensitive << clk.pos();
    }

    void cycle() {
        if (!rst_n.read()) {
            v1 = v2 = v3 = false;
            valid_out.write(false);
            d_out.write(sc_uint<32>(0));
            return;
        }

        // ---- stage 4: round the OLD accumulator register -------------------
        const bool        v4_next = v3;
        const sc_uint<32> d4_next = stage4_round<F, L, FA, FB, E_ZERO>(r3);

        // ---- stage 3: accumulate the OLD aligned register ------------------
        stage3_t<F, L, FA, FB, E_ZERO> r3_next = r3;
        if (v2) stage3_accumulate<F, L, FA, FB, E_ZERO>(r2, r3_next);

        // ---- stage 2: align the OLD product register ----------------------
        stage2_t<F, L, FA, FB, E_ZERO> r2_next = r2;
        if (v1) stage2_align<F, L, FA, FB, E_ZERO>(r1, r2_next);

        // ---- stage 1: decode + multiply the input -------------------------
        stage1_t<F, L, FA, FB, E_ZERO> r1_next;
        const bool v1_next = valid_in.read();
        if (v1_next) {
            sc_uint<8> a[L], b[L];
            for (int k = 0; k < L; ++k) { a[k] = a_in[k].read(); b[k] = b_in[k].read(); }
            stage1_decode_multiply<F, L, FA, FB, E_ZERO>(
                a, b, scale_a.read(), scale_b.read(), c_in.read(), r1_next);
        }

        // ---- commit the pipeline registers ---------------------------------
        const bool v2_next = v1;
        const bool v3_next = v2;
        r1 = r1_next; v1 = v1_next;
        r2 = r2_next; v2 = v2_next;
        r3 = r3_next; v3 = v3_next;

        d_out.write(v4_next ? d4_next : sc_uint<32>(0));
        valid_out.write(v4_next);
    }
};

}  // namespace mxfp8

#endif  // MXFP8_STFDPA_H
