// =============================================================================
//  hopper_fp8_wgmma.h
//
//  Bit-accurate SystemC model of the arithmetic datapath that an NVIDIA Hopper
//  (SM90) tensor core uses for the FP8 "wgmma" matrix multiply-accumulate:
//
//      wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3
//      wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3
//      wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2
//      wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e5m2
//
//  It computes ONE output element  d = c + sum_{k=0}^{31} a_k * b_k  the way the
//  hardware does, bit for bit.
//
//  ---------------------------------------------------------------------------
//  VERIFIED PARAMETERS
//  ---------------------------------------------------------------------------
//  From MMA-Sim (Xie et al., arXiv:2511.10909), as implemented in this repo at
//  src/mmasim/mmasim/nv_ptx/sim.py  class WGMMA  (lines 94-118) and
//  src/mmasim/mmasim/arithmetic/fdpa.py  t_fdpa / truncated_fused_sum:
//
//      +---------------------------+-----+-------+----------+----------------+
//      | instruction               |  F  | L_max |  e_zero  | output rounding|
//      +---------------------------+-----+-------+----------+----------------+
//      | wgmma k32 fp8 -> .f32     | 13  |  32   |  -133    | RZ-E8M13       |
//      | wgmma k32 fp8 -> .f16     | 13  |  32   |  -133    | RNE-FP16       |
//      | (Blackwell tcgen05, ref.) | 25  |  32   |  -133    | RZ-FP32        |
//      +---------------------------+-----+-------+----------+----------------+
//
//  F      : fractional bits the fused summation keeps below the alignment point
//  L_max  : max number of PRODUCTS fused into one summation node
//  e_zero : exponent given to terms that are zero
//
//  K = 32 = L_max, so the WHOLE K=32 dot product is ONE fused summation node:
//  there is no chaining and no intermediate FP32 rounding inside the K loop.
//
//  The accumulator C has the SAME TYPE AS THE OUTPUT (isa_wgmma.py: c_type =
//  d_type; the .f16 kernel really does load D as uint16_t): FP32 for the .f32
//  variants, FP16 for the .f16 variants. It is the FC template parameter here,
//  and it changes C's significand width (24 vs 11 bits) and subnormal floor
//  (2^-126 vs 2^-14), not just the final rounding.
//
//  ---------------------------------------------------------------------------
//  THE DATAPATH (4 pipelined stages)
//  ---------------------------------------------------------------------------
//    a_k, b_k (E4M3/E5M2)                                  c (FP32)
//         |                                                    |
//    [1] DECODE + MULTIPLY   sign + INTEGER significand + exponent, then an
//         |                  exact product. Products are left DENORMALISED:
//         |                  sig_p = sig_a*sig_b in [1,4), never renormalised.
//         v
//    [2] ALIGN               ONE alignment exponent ne_max = max(exp+kfrac);
//         |                  every term shifted onto it and TRUNCATED toward
//         |                  zero to F=13 fractional bits. No sticky, no guard.
//         v
//    [3] ACCUMULATE          exact binary adder tree over the 33 terms.
//         |                  No intermediate rounding anywhere.
//         v
//    [4] NORMALISE + ROUND   ONE conversion of S * 2^(ne_max-F):
//                            .f32 -> keep 14 significant bits, truncate (RZ)
//                            .f16 -> keep 11 significant bits, RNE
//
//  Why Hopper FP8 has a poor accuracy reputation, structurally:
//    * A single alignment point for all 33 terms, with TRUNCATION toward zero
//      and no guard/round/sticky. Every term shifted right loses its low bits
//      outright; that is the dominant error source, and it is why F=13 hurts.
//    * The .f32 result keeps only 1+13 = 14 significant bits. "FP32 accumulate"
//      does NOT mean 24 bits of result precision on this instruction.
//    * Rounding is RZ = a systematic bias, not RNE.
//
//  NOTE on e_zero: for this instruction e_zero = -133 is INERT. The smallest
//  product here is 2^-18 (E4M3) / 2^-32 (E5M2), i.e. a normalised exponent of
//  about -12 / -28, always far above -133, so a zero term can never be the
//  maximum and cannot raise ne_max. It becomes live only for the block-scaled
//  variants, where an E8M0 scale can push products below -133.
//
//  NOT MODELLED: the async wgmma issue/commit/wait machinery, shared-memory
//  matrix descriptors, the 64x8 output tile / thread-data mapping, throughput,
//  and the accumulator's round trip through registers between instructions.
//  Those are data-movement concerns; the numerical behaviour of the instruction
//  is entirely captured by the datapath below.
//
//  Built to sit alongside mxfp8_stfdpa.h (the Blackwell/B300 MXFP8 model).
// =============================================================================

#ifndef HOPPER_FP8_WGMMA_H
#define HOPPER_FP8_WGMMA_H

#include <systemc.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

namespace hopper_fp8 {

// ---------------------------------------------------------------------------
// Model parameters (MMA-Sim, arXiv:2511.10909) -- the defaults of this model
// ---------------------------------------------------------------------------
constexpr int DEFAULT_F      = 13;    // fused-summation fractional bits
constexpr int DEFAULT_L      = 32;    // L_max; equals K, so one fused node
constexpr int DEFAULT_E_ZERO = -133;  // exponent handed to zero terms

// ---------------------------------------------------------------------------
// Input element formats
// ---------------------------------------------------------------------------
// E4M3 (OCP FP8, no infinities): bias 7, 3 mantissa bits, subnormal weight
//   2^-9, and S.1111.111 is NaN (all other exp==15 values are normals, max 448).
// E5M2 (IEEE-like FP8): bias 15, 2 mantissa bits, subnormal weight 2^-16,
//   exp==31 gives Inf (m==0) / NaN (m!=0).
struct e4m3 {
    static constexpr int  EXP_BITS = 4;
    static constexpr int  MAN_BITS = 3;
    static constexpr int  BIAS     = 7;
    static constexpr bool HAS_INF  = false;
};
struct e5m2 {
    static constexpr int  EXP_BITS = 5;
    static constexpr int  MAN_BITS = 2;
    static constexpr int  BIAS     = 15;
    static constexpr bool HAS_INF  = true;
};

// Accumulator / output formats. For wgmma the accumulator type IS the output
// type (isa_wgmma.py: c_type = d_type), and the hardware agrees: the .f16
// variant loads D with LOAD_D_M64N8_F16() and a uint16_t* accumulator. That
// makes the C operand materially different between the two variants -- 24-bit
// vs 11-bit significand, and a subnormal floor of 2^-126 vs 2^-14 -- so it is
// a template parameter here, not a runtime flag.
struct f32 {
    static constexpr int  EXP_BITS = 8;
    static constexpr int  MAN_BITS = 23;
    static constexpr int  BIAS     = 127;
    static constexpr int  C_BITS   = 1 + EXP_BITS + MAN_BITS;  // 32
    static constexpr bool IS_FP16  = false;
};
struct f16 {
    static constexpr int  EXP_BITS = 5;
    static constexpr int  MAN_BITS = 10;
    static constexpr int  BIAS     = 15;
    static constexpr int  C_BITS   = 1 + EXP_BITS + MAN_BITS;  // 16
    static constexpr bool IS_FP16  = true;
};

// ---------------------------------------------------------------------------
// Derived signal widths
// ---------------------------------------------------------------------------
constexpr int ceil_log2(int n) {
    int b = 0, v = 1;
    while (v < n) { v <<= 1; ++b; }
    return b;
}

// All arithmetic is done with an explicit sign bit plus an UNSIGNED magnitude
// (sign-magnitude), because alignment truncates toward zero -- a two's-
// complement arithmetic right shift on a negative value would round toward
// -inf instead.
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
struct widths {
    static constexpr int NTERM = L + 1;  // L products + the C term

    // exact product significand: 15*15 = 225, 7*7 = 49.
    // (An operand's own significand is 1 + MAN_BITS bits, but it is never
    // stored as a signal -- products are formed straight from the operands --
    // so it has no named width here.)
    static constexpr int PROD_SIG = 2 * (FA::MAN_BITS + 1);       // 8 / 6
    // The accumulator C joins the same fused summation, so a term slot must be
    // wide enough for BOTH an FP8 product significand and C's significand:
    // 24 bits for a .f32 accumulator, 11 bits for .f16.
    static constexpr int C_SIG    = FC::MAN_BITS + 1;             // 24 / 11
    static constexpr int C_BITS   = FC::C_BITS;                   // 32 / 16
    static constexpr int TERM_SIG = PROD_SIG > C_SIG ? PROD_SIG : C_SIG;
    // Exponent field width. C dominates the range: an FP32 accumulator spans
    // -149 (subnormal) .. +104, an FP16 one -24 .. +5.
    static constexpr int EXP     = 10;
    // Largest possible alignment shift = exp - ne_max + F.
    //
    // Bound: every finite NONZERO term satisfies ne_max >= exp + kfrac, because
    // it contributes its own normalised exponent to the max(). So
    //     shift <= F - kfrac,
    // and the binding case is the term with the smallest kfrac. That is a
    // product (kfrac = 2*MAN_BITS) when the accumulator is wider, giving
    // +7 for E4M3 and +9 for E5M2 at F=13 -- NOT the +11 that a looser
    // argument suggests. (Negative shifts are unbounded in principle; EXP
    // bounds them in practice, and align_term() saturates past -64.)
    static constexpr int SHIFT_MAX =
        (F - 2 * FA::MAN_BITS) > (F - FC::MAN_BITS) ? (F - 2 * FA::MAN_BITS)
                                                    : (F - FC::MAN_BITS);
    // Aligned term in units of 2^(ne_max-F). A product significand is
    // < 2^(MAN_BITS+1), so the product is < 2^(2*MAN_BITS+2) and the aligned
    // value is < 2^(2*MAN_BITS+2) * 2^(F-2*MAN_BITS) = 2^(F+2). F+2 bits then
    // suffice for ANY input format -- for E4M3 at F=13 the worst case is
    // 225 * 2^7 = 28800 < 2^15.
    static constexpr int ALIGNED = F + 2;
    // accumulator magnitude = F + 1 integer bit + ceil(log2(L+1)) carry bits
    static constexpr int MAG     = F + 1 + ceil_log2(L + 1);

    static constexpr long long MAX_PROD_SIG =
        (long long)((1 << (FA::MAN_BITS + 1)) - 1) * (long long)((1 << (FB::MAN_BITS + 1)) - 1);
    static constexpr long long MAX_ALIGNED = MAX_PROD_SIG << (F - 2 * FA::MAN_BITS);
    // Aligned C is bounded by (2^C_SIG - 1) * 2^(F - MAN_BITS), which stays
    // below 2^(F+1) for both accumulators (2^14 for F=13).
    static constexpr long long MAX_C_SIG = (1LL << C_SIG) - 1;
    static constexpr long long MAX_C_ALIGNED =
        (F >= FC::MAN_BITS) ? (MAX_C_SIG << (F - FC::MAN_BITS))
                            : (MAX_C_SIG >> (FC::MAN_BITS - F));
    static constexpr long long MAX_SUM = (long long)L * MAX_ALIGNED + MAX_C_ALIGNED;
    static_assert(MAX_C_ALIGNED < (1LL << (F + 1)), "aligned C out of expected range");

    static_assert(SHIFT_MAX < 48, "positive alignment shift could overflow uint64_t");
    static_assert(SHIFT_MAX >= 0, "products must be able to shift left");
    static_assert(MAX_ALIGNED < (1LL << ALIGNED), "ALIGNED too narrow");
    static_assert(MAX_SUM < (1LL << MAG), "MAG too narrow for L accumulated terms");
    static_assert(MAG >= ALIGNED, "MAG must be at least ALIGNED");
};

// ---------------------------------------------------------------------------
// Decoded operand / term
// ---------------------------------------------------------------------------
enum : uint8_t { CLS_ZERO = 0, CLS_FINITE = 1, CLS_INF = 2, CLS_NAN = 3 };

// value = (-1)^sign * sig * 2^exp.
//   sig   : UNSIGNED INTEGER significand (never a float)
//   exp   : power-of-two weight of the LSB of sig
//   kfrac : number of fractional bits of sig, so the value's normalised
//           exponent -- the one frexp() reports -- is (exp + kfrac)
template <int SIG_W, int EXP_W>
struct decoded_t {
    bool           sign;
    sc_uint<SIG_W> sig;
    sc_int<EXP_W>  exp;
    int            kfrac;
    uint8_t        cls;

    decoded_t() : sign(false), sig(0), exp(0), kfrac(0), cls(CLS_ZERO) {}

    // Widen/narrow from another decoded_t. Used to fold an 8-bit product
    // significand and the 24-bit C significand into the same term type.
    template <int SIG2>
    decoded_t(const decoded_t<SIG2, EXP_W>& o)
        : sign(o.sign), sig(sc_uint<SIG_W>(o.sig.to_uint())), exp(o.exp),
          kfrac(o.kfrac), cls(o.cls) {}
};

template <int SIG_W, int EXP_W>
inline std::ostream& operator<<(std::ostream& os, const decoded_t<SIG_W, EXP_W>& d) {
    const char* c = (d.cls == CLS_ZERO)   ? "zero"
                    : (d.cls == CLS_FINITE) ? "fin"
                    : (d.cls == CLS_INF)    ? "inf"
                                            : "nan";
    os << (d.sign ? "-" : "+") << d.sig.to_uint() << "*2^" << d.exp.to_int() << "[" << c << "]";
    return os;
}

// sc_signal<decoded_t> needs equality and a trace hook.
template <int SIG_W, int EXP_W>
inline bool operator==(const decoded_t<SIG_W, EXP_W>& x, const decoded_t<SIG_W, EXP_W>& y) {
    return x.sign == y.sign && x.sig == y.sig && x.exp == y.exp &&
           x.kfrac == y.kfrac && x.cls == y.cls;
}
template <int SIG_W, int EXP_W>
inline bool operator!=(const decoded_t<SIG_W, EXP_W>& x, const decoded_t<SIG_W, EXP_W>& y) {
    return !(x == y);
}
template <int SIG_W, int EXP_W>
inline void sc_trace(sc_core::sc_trace_file* tf, const decoded_t<SIG_W, EXP_W>& d,
                     const std::string& n) {
    sc_core::sc_trace(tf, d.sig, n + ".sig");
    sc_core::sc_trace(tf, d.exp, n + ".exp");
}

// ---------------------------------------------------------------------------
// [1a] DECODE -- unpack an FP8 element into (sign, integer significand, exp)
// ---------------------------------------------------------------------------
// An INTEGER unpack, not a float conversion: the tensor core never materialises
// a float for the operands, and every FP8 value is an exact integer multiple of
// a power of two.
template <class FMT, int EXP_W>
inline decoded_t<FMT::MAN_BITS + 1, EXP_W> decode_fp8(const sc_uint<8>& bits) {
    typedef decoded_t<FMT::MAN_BITS + 1, EXP_W> D;
    const bool sign = bits[7];
    const int  ef   = (int)((bits.to_uint() >> FMT::MAN_BITS) & ((1u << FMT::EXP_BITS) - 1));
    const int  mf   = (int)(bits.to_uint() & ((1u << FMT::MAN_BITS) - 1));
    D d;
    d.sign  = sign;
    d.kfrac = FMT::MAN_BITS;

    if (ef == 0) {  // zero or subnormal
        if (mf == 0) {
            d.cls = CLS_ZERO;
            d.sig = 0;
            d.exp = 0;
            return d;
        }
        // subnormal: m * 2^(1 - BIAS - MAN_BITS)
        d.cls = CLS_FINITE;
        d.sig = (uint32_t)mf;
        d.exp = sc_int<EXP_W>(1 - FMT::BIAS - FMT::MAN_BITS);
        return d;
    }
    if (ef == (1 << FMT::EXP_BITS) - 1) {  // all-ones exponent field
        if (FMT::HAS_INF && mf == 0) {
            d.cls = CLS_INF;
            return d;
        }
        // E4M3: only S.1111.111 is NaN; E5M2: any mf != 0 is NaN
        if (!FMT::HAS_INF && mf != (1u << FMT::MAN_BITS) - 1) {
            d.cls = CLS_FINITE;
            d.sig = (uint32_t)((1 << FMT::MAN_BITS) | mf);
            d.exp = sc_int<EXP_W>(ef - FMT::BIAS - FMT::MAN_BITS);
            return d;
        }
        d.cls = CLS_NAN;
        return d;
    }
    d.cls = CLS_FINITE;
    d.sig = (uint32_t)((1 << FMT::MAN_BITS) | mf);
    d.exp = sc_int<EXP_W>(ef - FMT::BIAS - FMT::MAN_BITS);
    return d;
}

// ---------------------------------------------------------------------------
// Decode the accumulator input C -- FP32 for a .f32 accumulator, FP16 for .f16
// ---------------------------------------------------------------------------
// Same integer decomposition as the FP8 decode, just at the accumulator's
// width. The subnormal floor is 1 - BIAS, which is 2^-126 for FP32 and 2^-14
// for FP16 -- frexp_and_normalize()'s e_subnormal in the Python model.
template <class FC, int EXP_W>
inline decoded_t<FC::MAN_BITS + 1, EXP_W> decode_accumulator(
    const sc_uint<FC::MAN_BITS + 1 + FC::EXP_BITS>& bits) {
    typedef decoded_t<FC::MAN_BITS + 1, EXP_W> D;
    D d;
    d.sign  = bits[FC::MAN_BITS + FC::EXP_BITS];
    d.kfrac = FC::MAN_BITS;
    const int      ef = (int)((bits.to_uint() >> FC::MAN_BITS) &
                              ((1u << FC::EXP_BITS) - 1));
    const uint32_t mf = bits.to_uint() & ((1u << FC::MAN_BITS) - 1);
    if (ef == (1 << FC::EXP_BITS) - 1) {
        if (mf == 0) { d.cls = CLS_INF; return d; }
        d.cls = CLS_NAN;
        return d;
    }
    if (ef == 0) {
        if (mf == 0) { d.cls = CLS_ZERO; return d; }
        d.cls = CLS_FINITE;                                   // subnormal
        d.sig = mf;
        d.exp = sc_int<EXP_W>(1 - FC::BIAS - FC::MAN_BITS);
        return d;
    }
    d.cls = CLS_FINITE;
    d.sig = (1u << FC::MAN_BITS) | mf;
    d.exp = sc_int<EXP_W>(ef - FC::BIAS - FC::MAN_BITS);
    return d;
}

// ---------------------------------------------------------------------------
// [1b] MULTIPLY -- exact, no normalisation, no rounding
// ---------------------------------------------------------------------------
// sig_a*sig_b is an exact integer (<= 225 for E4M3) and the exponents are
// summed as integers. The product is deliberately left DENORMALISED: its
// significand can be >= 2, which is what needs the second integer bit in the
// alignment register of stage 2.
template <int SIG_W, int EXP_W>
inline decoded_t<2 * SIG_W, EXP_W> multiply(const decoded_t<SIG_W, EXP_W>& x,
                                            const decoded_t<SIG_W, EXP_W>& y) {
    typedef decoded_t<2 * SIG_W, EXP_W> T;
    T t;
    if (x.cls == CLS_NAN || y.cls == CLS_NAN) { t.cls = CLS_NAN; return t; }
    if (x.cls == CLS_INF || y.cls == CLS_INF) {
        // Inf * 0 -> NaN, so the zero check must come second
        if (x.cls == CLS_ZERO || y.cls == CLS_ZERO) { t.cls = CLS_NAN; return t; }
        t.cls  = CLS_INF;
        t.sign = x.sign ^ y.sign;
        return t;
    }
    if (x.cls == CLS_ZERO || y.cls == CLS_ZERO) { t.cls = CLS_ZERO; return t; }
    t.cls  = CLS_FINITE;
    t.sign = x.sign ^ y.sign;
    t.sig  = sc_uint<2 * SIG_W>(x.sig.to_uint() * y.sig.to_uint());
    t.exp  = sc_int<EXP_W>(x.exp.to_int() + y.exp.to_int());
    t.kfrac = x.kfrac + y.kfrac;
    return t;
}

// ---------------------------------------------------------------------------
// [2a] The exponent every term is aligned to
// ---------------------------------------------------------------------------
// normalised exponent = exp + kfrac. Zero terms are pinned to e_zero so they
// still take part in the max(); see the header note on why that is inert here.
template <int SIG_W, int EXP_W>
inline int normalised_exp(const decoded_t<SIG_W, EXP_W>& t, int e_zero) {
    if (t.cls == CLS_ZERO || t.sig.to_uint() == 0) return e_zero;
    if (t.cls != CLS_FINITE) return -1;  // Inf/NaN never reach alignment
    return t.exp.to_int() + t.kfrac;
}

// ---------------------------------------------------------------------------
// [2b] ALIGN one term: shift onto the common quantum, TRUNCATE toward zero
// ---------------------------------------------------------------------------
//   term  = (-1)^sign * sig * 2^exp
//   want  = trunc( sig * 2^(exp - ne_max + F) )
// The quantum of the aligned integer is 2^(ne_max - F). Bits shifted out are
// DISCARDED -- there is no sticky bit and no round-to-nearest.
template <int ALIGNED_W, int SIG_W, int EXP_W>
inline sc_uint<ALIGNED_W> align_term(const decoded_t<SIG_W, EXP_W>& t, int ne_max, int F) {
    if (t.cls != CLS_FINITE || t.sig.to_uint() == 0) return sc_uint<ALIGNED_W>(0);
    const int sh = t.exp.to_int() - ne_max + F;
    uint64_t  v;
    if (sh >= 0) {
        // widths<>::SHIFT_MAX (static_asserted < 48) guarantees no overflow, so
        // there is deliberately no "too big -> 0" fallback here: silently
        // returning 0 for a large shift would contradict the truncation
        // semantics of the negative arm.
        v = (uint64_t)t.sig.to_uint() << sh;
    } else {
        const int r = -sh;
        v = (r >= 64) ? 0ull : ((uint64_t)t.sig.to_uint() >> r);
    }
    return sc_uint<ALIGNED_W>(v);
}

// ---------------------------------------------------------------------------
// [3] ACCUMULATE -- a genuine sign-magnitude binary adder tree
// ---------------------------------------------------------------------------
// Every leaf is already an exact integer on a common quantum and the only
// rounding happens in stage 4, so the tree's ASSOCIATIVITY DOES NOT MATTER:
// any tree shape gives the same answer. The pairwise tree is a structural
// illustration, not a numerical requirement. (Contrast the block-scaled FP4
// path, which pre-adds groups of G=16 products before scaling, where the
// grouping IS numerically significant.)
inline int64_t adder_tree(const int64_t* in, int n) {
    int64_t cur[64];
    for (int i = 0; i < n; ++i) cur[i] = in[i];
    while (n > 1) {
        int m = 0;
        for (int i = 0; i + 1 < n; i += 2) cur[m++] = cur[i] + cur[i + 1];
        if (n & 1) cur[m++] = cur[n - 1];
        n = m;
    }
    return cur[0];
}

// ---------------------------------------------------------------------------
// [4] Special values -- decided once, for the whole dot product
// ---------------------------------------------------------------------------
enum : uint8_t { SP_NONE = 0, SP_NAN = 1, SP_PINF = 2, SP_NINF = 3 };

template <int SIG_W, int EXP_W, int NTERM>
inline uint8_t special_code(const decoded_t<SIG_W, EXP_W>* t) {
    bool nan = false, p_inf = false, n_inf = false;
    for (int i = 0; i < NTERM; ++i) {
        if (t[i].cls == CLS_NAN) {
            nan = true;
        } else if (t[i].cls == CLS_INF) {
            // Track the signs separately: a +1/-1 count would let one +Inf and
            // one -Inf cancel to zero and slip through as a finite result.
            if (t[i].sign) n_inf = true;
            else           p_inf = true;
        }
    }
    if (nan || (p_inf && n_inf)) return SP_NAN;  // +Inf + -Inf -> NaN
    if (p_inf) return SP_PINF;
    if (n_inf) return SP_NINF;
    return SP_NONE;
}

inline uint32_t special_bits(uint8_t code, bool fp16) {
    switch (code) {
        case SP_NAN:  return fp16 ? 0x7FFFu : 0x7FFFFFFFu;  // canonical quiet NaN
        case SP_PINF: return fp16 ? 0x7C00u : 0x7F800000u;
        case SP_NINF: return fp16 ? 0xFC00u : 0xFF800000u;
        default:      return 0u;
    }
}

inline int msb_index(uint64_t a) { return 63 - __builtin_clzll(a); }

// ---------------------------------------------------------------------------
// [4a] FINAL CONVERSION, .f32 output: "RZ-E8M13"
// ---------------------------------------------------------------------------
// The MMA-Sim model converts the exact sum with rho = "RZ-E8M13", implemented
// as: renormalise, TRUNCATE the significand toward zero to 13 fractional bits
// (keep 1+13 = 14 significant bits), then place it in an FP32 container. The
// exponent range is FP32's (-126 subnormal boundary, overflow to Inf), NOT an
// 8-bit range -- the "E8M13" name describes the significand truncation point.
inline sc_uint<32> round_rz_e8m13(int64_t S, int ne_max, int F) {
    if (S == 0) return sc_uint<32>(0x00000000u);  // cancellation / all-zero -> +0.0
    const bool     sign = (S < 0);
    const uint64_t A    = sign ? (uint64_t)(-S) : (uint64_t)S;
    const int      E_V  = ne_max - F;  // value = A * 2^E_V

    const int msb    = msb_index(A);
    const int e_norm = E_V + msb;

    uint64_t At;  // truncated magnitude, value = At * 2^E_res
    int      E_res;
    if (e_norm >= -126) {
        // Clear the bits below the top 14 significant bits. The scale is
        // unchanged, so At still spans msb+1 bit positions -- do not assume At
        // is only 14 bits wide.
        const int drop = msb - 13;
        At    = (drop > 0) ? (A & ~((1ull << drop) - 1)) : A;
        E_res = E_V;
    } else {
        // Subnormal: the model renormalises against -126 first, so the working
        // quantum is 2^(-126-13) = 2^-139.
        //
        // For THIS instruction that arm is lossless: every finite nonzero term
        // has ne >= -126 (an FP32 subnormal C is the floor; FP8 products bottom
        // out near -12 / -28), hence ne_max >= -126 and E_V >= -139, so
        // sh = E_V + 139 >= 0 -- a pure left shift. Nonzero sums therefore
        // never round to zero here and never drop bits; the result is exactly
        // representable as an FP32 subnormal. (Both facts would stop holding
        // for a wider exponent range, e.g. with E8M0 block scales.)
        const int sh = E_V + 139;
        At    = (sh >= 0) ? (A << sh) : ((sh <= -64) ? 0ull : (A >> (-sh)));
        E_res = -139;
    }
    if (At == 0) return sc_uint<32>(sign ? 0x80000000u : 0x00000000u);

    const int msb_t = msb_index(At);
    const int e_unb = E_res + msb_t;
    // The 13 fractional bits are bits [msb_t-1 .. msb_t-13] of At.
    const uint64_t frac13 = (msb_t >= 13) ? ((At >> (msb_t - 13)) & 0x1FFFull)
                                          : ((At << (13 - msb_t)) & 0x1FFFull);

    if (e_unb > 127) return sc_uint<32>(sign ? 0xFF800000u : 0x7F800000u);

    if (e_unb >= -126) {
        const uint32_t bits = ((uint32_t)(e_unb + 127) << 23) | (uint32_t)(frac13 << 10);
        return sc_uint<32>(bits | (sign ? 0x80000000u : 0u));
    }
    // FP32 subnormal output: rescale the quantum 2^E_res to 2^-149.
    const int sh2 = E_res + 149;
    const uint64_t m = (sh2 >= 0) ? (At << sh2) : ((sh2 <= -64) ? 0ull : (At >> (-sh2)));
    return sc_uint<32>((uint32_t)m | (sign ? 0x80000000u : 0u));
}

// ---------------------------------------------------------------------------
// [4b] FINAL CONVERSION, .f16 output: "RNE-FP16"
// ---------------------------------------------------------------------------
// Round the significand to 10 fractional bits (1+10 = 11 significant bits) with
// round-to-nearest-EVEN, then place it in FP16. Note the asymmetry with the
// .f32 case: the .f16-output variant of the same instruction rounds to nearest,
// while the .f32-output variant truncates toward zero.
inline sc_uint<16> round_rne_fp16(int64_t S, int ne_max, int F) {
    if (S == 0) return sc_uint<16>(0x0000u);
    const bool     sign = (S < 0);
    const uint64_t A    = sign ? (uint64_t)(-S) : (uint64_t)S;
    const int      E_V  = ne_max - F;

    const int msb    = msb_index(A);
    const int e_norm = E_V + msb;

    uint64_t Q;  // rounded significand, value = Q * 2^E_res
    int      E_res;
    if (e_norm >= -14) {
        const int drop = msb - 10;  // bits to discard to keep 11 significant bits
        if (drop > 0) {
            Q = A >> drop;
            const uint64_t rem  = A & ((1ull << drop) - 1);
            const uint64_t half = 1ull << (drop - 1);
            if (rem > half || (rem == half && (Q & 1))) Q++;  // RNE
        } else {
            Q = A << (-drop);  // fewer than 11 significant bits so far
        }
        // Q = A * 2^-drop, so the scale must absorb the shift in BOTH directions.
        E_res = E_V + drop;
        if (Q == (1ull << 11)) {  // rounding carried out of the significand
            Q >>= 1;
            E_res += 1;
        }
    } else {
        // Subnormal: renormalise against -14, so the quantum becomes 2^-24 and
        // the rounding applies to the real value A*2^(E_V+24). The quantum is
        // fixed here, so E_res stays -24 in both shift directions.
        const int sh = E_V + 24;
        if (sh >= 0) {
            Q = A << sh;
        } else {
            const int d = -sh;
            if (d >= 64) {
                Q = 0;
            } else {
                Q = A >> d;
                const uint64_t rem  = A & ((1ull << d) - 1);
                const uint64_t half = 1ull << (d - 1);
                if (rem > half || (rem == half && (Q & 1))) Q++;  // RNE
            }
        }
        E_res = -24;
    }
    if (Q == 0) return sc_uint<16>(sign ? 0x8000u : 0x0000u);

    const int msb_q = msb_index(Q);
    const int e_unb = E_res + msb_q;

    if (e_unb > 15) return sc_uint<16>(sign ? 0xFC00u : 0x7C00u);

    if (e_unb >= -14) {
        const uint64_t frac = (msb_q >= 10) ? ((Q >> (msb_q - 10)) & 0x3FFull)
                                            : ((Q << (10 - msb_q)) & 0x3FFull);
        const uint16_t bits = (uint16_t)(((e_unb + 15) << 10) | (int)frac);
        return sc_uint<16>(bits | (sign ? 0x8000u : 0u));
    }
    const int sh2 = E_res + 24;
    const uint64_t m = (sh2 >= 0) ? (Q << sh2) : 0ull;
    return sc_uint<16>((uint16_t)m | (sign ? 0x8000u : 0u));
}

// ---------------------------------------------------------------------------
// The whole datapath, combinationally
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline sc_uint<32> hopper_fp8_dot(
    const sc_uint<8>* a_bits,
    const sc_uint<8>* b_bits,
    const sc_uint<widths<F, L, FA, FB, FC, E_ZERO>::C_BITS>& c_bits) {
    typedef widths<F, L, FA, FB, FC, E_ZERO> W;
    typedef decoded_t<W::TERM_SIG, W::EXP> term_t;

    term_t t[W::NTERM];
    for (int k = 0; k < L; ++k) {
        const decoded_t<FA::MAN_BITS + 1, W::EXP> da = decode_fp8<FA, W::EXP>(a_bits[k]);
        const decoded_t<FB::MAN_BITS + 1, W::EXP> db = decode_fp8<FB, W::EXP>(b_bits[k]);
        t[k] = multiply<FA::MAN_BITS + 1, W::EXP>(da, db);
    }
    // The C term joins the SAME fused summation as the products.
    t[L] = decode_accumulator<FC, W::EXP>(c_bits);

    const uint8_t sp = special_code<W::TERM_SIG, W::EXP, W::NTERM>(t);
    if (sp != SP_NONE)
        return sc_uint<32>(special_bits(sp, FC::IS_FP16));

    int ne_max = normalised_exp<W::TERM_SIG, W::EXP>(t[0], E_ZERO);
    for (int i = 1; i < W::NTERM; ++i)
        ne_max = std::max(ne_max, normalised_exp<W::TERM_SIG, W::EXP>(t[i], E_ZERO));

    int64_t mag[W::NTERM];
    for (int i = 0; i < W::NTERM; ++i) {
        const sc_uint<W::ALIGNED> m = align_term<W::ALIGNED, W::TERM_SIG, W::EXP>(t[i], ne_max, F);
        mag[i] = t[i].sign ? -(int64_t)m.to_uint() : (int64_t)m.to_uint();
    }
    const int64_t S = adder_tree(mag, W::NTERM);

    if (FC::IS_FP16) return sc_uint<32>(round_rne_fp16(S, ne_max, F).to_uint());
    return round_rz_e8m13(S, ne_max, F);
}

// ---------------------------------------------------------------------------
// Pipeline registers
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
struct stage1_t {  // decoded + multiplied terms
    typedef widths<F, L, FA, FB, FC, E_ZERO> W;
    decoded_t<W::TERM_SIG, W::EXP> term[W::NTERM];  // 24: exact |sig_a*sig_b| (8b) or C (24b)
                                                    //     term[L] is the FP32 C operand
    uint8_t special = SP_NONE;                      // 2 : SP_NONE/NAN/PINF/NINF
};

template <int F, int L, class FA, class FB, class FC, int E_ZERO>
struct stage2_t {  // aligned magnitudes and the shared alignment exponent
    typedef widths<F, L, FA, FB, FC, E_ZERO> W;
    sc_uint<W::ALIGNED> mag[W::NTERM];  // 15: |term| in units of 2^(ne_max-F)
    bool                neg[W::NTERM];  // 1 : sign kept separately (sign-magnitude)
    sc_int<W::EXP>      ne_max;         // 10: the one alignment exponent
    uint8_t             special = SP_NONE;
    stage2_t() : ne_max(0) {
        for (int i = 0; i < W::NTERM; ++i) { mag[i] = 0; neg[i] = false; }
    }
};

template <int F, int L, class FA, class FB, class FC, int E_ZERO>
struct stage3_t {  // accumulated exact sum
    typedef widths<F, L, FA, FB, FC, E_ZERO> W;
    sc_int<W::MAG + 1> S;        // 21: exact sum, sign + magnitude
    sc_int<W::EXP>     ne_max;   // 10
    uint8_t            special = SP_NONE;
    stage3_t() : S(0), ne_max(0) {}
};

// sc_signal<T> needs equality, a trace hook and operator<< for every register type.
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline std::ostream& operator<<(std::ostream& os, const stage1_t<F, L, FA, FB, FC, E_ZERO>& v) {
    os << "stage1(sp=" << (int)v.special << ")";
    return os;
}
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline std::ostream& operator<<(std::ostream& os, const stage2_t<F, L, FA, FB, FC, E_ZERO>& v) {
    os << "stage2(ne_max=" << v.ne_max << ", sp=" << (int)v.special << ")";
    return os;
}
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline std::ostream& operator<<(std::ostream& os, const stage3_t<F, L, FA, FB, FC, E_ZERO>& v) {
    os << "stage3(S=" << v.S << ", ne_max=" << v.ne_max << ", sp=" << (int)v.special << ")";
    return os;
}

template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline bool operator==(const stage1_t<F, L, FA, FB, FC, E_ZERO>& x,
                       const stage1_t<F, L, FA, FB, FC, E_ZERO>& y) {
    if (x.special != y.special) return false;
    for (int i = 0; i < widths<F, L, FA, FB, FC, E_ZERO>::NTERM; ++i)
        if (!(x.term[i] == y.term[i])) return false;
    return true;
}
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline void sc_trace(sc_core::sc_trace_file* tf, const stage1_t<F, L, FA, FB, FC, E_ZERO>& v,
                     const std::string& n) {
    sc_core::sc_trace(tf, v.special, n + ".special");
}
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline void sc_trace(sc_core::sc_trace_file* tf, const stage2_t<F, L, FA, FB, FC, E_ZERO>& v,
                     const std::string& n) {
    sc_core::sc_trace(tf, v.ne_max, n + ".ne_max");
    sc_core::sc_trace(tf, v.special, n + ".special");
}
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline bool operator==(const stage2_t<F, L, FA, FB, FC, E_ZERO>& x,
                       const stage2_t<F, L, FA, FB, FC, E_ZERO>& y) {
    if (x.ne_max != y.ne_max || x.special != y.special) return false;
    for (int i = 0; i < widths<F, L, FA, FB, FC, E_ZERO>::NTERM; ++i)
        if (x.mag[i] != y.mag[i] || x.neg[i] != y.neg[i]) return false;
    return true;
}
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline bool operator==(const stage3_t<F, L, FA, FB, FC, E_ZERO>& x,
                       const stage3_t<F, L, FA, FB, FC, E_ZERO>& y) {
    return x.S == y.S && x.ne_max == y.ne_max && x.special == y.special;
}
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline void sc_trace(sc_core::sc_trace_file* tf, const stage3_t<F, L, FA, FB, FC, E_ZERO>& v,
                     const std::string& n) {
    sc_core::sc_trace(tf, v.S, n + ".S");
    sc_core::sc_trace(tf, v.ne_max, n + ".ne_max");
    sc_core::sc_trace(tf, v.special, n + ".special");
}

// ---------------------------------------------------------------------------
// Stage functions (used by both the RTL stages and the testbench)
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline void stage1_decode_multiply(
    const sc_uint<8>* a_bits,
    const sc_uint<8>* b_bits,
    const sc_uint<widths<F, L, FA, FB, FC, E_ZERO>::C_BITS>& c_bits,
    stage1_t<F, L, FA, FB, FC, E_ZERO>& out) {
    typedef widths<F, L, FA, FB, FC, E_ZERO> W;
    for (int k = 0; k < L; ++k) {
        const decoded_t<FA::MAN_BITS + 1, W::EXP> da = decode_fp8<FA, W::EXP>(a_bits[k]);
        const decoded_t<FB::MAN_BITS + 1, W::EXP> db = decode_fp8<FB, W::EXP>(b_bits[k]);
        out.term[k] = multiply<FA::MAN_BITS + 1, W::EXP>(da, db);
    }
    out.term[L] = decode_accumulator<FC, W::EXP>(c_bits);
    out.special = special_code<W::TERM_SIG, W::EXP, W::NTERM>(out.term);
}

template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline void stage2_align(const stage1_t<F, L, FA, FB, FC, E_ZERO>& in,
                         stage2_t<F, L, FA, FB, FC, E_ZERO>& out) {
    typedef widths<F, L, FA, FB, FC, E_ZERO> W;
    int ne = E_ZERO;
    for (int i = 0; i < W::NTERM; ++i)
        ne = std::max(ne, normalised_exp<W::TERM_SIG, W::EXP>(in.term[i], E_ZERO));
    out.ne_max = sc_int<W::EXP>(ne);
    out.special = in.special;
    for (int i = 0; i < W::NTERM; ++i) {
        out.mag[i] = align_term<W::ALIGNED, W::TERM_SIG, W::EXP>(in.term[i], ne, F);
        out.neg[i] = in.term[i].sign && (in.term[i].cls == CLS_FINITE) &&
                     (in.term[i].sig.to_uint() != 0);
    }
}

template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline void stage3_accumulate(const stage2_t<F, L, FA, FB, FC, E_ZERO>& in,
                              stage3_t<F, L, FA, FB, FC, E_ZERO>& out) {
    typedef widths<F, L, FA, FB, FC, E_ZERO> W;
    int64_t v[W::NTERM];
    for (int i = 0; i < W::NTERM; ++i) {
        const int64_t m = (int64_t)in.mag[i].to_uint();
        v[i] = in.neg[i] ? -m : m;
    }
    out.S       = sc_int<W::MAG + 1>(adder_tree(v, W::NTERM));
    out.ne_max  = in.ne_max;
    out.special = in.special;
}

template <int F, int L, class FA, class FB, class FC, int E_ZERO>
inline sc_uint<32> stage4_normalise_round(const stage3_t<F, L, FA, FB, FC, E_ZERO>& in) {
    if (in.special != SP_NONE) return sc_uint<32>(special_bits(in.special, FC::IS_FP16));
    const int64_t S  = in.S.to_int64();
    const int     ne = in.ne_max.to_int();
    if (FC::IS_FP16) return sc_uint<32>(round_rne_fp16(S, ne, F).to_uint());
    return round_rz_e8m13(S, ne, F);
}

// ===========================================================================
//  SystemC modules
// ===========================================================================

// ---------------------------------------------------------------------------
// Stage 1: DECODE + MULTIPLY
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
SC_MODULE(stage1_module) {
    typedef widths<F, L, FA, FB, FC, E_ZERO> W;
    sc_in<bool>         clk, rst_n, in_valid;
    sc_in<sc_uint<8> >  a[L];
    sc_in<sc_uint<8> >  b[L];
    sc_in<sc_uint<W::C_BITS> > c;
    sc_out<bool>        out_valid;
    sc_out<stage1_t<F, L, FA, FB, FC, E_ZERO> > out;

    SC_CTOR(stage1_module) { SC_CTHREAD(run, clk.pos()); }

    void run() {
        stage1_t<F, L, FA, FB, FC, E_ZERO> z;
        z.special = SP_NONE;
        out_valid.write(false);
        out.write(z);
        wait();
        while (true) {
            if (!rst_n.read()) {
                out_valid.write(false);
                out.write(z);
            } else {
                out_valid.write(in_valid.read());
                sc_uint<8> ab[L], bb[L];
                for (int k = 0; k < L; ++k) { ab[k] = a[k].read(); bb[k] = b[k].read(); }
                stage1_t<F, L, FA, FB, FC, E_ZERO> r;
                stage1_decode_multiply<F, L, FA, FB, FC, E_ZERO>(ab, bb, c.read(), r);
                out.write(r);
            }
            wait();
        }
    }
};

// ---------------------------------------------------------------------------
// Stage 2: ALIGN (find ne_max, then shift + truncate every term)
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
SC_MODULE(stage2_module) {
    sc_in<bool> clk, rst_n, in_valid;
    sc_in<stage1_t<F, L, FA, FB, FC, E_ZERO> > in;
    sc_out<bool> out_valid;
    sc_out<stage2_t<F, L, FA, FB, FC, E_ZERO> > out;

    SC_CTOR(stage2_module) { SC_CTHREAD(run, clk.pos()); }

    void run() {
        stage2_t<F, L, FA, FB, FC, E_ZERO> z;
        z.ne_max = 0;
        z.special = SP_NONE;
        out_valid.write(false);
        out.write(z);
        wait();
        while (true) {
            if (!rst_n.read()) {
                out_valid.write(false);
                out.write(z);
            } else {
                out_valid.write(in_valid.read());
                stage2_t<F, L, FA, FB, FC, E_ZERO> r;
                stage2_align<F, L, FA, FB, FC, E_ZERO>(in.read(), r);
                out.write(r);
            }
            wait();
        }
    }
};

// ---------------------------------------------------------------------------
// Stage 3: ACCUMULATE (exact adder tree, no rounding)
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
SC_MODULE(stage3_module) {
    sc_in<bool> clk, rst_n, in_valid;
    sc_in<stage2_t<F, L, FA, FB, FC, E_ZERO> > in;
    sc_out<bool> out_valid;
    sc_out<stage3_t<F, L, FA, FB, FC, E_ZERO> > out;

    SC_CTOR(stage3_module) { SC_CTHREAD(run, clk.pos()); }

    void run() {
        stage3_t<F, L, FA, FB, FC, E_ZERO> z;
        z.S = 0;
        z.ne_max = 0;
        z.special = SP_NONE;
        out_valid.write(false);
        out.write(z);
        wait();
        while (true) {
            if (!rst_n.read()) {
                out_valid.write(false);
                out.write(z);
            } else {
                out_valid.write(in_valid.read());
                stage3_t<F, L, FA, FB, FC, E_ZERO> r;
                stage3_accumulate<F, L, FA, FB, FC, E_ZERO>(in.read(), r);
                out.write(r);
            }
            wait();
        }
    }
};

// ---------------------------------------------------------------------------
// Stage 4: NORMALISE + ROUND + PACK
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
SC_MODULE(stage4_module) {
    sc_in<bool> clk, rst_n, in_valid;
    sc_in<stage3_t<F, L, FA, FB, FC, E_ZERO> > in;
    sc_out<bool> out_valid;
    sc_out<sc_uint<FC::C_BITS> > d;  // FP32 pattern for .f32, FP16 pattern for .f16

    SC_CTOR(stage4_module) { SC_CTHREAD(run, clk.pos()); }

    void run() {
        out_valid.write(false);
        d.write(0);
        wait();
        while (true) {
            if (!rst_n.read()) {
                out_valid.write(false);
                d.write(0);
            } else {
                out_valid.write(in_valid.read());
                d.write(sc_uint<FC::C_BITS>(
                    stage4_normalise_round<F, L, FA, FB, FC, E_ZERO>(in.read()).to_uint()));
            }
            wait();
        }
    }
};

// ---------------------------------------------------------------------------
// Top level: the Hopper FP8 wgmma dot-product-accumulate datapath
// ---------------------------------------------------------------------------
template <int F, int L, class FA, class FB, class FC, int E_ZERO>
SC_MODULE(hopper_fp8_wgmma) {
    sc_in<bool>         clk, rst_n, in_valid;
    sc_in<sc_uint<8> >  a[L];
    sc_in<sc_uint<8> >  b[L];
    sc_in<sc_uint<widths<F, L, FA, FB, FC, E_ZERO>::C_BITS> > c;
    sc_out<bool>        out_valid;
    sc_out<sc_uint<FC::C_BITS> > d;  // FP32 pattern for .f32, FP16 pattern for .f16

    sc_signal<bool> v1, v2, v3;
    sc_signal<stage1_t<F, L, FA, FB, FC, E_ZERO> > s1;
    sc_signal<stage2_t<F, L, FA, FB, FC, E_ZERO> > s2;
    sc_signal<stage3_t<F, L, FA, FB, FC, E_ZERO> > s3;

    stage1_module<F, L, FA, FB, FC, E_ZERO>* u1;
    stage2_module<F, L, FA, FB, FC, E_ZERO>* u2;
    stage3_module<F, L, FA, FB, FC, E_ZERO>* u3;
    stage4_module<F, L, FA, FB, FC, E_ZERO>* u4;

    SC_CTOR(hopper_fp8_wgmma) {
        u1 = new stage1_module<F, L, FA, FB, FC, E_ZERO>("u1");
        u1->clk(clk); u1->rst_n(rst_n); u1->in_valid(in_valid);
        for (int k = 0; k < L; ++k) { u1->a[k](a[k]); u1->b[k](b[k]); }
        u1->c(c);
        u1->out_valid(v1);
        u1->out(s1);

        u2 = new stage2_module<F, L, FA, FB, FC, E_ZERO>("u2");
        u2->clk(clk); u2->rst_n(rst_n); u2->in_valid(v1);
        u2->in(s1);
        u2->out_valid(v2);
        u2->out(s2);

        u3 = new stage3_module<F, L, FA, FB, FC, E_ZERO>("u3");
        u3->clk(clk); u3->rst_n(rst_n); u3->in_valid(v2);
        u3->in(s2);
        u3->out_valid(v3);
        u3->out(s3);

        u4 = new stage4_module<F, L, FA, FB, FC, E_ZERO>("u4");
        u4->clk(clk); u4->rst_n(rst_n); u4->in_valid(v3);
        u4->in(s3);
        u4->out_valid(out_valid);
        u4->d(d);
    }

    ~hopper_fp8_wgmma() { delete u1; delete u2; delete u3; delete u4; }
};

}  // namespace hopper_fp8

#endif  // HOPPER_FP8_WGMMA_H
