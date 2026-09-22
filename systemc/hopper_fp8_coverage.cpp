// =============================================================================
//  hopper_fp8_coverage.cpp
//
//  Branch-coverage probe for the Hopper FP8 wgmma datapath, run over a dump
//  produced by hopper_fp8_wgmma_tb --dump.
//
//  It exists because of a real failure mode: the testbench's "--finite"
//  stimulus was silently 100% NaN propagation, so 20000 "finite" E5M2
//  comparisons exercised no numeric arithmetic at all and still printed PASS.
//  Bit-exactness on blocks that never reach the datapath is worthless, and
//  nothing in the test output made that visible. This probe makes it visible.
//
//  It reproduces the datapath with the SAME header functions the RTL stages
//  use, then classifies which arm of the final conversion each block takes.
//
//      g++ -std=c++17 -I$(SYSTEMC_HOME)/include -I. hopper_fp8_coverage.cpp \
//          -o hopper_fp8_coverage -L$(SYSTEMC_HOME)/lib -lsystemc \
//          -Wl,-rpath,$(SYSTEMC_HOME)/lib
//      ./hopper_fp8_coverage hopper_dump.txt hopper_dump.txt.finite
//
//  Gates worth watching (all were 0 at some point during development):
//    * fp32 / fp16 SUBnormal output arms -- need deliberately tiny values, so
//      random stimulus essentially never reaches them; they are pinned by
//      golden vectors, not by the dump.
//    * the fp16 RNE carry out of the significand.
//    * S == 0 (all-zero or exact cancellation).
//    * "numeric blocks", i.e. blocks that survive the Inf/NaN check at all.
// =============================================================================

#include "hopper_fp8_wgmma.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace hopper_fp8;

namespace {

struct Counts {
    long n = 0;             // blocks that reach the numeric (non-special) path
    long total = 0;         // blocks seen
    long special = 0;       // blocks collapsed by a NaN/Inf operand
    long s_zero = 0;
    long rz32_normal = 0, rz32_truncating = 0, rz32_subnormal = 0, rz32_overflow = 0;
    long rne16_normal = 0, rne16_normal_rounding = 0, rne16_carry = 0, rne16_subnormal = 0,
         rne16_overflow = 0;
};

template <class FMT, class FC>
void classify(const uint8_t* a, const uint8_t* b, uint32_t cbits, Counts& C) {
    typedef widths<DEFAULT_F, DEFAULT_L, FMT, FMT, FC, DEFAULT_E_ZERO> W;
    typedef decoded_t<W::TERM_SIG, W::EXP> term_t;
    const int F = DEFAULT_F, L = DEFAULT_L;

    term_t t[W::NTERM];
    for (int k = 0; k < L; ++k) {
        const decoded_t<FMT::MAN_BITS + 1, W::EXP> da = decode_fp8<FMT, W::EXP>(sc_uint<8>(a[k]));
        const decoded_t<FMT::MAN_BITS + 1, W::EXP> db = decode_fp8<FMT, W::EXP>(sc_uint<8>(b[k]));
        t[k] = multiply<FMT::MAN_BITS + 1, W::EXP>(da, db);
    }
    t[L] = decode_accumulator<FC, W::EXP>(sc_uint<W::C_BITS>(cbits));

    if (special_code<W::TERM_SIG, W::EXP, W::NTERM>(t) != SP_NONE) {
        C.special++;
        return;
    }
    int ne = DEFAULT_E_ZERO;
    for (int i = 0; i < W::NTERM; ++i)
        ne = std::max(ne, normalised_exp<W::TERM_SIG, W::EXP>(t[i], DEFAULT_E_ZERO));

    int64_t v[W::NTERM];
    for (int i = 0; i < W::NTERM; ++i) {
        const int64_t m =
            (int64_t)align_term<W::ALIGNED, W::TERM_SIG, W::EXP>(t[i], ne, F).to_uint();
        const bool neg = t[i].sign && t[i].cls == CLS_FINITE && t[i].sig.to_uint() != 0;
        v[i] = neg ? -m : m;
    }
    const int64_t S = adder_tree(v, W::NTERM);
    C.n++;
    if (S == 0) {
        C.s_zero++;
        return;
    }
    const uint64_t A    = S < 0 ? (uint64_t)(-S) : (uint64_t)S;
    const int      E_V  = ne - F;
    const int      msb  = 63 - __builtin_clzll(A);
    const int      e_norm = E_V + msb;

    // Overflow is measured on the ACTUAL result bit pattern rather than
    // re-derived from the exponent: an approximation here is easy to get
    // subtly wrong (the first version forgot the significand's own msb).
    if (FC::IS_FP16) {
        const uint16_t r = round_rne_fp16(S, ne, F).to_uint();
        if (((r >> 10) & 0x1F) == 0x1F && (r & 0x3FF) == 0) C.rne16_overflow++;
        if (e_norm >= -14) {
            C.rne16_normal++;
            const int drop = msb - 10;
            if (drop > 0) {
                C.rne16_normal_rounding++;
                uint64_t       Q    = A >> drop;
                const uint64_t rem  = A & ((1ull << drop) - 1);
                const uint64_t half = 1ull << (drop - 1);
                if (rem > half || (rem == half && (Q & 1))) Q++;
                if (Q == (1ull << 11)) C.rne16_carry++;
            }
        } else {
            C.rne16_subnormal++;
        }
    } else {
        const uint32_t r = round_rz_e8m13(S, ne, F).to_uint();
        if (((r >> 23) & 0xFF) == 0xFF && (r & 0x7FFFFF) == 0) C.rz32_overflow++;
        if (e_norm >= -126) {
            C.rz32_normal++;
            if (msb - 13 > 0) C.rz32_truncating++;
        } else {
            C.rz32_subnormal++;
        }
    }
}

void report(const std::string& label, const Counts& c) {
    std::cout << "  " << label << "\n"
              << "    blocks=" << c.total << "  numeric=" << c.n
              << "  Inf/NaN-collapsed=" << c.special << " ("
              << (c.total ? (100.0 * c.special / c.total) : 0.0) << "%)\n"
              << "    f32: subnormal=" << c.rz32_subnormal
              << " truncating=" << c.rz32_truncating << " overflow=" << c.rz32_overflow << "\n"
              << "    f16: subnormal=" << c.rne16_subnormal
              << " rounding=" << c.rne16_normal_rounding << " carry=" << c.rne16_carry
              << " overflow=" << c.rne16_overflow << "\n"
              << "    S==0=" << c.s_zero << "\n";
}

}  // namespace

int sc_main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <dump> [more dumps...]\n";
        return 2;
    }
    for (int arg = 1; arg < argc; ++arg) {
        std::ifstream in(argv[arg]);
        if (!in) {
            std::cerr << "cannot open " << argv[arg] << "\n";
            return 2;
        }
        std::cout << "=== " << argv[arg] << " ===\n";

        // One pair of counters per section, reported once when the section ends
        // (at the next header, or at EOF).
        std::string fmt;
        Counts      c32, c16;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (!fmt.empty()) {
                    report(fmt + " .f32", c32);
                    report(fmt + " .f16", c16);
                }
                const size_t p = line.find("fmt=");
                if (p == std::string::npos) continue;
                fmt = line.substr(p + 4, 4);
                c32 = Counts();
                c16 = Counts();
                continue;
            }
            // Parse as long long: c32 is a uint32 printed in decimal and can
            // exceed INT_MAX, which makes `ss >> int` fail and silently
            // truncate the line (leaving v.size() < 68 and later indices out
            // of bounds). That bug produced plausible-looking but wrong
            // coverage numbers until the field count was checked.
            std::istringstream     ss(line);
            std::vector<long long> v;
            long long              x;
            while (ss >> x) v.push_back(x);
            if (v.size() != 68) {
                std::cerr << "malformed dump line (" << v.size() << " fields)\n";
                return 2;
            }
            uint8_t a[32], b[32];
            for (int k = 0; k < 32; ++k) {
                a[k] = (uint8_t)v[k];
                b[k] = (uint8_t)v[32 + k];
            }
            c32.total++;
            c16.total++;
            if (fmt == "e4m3") {
                classify<e4m3, f32>(a, b, (uint32_t)v[64], c32);
                classify<e4m3, f16>(a, b, (uint32_t)v[65], c16);
            } else {
                classify<e5m2, f32>(a, b, (uint32_t)v[64], c32);
                classify<e5m2, f16>(a, b, (uint32_t)v[65], c16);
            }
        }
        if (!fmt.empty()) {
            report(fmt + " .f32", c32);
            report(fmt + " .f16", c16);
        }
    }
    return 0;
}
