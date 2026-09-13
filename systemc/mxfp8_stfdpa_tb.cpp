// =============================================================================
//  mxfp8_stfdpa_tb.cpp -- SystemC testbench for the Blackwell MXFP8 fused
//  32-element dot product (D = A*B, no accumulate).
//
//  Three things are checked:
//    1. golden vectors   -- bit patterns taken from the verified Python
//                           reference (mxfp8_stfdpa.py / MMA-Sim)
//    2. pipeline == comb -- the 4-stage SystemC pipeline must reproduce the
//                           combinational model bit for bit, in order
//    3. --dump N         -- emit random stimulus + results so that
//                           verify_vs_reference.py can diff against the
//                           Python reference model on the same inputs
//
//  usage:
//      ./mxfp8_stfdpa_tb                       # golden vectors + pipeline check
//      ./mxfp8_stfdpa_tb --dump 2000           # diff-test stream (e4m3)
//      ./mxfp8_stfdpa_tb --dump 2000 --fmt e5m2
//      ./mxfp8_stfdpa_tb --dump 2000 --fmt mixed
// =============================================================================

#include "mxfp8_stfdpa.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

using namespace mxfp8;
using std::printf;

static const int L = 32;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static void fill32(sc_uint<8>* v, std::initializer_list<int> vals)
{
    for (int k = 0; k < L; ++k) v[k] = 0;
    int i = 0;
    for (int x : vals) v[i++] = sc_uint<8>(unsigned(x));
}

static const char* hex8(const sc_uint<8>& v)
{
    static char buf[8][8];
    static int  n = 0;
    char* p = buf[n++ & 7];
    std::snprintf(p, 8, "%02X", unsigned(v.to_uint()));
    return p;
}

static unsigned bits_of(const sc_uint<32>& v) { return unsigned(v.to_uint()); }

// ---------------------------------------------------------------------------
// splitmix64 -- deterministic stimulus generator
// ---------------------------------------------------------------------------
struct rng_t {
    unsigned long long s;
    explicit rng_t(unsigned long long seed) : s(seed) {}
    unsigned long long next()
    {
        unsigned long long z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    unsigned u32() { return unsigned(next() >> 32); }
    int below(int n) { return int(u32() % unsigned(n)); }
};

// finite e4m3 byte (never 0x7F / 0xFF)
static sc_uint<8> rand_e4m3(rng_t& r, bool allow_special)
{
    unsigned b = r.u32() & 0xFFu;
    if (allow_special) return sc_uint<8>(b);
    unsigned sign = b & 0x80u, exp = (b >> 3) & 0xFu, man = b & 0x7u;
    if (exp == 0xFu && man == 0x7u) man = 0x6u;      // avoid NaN
    return sc_uint<8>(sign | (exp << 3) | man);
}

// finite e5m2 byte (never Inf/NaN)
static sc_uint<8> rand_e5m2(rng_t& r, bool allow_special)
{
    unsigned b = r.u32() & 0xFFu;
    if (allow_special) return sc_uint<8>(b);
    unsigned sign = b & 0x80u, exp = (b >> 2) & 0x1Fu, man = b & 0x3u;
    if (exp == 0x1Fu) exp = 0x1Eu;                   // avoid Inf/NaN
    return sc_uint<8>(sign | (exp << 2) | man);
}

// ---------------------------------------------------------------------------
// golden vectors (expected bits from the Python reference model)
// ---------------------------------------------------------------------------
struct golden_t {
    const char*        name;
    std::initializer_list<int> a;
    std::initializer_list<int> b;
    int                sa, sb;
    unsigned           c;
    unsigned           expect;
    int                fmt;      // 0 = e4m3 x e4m3, 1 = e5m2 x e5m2
};

static int run_golden()
{
    const golden_t G[] = {
        // D = A*B, no accumulate (C = 0), 3 terms + C = 0.5
        {"basic 1*1+2*2+3*3+0.5", {0x38, 0x40, 0x44}, {0x38, 0x40, 0x44}, 0x7F, 0x7F, 0x3F000000u, 0x41680000u, 0},
        // alignment truncation at F = 25 fractional bits
        {"trunc near (e_max=0)",  {0x38, 0x01},       {0x38, 0x01},       0x7F, 0x7F, 0x00000000u, 0x3F800020u, 0},
        {"trunc far  (e_max=16)", {0x78, 0x01},       {0x78, 0x01},       0x7F, 0x7F, 0x00000000u, 0x47800000u, 0},
        // denormalised products: same value, different alignment resolution
        {"denorm 12*24 (sig 2.25)", {0x54, 0x07, 0x01}, {0x5C, 0x01, 0x01}, 0x7F, 0x7F, 0, 0x43900001u, 0},
        {"denorm 18*16 (sig 1.125)", {0x59, 0x07, 0x01}, {0x58, 0x01, 0x01}, 0x7F, 0x7F, 0, 0x43900000u, 0},
        // output rounding is RZ: S = 2**25 + 1, result still 1.0
        {"RZ output (S=2^25+1)",  {0x3C, 0x08},       {0x3C, 0x0C},       0x7F, 0x7F, 0, 0x3F800000u, 1},
        // FP32 subnormal result, sign kept on underflow
        {"subnormal out 2^-129",  {0x01},             {0x38},             0x07, 0x7F, 0, 0x00100000u, 0},
        {"underflow keeps sign",  {0x81},             {0x38},             0x00, 0x6E, 0, 0x80000000u, 0},
        // C == 0 raises e_max to the E_ZERO = -133 floor and kills the product
        {"zero-C floor -> +0",    {0x82},             {0x02},             0x00, 0x6E, 0, 0x00000000u, 0},
        // C is added early and swamps the products
        {"C = 2^24 swamps",       {0x38},             {0x38},             0x7F, 0x7F, 0x4B800000u, 0x4B800000u, 0},
        // all-zero block
        {"all zero -> +0",        {},                 {},                 0x7F, 0x7F, 0, 0x00000000u, 0},
        // special values
        {"E8M0 NaN scale",        {0x38},             {0x38},             0xFF, 0x7F, 0, 0x7FFFFFFFu, 0},
        {"E4M3 NaN in A",         {0x7F},             {0x38},             0x7F, 0x7F, 0, 0x7FFFFFFFu, 0},
        {"E5M2 Inf * 0",          {0x7C},             {0x00},             0x7F, 0x7F, 0, 0x7FFFFFFFu, 1},
        {"E5M2 +Inf + -Inf",      {0x7C, 0xFC},       {0x3C, 0x3C},       0x7F, 0x7F, 0, 0x7FFFFFFFu, 1},
        {"E5M2 +Inf survives",    {0x7C},             {0x3C},             0x7F, 0x7F, 0, 0x7F800000u, 1},
    };

    int fail = 0;
    for (const golden_t& g : G) {
        sc_uint<8> a[L], b[L];
        fill32(a, g.a);
        fill32(b, g.b);
        const sc_uint<32> c(g.c);
        const unsigned got = (g.fmt == 0)
            ? bits_of(mxfp8_dot<25, L, e4m3, e4m3>(a, b, sc_uint<8>(unsigned(g.sa)), sc_uint<8>(unsigned(g.sb)), c))
            : bits_of(mxfp8_dot<25, L, e5m2, e5m2>(a, b, sc_uint<8>(unsigned(g.sa)), sc_uint<8>(unsigned(g.sb)), c));
        const bool ok = (got == g.expect);
        if (!ok) ++fail;
        printf("  [%s] %-26s got 0x%08X  expect 0x%08X\n", ok ? "PASS" : "FAIL", g.name, got, g.expect);
    }
    printf("golden vectors: %zu cases, %d failure(s)\n", sizeof(G) / sizeof(G[0]), fail);
    return fail;
}

// ---------------------------------------------------------------------------
// combinational / pipeline equivalence
// ---------------------------------------------------------------------------
struct Case {
    sc_uint<8> a[L], b[L];
    sc_uint<8> sa, sb;
    sc_uint<32> c;
    bool       e5m2;
};

static void gen_case(rng_t& r, int mode, bool is5, Case& k)
{
    const bool special = (mode == 0);
    for (int i = 0; i < L; ++i) {
        k.a[i] = is5 ? rand_e5m2(r, special) : rand_e4m3(r, special);
        k.b[i] = is5 ? rand_e5m2(r, special) : rand_e4m3(r, special);
    }
    k.e5m2 = is5;
    k.sa = sc_uint<8>(unsigned(0x7F + r.below(11) - 5));            // 2^-5 .. 2^5
    k.sb = sc_uint<8>(unsigned(0x7F + r.below(11) - 5));
    k.c  = sc_uint<32>(0);

    switch (mode) {
    case 0:  // fully random bytes and scales: NaN / Inf / subnormal / zero
        k.sa = sc_uint<8>(unsigned(r.u32() & 0xFFu));
        k.sb = sc_uint<8>(unsigned(r.u32() & 0xFFu));
        k.c  = sc_uint<32>(r.u32());
        break;
    case 1:  // finite values, wide exponent spread
        break;
    case 2:  // full-range E8M0 scales: exercises the E_ZERO = -133 floor
        k.sa = sc_uint<8>(unsigned(r.below(255)));
        k.sb = sc_uint<8>(unsigned(r.below(255)));
        for (int i = 0; i < L; ++i) {
            k.a[i] = sc_uint<8>(unsigned(0x38 | (r.below(2) ? 0x80 : 0)));   // +/-1.0
            k.b[i] = sc_uint<8>(unsigned(0x38 | (r.below(2) ? 0x80 : 0)));
        }
        break;
    case 3:  // one dominant term + many tiny ones (alignment truncation)
        for (int i = 0; i < L; ++i) {
            k.a[i] = sc_uint<8>(0x01);                                   // 2^-9
            k.b[i] = sc_uint<8>(0x01);
        }
        k.a[0] = sc_uint<8>(unsigned(0x40 | (r.below(2) ? 0x80 : 0)));    // +/-2.0
        k.b[0] = sc_uint<8>(unsigned(0x78));                             // 256.0
        break;
    case 4:  // all-zero block, random C
        for (int i = 0; i < L; ++i) { k.a[i] = 0; k.b[i] = 0; }
        k.c = sc_uint<32>(r.u32());
        if ((k.c.to_uint() >> 23) & 0xFF) {}                             // any FP32 is fine
        break;
    default:
        break;
    }
}

// ---------------------------------------------------------------------------
// the DUT wrapper
// ---------------------------------------------------------------------------
SC_MODULE(Tb) {
    sc_clock              clk;
    sc_signal<bool>       rst_n, valid_in;
    sc_signal<sc_uint<8>> sa, sb;
    sc_signal<sc_uint<8>> a[L], b[L];
    sc_signal<sc_uint<32>> cin, d;
    sc_signal<bool>       vout;

    mxfp8_dot_block<25, L, e4m3, e4m3>* dut;

    Tb(sc_module_name n) : clk("clk", 10, SC_NS), dut(nullptr)
    {
        dut = new mxfp8_dot_block<25, L, e4m3, e4m3>("dut");
        dut->clk(clk); dut->rst_n(rst_n); dut->valid_in(valid_in);
        dut->scale_a(sa); dut->scale_b(sb); dut->c_in(cin);
        dut->valid_out(vout); dut->d_out(d);
        for (int i = 0; i < L; ++i) { dut->a_in[i](a[i]); dut->b_in[i](b[i]); }
        rst_n.write(false); valid_in.write(false);
    }

    // exactly one clock period (posedge -> posedge)
    void tick() { sc_start(10, SC_NS); }

    void drive(const Case& k)
    {
        for (int i = 0; i < L; ++i) { a[i].write(k.a[i]); b[i].write(k.b[i]); }
        sa.write(k.sa); sb.write(k.sb); cin.write(k.c);
        valid_in.write(true);
    }

    // push every case back to back, return the outputs in order
    std::vector<sc_uint<32>> run(const std::vector<Case>& cases)
    {
        std::vector<sc_uint<32>> outs;
        rst_n.write(false); valid_in.write(false);
        for (int i = 0; i < 3; ++i) tick();
        rst_n.write(true);

        size_t next = 0;
        const int guard = int(cases.size()) + 8 * mxfp8_dot_block<25, L, e4m3, e4m3>::LATENCY + 8;
        for (int cyc = 0; cyc < guard; ++cyc) {
            if (next < cases.size()) drive(cases[next++]);
            else                     valid_in.write(false);
            tick();
            if (vout.read()) outs.push_back(d.read());
            if (outs.size() >= cases.size() && next >= cases.size()) break;
        }
        return outs;
    }
};

static int run_pipeline_check()
{
    Tb tb("tb");
    rng_t r(0xC0FFEEull);

    std::vector<Case> cases;
    for (int mode = 1; mode <= 4; ++mode)
        for (int i = 0; i < 64; ++i) { Case k; gen_case(r, mode, false, k); cases.push_back(k); }

    const std::vector<sc_uint<32>> got = tb.run(cases);

    int fail = 0;
    if (got.size() != cases.size()) {
        printf("  [FAIL] pipeline produced %zu outputs for %zu inputs\n", got.size(), cases.size());
        return 1;
    }
    for (size_t i = 0; i < cases.size(); ++i) {
        const unsigned exp = bits_of(mxfp8_dot<25, L, e4m3, e4m3>(cases[i].a, cases[i].b, cases[i].sa, cases[i].sb, cases[i].c));
        if (bits_of(got[i]) != exp) {
            if (++fail <= 5)
                printf("  [FAIL] case %zu: pipeline 0x%08X != comb 0x%08X\n", i, bits_of(got[i]), exp);
        }
    }
    printf("pipeline == combinational: %zu cases, %d failure(s), latency %d cycles\n",
           cases.size(), fail, mxfp8_dot_block<25, L, e4m3, e4m3>::LATENCY);
    return fail;
}

// ---------------------------------------------------------------------------
// diff-test stream for verify_vs_reference.py
// ---------------------------------------------------------------------------
// the fused-adder fractional width F is a template parameter; the verified
// Blackwell/B300 operating point is F = 25 (default)
template <int F>
static unsigned eval_block(const Case& k)
{
    return k.e5m2 ? bits_of(mxfp8_dot<F, L, e5m2, e5m2>(k.a, k.b, k.sa, k.sb, k.c))
                  : bits_of(mxfp8_dot<F, L, e4m3, e4m3>(k.a, k.b, k.sa, k.sb, k.c));
}

static void dump_stream(int n, const char* fmt, unsigned long long seed, int F)
{
    rng_t r(seed);
    const bool only5 = (std::strcmp(fmt, "e5m2") == 0);
    const bool mixed = (std::strcmp(fmt, "mixed") == 0);

    // self-describing header: the verifier picks F up from here
    printf("# F=%d adder_magnitude_bits=%d\n", F, F + 1 + mxfp8::ceil_log2(L + 1));

    for (int i = 0; i < n; ++i) {
        const int  mode  = i % 5;
        const bool is5   = only5 || (mixed && (i % 2) == 1);
        Case k;
        gen_case(r, mode, is5, k);

        unsigned got;
        switch (F) {
        case 24: got = eval_block<24>(k); break;
        case 26: got = eval_block<26>(k); break;
        default: got = eval_block<25>(k); break;
        }

        printf("FMT=%s A=", is5 ? "e5m2" : "e4m3");
        for (int j = 0; j < L; ++j) printf("%s", hex8(k.a[j]));
        printf(" B=");
        for (int j = 0; j < L; ++j) printf("%s", hex8(k.b[j]));
        printf(" SA=%s SB=%s C=%08X D=%08X\n", hex8(k.sa), hex8(k.sb), unsigned(k.c.to_uint()), got);
    }
}

// ---------------------------------------------------------------------------
int sc_main(int argc, char* argv[])
{
    int         dump_n = 0;
    const char* fmt    = "e4m3";
    int         opt_F  = 25;
    unsigned long long seed = 0x1234567ull;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump_n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--fmt") && i + 1 < argc) fmt = argv[++i];
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 0);
        else if (!std::strcmp(argv[i], "--F") && i + 1 < argc) opt_F = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--help")) {
            printf("usage: %s [--dump N] [--fmt e4m3|e5m2|mixed] [--seed S] [--F 24|25|26]\n", argv[0]);
            return 0;
        }
    }

    // keep stdout pure data in dump mode
    FILE* info = (dump_n > 0) ? stderr : stdout;
    fprintf(info, "Blackwell / B300 MXFP8 fused 32-element dot product (D = A*B, no accumulate)\n");
    fprintf(info, "F = %d fractional bits, adder register = F + 1 + ceil(log2(L+1)) = %d magnitude bits (+1 sign)\n",
            mxfp8_dot_block<25, L, e4m3, e4m3>::ADDER_FRAC_BITS,
            mxfp8_dot_block<25, L, e4m3, e4m3>::ADDER_WIDTH);
    fprintf(info, "(the same datapath is templated: F=24 -> %d bits, F=26 -> %d bits)\n",
            mxfp8_dot_block<24, L, e4m3, e4m3>::ADDER_WIDTH,
            mxfp8_dot_block<26, L, e4m3, e4m3>::ADDER_WIDTH);
    fprintf(info, "rounding = RZ-FP32, block scale = E8M0 (scale_vec::1X), zero-term floor = -133\n\n");

    if (dump_n > 0) { dump_stream(dump_n, fmt, seed, opt_F); return 0; }

    printf("-- golden vectors --\n");
    int fail = run_golden();
    printf("\n-- pipeline --\n");
    fail += run_pipeline_check();

    printf("\n%s\n", fail == 0 ? "ALL TESTS PASSED" : "FAILURES PRESENT");
    return fail == 0 ? 0 : 1;
}
