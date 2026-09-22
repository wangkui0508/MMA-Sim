// =============================================================================
//  hopper_fp8_wgmma_tb.cpp
//
//  Testbench for the Hopper (SM90) FP8 wgmma datapath model.
//
//    ./hopper_fp8_wgmma_tb                     golden vectors + pipeline check
//    ./hopper_fp8_wgmma_tb --dump N --fmt e4m3 --seed 0x1234
//                                              emit N random blocks for the
//                                              Python differential verifier
//    ./hopper_fp8_wgmma_tb --walkthrough        annotated single-block trace
//
//  Three independent checks:
//    1. GOLDEN  -- fixed blocks whose expected bits were produced by the
//                  MMA-Sim Python oracle (WGMMA("Hopper", ...)).
//    2. PIPELINE-- the 4-stage SystemC pipeline must agree with the
//                  combinational hopper_fp8_dot() on every block.
//    3. DUMP    -- vectors for verify_hopper_fp8_vs_mmasim.py, which does the
//                  real differential test against the repo's Python model.
// =============================================================================

#include "hopper_fp8_wgmma.h"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace hopper_fp8;

// ---------------------------------------------------------------------------
// Golden vectors. Expected bits generated with the MMA-Sim oracle:
//   WGMMA("Hopper","m64n8k32.f32.<fmt>.<fmt>")  -> exp32   (RZ-E8M13)
//   WGMMA("Hopper","m64n8k32.f16.<fmt>.<fmt>")  -> exp16   (RNE-FP16)
// ---------------------------------------------------------------------------
#define R32(v)                                                                        \
    v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, \
        v, v, v, v, v
#define Z32 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define MAX_THEN_MIN 0x7E, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, \
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01

struct Golden {
    const char* name;
    const char* fmt;
    uint8_t     a[32];
    uint8_t     b[32];
    uint32_t    c32;    // FP32 accumulator bits for the .f32 variant
    uint16_t    c16;    // FP16 accumulator bits for the .f16 variant
    uint32_t    exp32;  // from WGMMA("Hopper","...f32.<fmt>.<fmt>")
    uint32_t    exp16;  // from WGMMA("Hopper","...f16.<fmt>.<fmt>")
};

static const Golden GOLDEN[] = {
    {"all +0", "e4m3", {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 0x00000000u, 0x0000u, 0x00000000u, 0x0000u},
    {"all -0", "e4m3", {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80}, {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80}, 0x00000000u, 0x0000u, 0x00000000u, 0x0000u},
    {"all -0, c=-0", "e4m3", {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80}, {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 0x80000000u, 0x8000u, 0x00000000u, 0x0000u},
    {"1.875*1.875 x32", "e4m3", {0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f}, {0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f}, 0x00000000u, 0x0000u, 0x42E10000u, 0x5708u},
    {"1.0*1.0 x32, c=1.0", "e4m3", {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, 0x3F800000u, 0x3C00u, 0x42040000u, 0x5020u},
    {"exact cancellation", "e4m3", {0x38,0xb8,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x38,0x38,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 0x00000000u, 0x0000u, 0x00000000u, 0x0000u},
    {"max*max overflow", "e4m3", {0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e}, {0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e,0x7e}, 0x7F7FFFFFu, 0x7C00u, 0x7F7FFC00u, 0x7C00u},
    {"min subnormal^2", "e4m3", {0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01}, {0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01}, 0x00000000u, 0x0000u, 0x39000000u, 0x0800u},
    {"min subnormal^2 +c", "e4m3", {0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01}, {0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01}, 0x00000001u, 0x0000u, 0x39000000u, 0x0800u},
    {"NaN in A", "e4m3", {0x7f,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, 0x00000000u, 0x0000u, 0x7FFFFFFFu, 0x7FFFu},
    {"c = NaN", "e4m3", {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, 0x7FC00000u, 0x7E00u, 0x7FFFFFFFu, 0x7FFFu},
    {"c = +Inf", "e4m3", {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, 0x7F800000u, 0x7C00u, 0x7F800000u, 0x7C00u},
    {"wide dynamic range", "e4m3", {0x7e,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01}, {0x7e,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01}, 0x00000000u, 0x0000u, 0x48440000u, 0x7C00u},
    {"c not fp16-exact", "e4m3", {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, 0x461CFB6Cu, 0x70E8u, 0x461D7800u, 0x70ECu},
    {"c fp16-subnormal", "e4m3", {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, {0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38}, 0x33800000u, 0x0001u, 0x42000000u, 0x5000u},
    {"+Inf * 1.0", "e5m2", {0x7c,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c}, 0x00000000u, 0x0000u, 0x7F800000u, 0x7C00u},
    {"-Inf * 1.0", "e5m2", {0xfc,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c}, 0x00000000u, 0x0000u, 0xFF800000u, 0xFC00u},
    {"Inf * 0 -> NaN", "e5m2", {0x7c,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 0x00000000u, 0x0000u, 0x7FFFFFFFu, 0x7FFFu},
    {"+Inf and -Inf -> NaN", "e5m2", {0x7c,0xfc,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c}, 0x00000000u, 0x0000u, 0x7FFFFFFFu, 0x7FFFu},
    {"c = +Inf (e5m2)", "e5m2", {0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c}, {0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c,0x3c}, 0x7F800000u, 0x7C00u, 0x7F800000u, 0x7C00u},
    {"1.75*1.75 x32", "e5m2", {0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f}, {0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f,0x3f}, 0x00000000u, 0x0000u, 0x42C40000u, 0x5620u},
    {"C32 tiny subnormal (f32 sub out)", "e4m3", {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 0x00000400u, 0x0000u, 0x00000400u, 0x0000u},
    {"C16 min subnormal (f16 sub out)", "e4m3", {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 0x00000000u, 0x0001u, 0x00000000u, 0x0001u},
    {"f16 RNE carry out of significand", "e5m2", {0x3c,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, {0x10,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 0x3F800000u, 0x3FFFu, 0x3F801000u, 0x4000u},
};
static const int NGOLDEN = (int)(sizeof(GOLDEN) / sizeof(GOLDEN[0]));

// ---------------------------------------------------------------------------
// Deterministic PRNG: identical vectors on every platform/run.
// ---------------------------------------------------------------------------
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint32_t next() {  // xorshift64*
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return (uint32_t)((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
    uint32_t below(uint32_t n) { return next() % n; }
};

// ---------------------------------------------------------------------------
// Random block generators. The pools deliberately over-sample zeros,
// subnormals, the extremes and (for E5M2) Inf/NaN.
// ---------------------------------------------------------------------------
static const uint8_t E4M3_POOL[] = {
    0x00, 0x80, 0x01, 0x81, 0x02, 0x82, 0x07, 0x87, 0x08, 0x88,
    0x38, 0xB8, 0x3C, 0xBC, 0x40, 0xC0, 0x7E, 0xFE, 0x7F, 0xFF,
};
static const uint8_t E5M2_POOL[] = {
    0x00, 0x80, 0x01, 0x81, 0x03, 0x83, 0x04, 0x84, 0x3C, 0xBC,
    0x40, 0xC0, 0x7B, 0xFB, 0x7C, 0xFC, 0x7D, 0x7E, 0xFD, 0xFE,
};

// True for byte patterns that are Inf or NaN in the given format.
static bool is_special_byte(uint8_t bits, int fmt_id) {
    if (fmt_id == 0) {  // E4M3: no Inf; S.1111.111 is the only NaN
        return ((bits >> 3) & 0xF) == 0xF && (bits & 0x7) == 0x7;
    }
    return ((bits >> 2) & 0x1F) == 0x1F;  // E5M2: all-ones exponent -> Inf or NaN
}

static uint8_t gen_byte_raw(Rng& rng, const uint8_t* pool, int pool_n, int fmt_id, int kind) {
    switch (kind) {
        case 0: return pool[rng.below((uint32_t)pool_n)];
        case 1: {  // finite, wide exponent spread
            const int man_bits = (fmt_id == 0) ? 3 : 2;
            const int max_e    = (fmt_id == 0) ? 14 : 30;  // largest non-all-ones field
            // `e` is an EXPONENT FIELD, so it must stay <= max_e. (An earlier
            // version used below(2*max_e + 1): every drawn e above max_e wrapped
            // through the uint8_t cast into a different exponent field, which is
            // how ~1/60 of "finite" E5M2 operands silently became Inf/NaN.)
            const int e = (int)rng.below((uint32_t)max_e + 1);
            const int m = (int)rng.below(1u << man_bits);
            return (uint8_t)((e << man_bits) | m | (rng.below(2) ? 0x80 : 0));
        }
        default: return (uint8_t)rng.next();
    }
}

// finite_only must reject Inf/NaN from EVERY kind: the curated pools contain
// special bytes, and the uniform kind always can. A single special operand
// collapses the whole dot product to NaN/Inf, which would make the "finite"
// stimulus exercise nothing but NaN propagation.
static uint8_t gen_byte(Rng& rng, const uint8_t* pool, int pool_n, int fmt_id, int kind,
                        bool finite_only) {
    for (int attempt = 0; attempt < 256; ++attempt) {
        const uint8_t v = gen_byte_raw(rng, pool, pool_n, fmt_id, kind);
        if (!finite_only || !is_special_byte(v, fmt_id)) return v;
    }
    return 0x00;  // +0 is always finite
}

struct Block {
    uint8_t  a[32], b[32];
    uint32_t c32;   // accumulator bits for the .f32 variant  (FP32 accumulator)
    uint16_t c16;   // accumulator bits for the .f16 variant  (FP16 accumulator)
};

static Block gen_block(Rng& rng, int fmt_id, bool finite_only) {
    const uint8_t* pool   = (fmt_id == 0) ? E4M3_POOL : E5M2_POOL;
    const int      pool_n = (fmt_id == 0) ? (int)(sizeof(E4M3_POOL)) : (int)(sizeof(E5M2_POOL));
    Block blk;
    for (int k = 0; k < 32; ++k) {
        int kind = finite_only ? (int)rng.below(2) : (int)rng.below(3);
        blk.a[k] = gen_byte(rng, pool, pool_n, fmt_id, kind, finite_only);
        kind     = finite_only ? (int)rng.below(2) : (int)rng.below(3);
        blk.b[k] = gen_byte(rng, pool, pool_n, fmt_id, kind, finite_only);
    }
    // The accumulator type is the output type (c_type = d_type), so each
    // variant gets its own C, generated independently -- which is also what
    // tests/equivalence/helper.py does. Mostly finite, with a healthy share of
    // zeros and a few specials.
    uint32_t r = rng.next();
    if (r % 8 == 0) {
        blk.c32 = 0x00000000u;
    } else if (r % 8 == 1) {
        blk.c32 = 0x80000000u;
    } else if (!finite_only && r % 32 == 2) {
        blk.c32 = 0x7F800000u;
    } else if (!finite_only && r % 32 == 3) {
        blk.c32 = 0xFF800000u;
    } else {
        do { r = rng.next(); } while (((r >> 23) & 0xFF) == 0xFF);
        blk.c32 = r;
    }
    uint32_t r16 = rng.next();
    if (r16 % 8 == 0) {
        blk.c16 = 0x0000u;
    } else if (r16 % 8 == 1) {
        blk.c16 = 0x8000u;
    } else if (!finite_only && r16 % 32 == 2) {
        blk.c16 = 0x7C00u;
    } else if (!finite_only && r16 % 32 == 3) {
        blk.c16 = 0xFC00u;
    } else {
        do { r16 = rng.next(); } while (((r16 >> 10) & 0x1F) == 0x1F);
        blk.c16 = (uint16_t)(r16 & 0xFFFFu);
    }
    return blk;
}

// ---------------------------------------------------------------------------
// A DUT plus its stimulus signals.
// ---------------------------------------------------------------------------
template <class FMT>
struct Dut {
    static const int F  = DEFAULT_F;
    static const int L  = DEFAULT_L;
    static const int EZ = DEFAULT_E_ZERO;
    // The accumulator/output format is a template parameter: FP32 for the .f32
    // variant, FP16 for .f16.
    typedef hopper_fp8_wgmma<F, L, FMT, FMT, f32, EZ> TOP32;
    typedef hopper_fp8_wgmma<F, L, FMT, FMT, f16, EZ> TOP16;

    sc_signal<bool>         rst_n, in_valid, out_valid32, out_valid16;
    sc_signal<sc_uint<8> >  a[L], b[L];
    sc_signal<sc_uint<32> > c32, d32;
    sc_signal<sc_uint<16> > c16, d16;
    TOP32*                  u32;
    TOP16*                  u16;

    static const int LATENCY = 5;  // 4 stages + the driving cycle

    Dut(sc_module_name nm, sc_clock& clk) {
        u32 = new TOP32((std::string(nm) + "_f32").c_str());
        u32->clk(clk); u32->rst_n(rst_n); u32->in_valid(in_valid);
        for (int k = 0; k < L; ++k) { u32->a[k](a[k]); u32->b[k](b[k]); }
        u32->c(c32); u32->out_valid(out_valid32); u32->d(d32);

        u16 = new TOP16((std::string(nm) + "_f16").c_str());
        u16->clk(clk); u16->rst_n(rst_n); u16->in_valid(in_valid);
        for (int k = 0; k < L; ++k) { u16->a[k](a[k]); u16->b[k](b[k]); }
        u16->c(c16); u16->out_valid(out_valid16); u16->d(d16);
    }
    ~Dut() { delete u32; delete u16; }

    void write(const Block& blk) {
        for (int k = 0; k < L; ++k) {
            a[k].write((uint8_t)blk.a[k]);
            b[k].write((uint8_t)blk.b[k]);
        }
        c32.write(blk.c32);
        c16.write(blk.c16);
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void fmt_of(const char* fmt, int& fmt_id, bool& is_e5m2) {
    is_e5m2 = (std::strcmp(fmt, "e5m2") == 0);
    fmt_id  = is_e5m2 ? 1 : 0;
}

// The combinational reference, dispatched on the format.
template <class FMT, class FC>
static uint32_t comb_dot(const Block& blk) {
    typedef widths<DEFAULT_F, DEFAULT_L, FMT, FMT, FC, DEFAULT_E_ZERO> W;
    sc_uint<8> ab[32], bb[32];
    for (int k = 0; k < 32; ++k) { ab[k] = blk.a[k]; bb[k] = blk.b[k]; }
    sc_uint<W::C_BITS> cb = FC::IS_FP16 ? (uint32_t)blk.c16 : blk.c32;
    return hopper_fp8_dot<DEFAULT_F, DEFAULT_L, FMT, FMT, FC, DEFAULT_E_ZERO>(ab, bb, cb)
        .to_uint();
}

// ---------------------------------------------------------------------------
// Annotated single-block walkthrough of the datapath (--walkthrough).
// This is the "how does Hopper FP8 wgmma actually work" dump: it shows every
// stage on one block that deliberately spans a wide exponent range, so the
// alignment truncation is visible.
// ---------------------------------------------------------------------------
template <class FMT, class FC>
static void walkthrough(const Block& blk, const char* fn) {
    typedef widths<DEFAULT_F, DEFAULT_L, FMT, FMT, FC, DEFAULT_E_ZERO> W;
    typedef decoded_t<W::TERM_SIG, W::EXP> term_t;
    const int F = DEFAULT_F, L = DEFAULT_L, EZ = DEFAULT_E_ZERO;

    std::cout << "=====================================================================\n";
    std::cout << " wgmma.mma_async.sync.aligned.m64n8k32." << (FC::IS_FP16 ? "f16" : "f32")
              << "." << fn << "." << fn << "  --  one output element\n";
    std::cout << " accumulator C is " << (FC::IS_FP16 ? "FP16 (11-bit significand)"
                                                     : "FP32 (24-bit significand)")
              << "; it joins the same fused summation\n";
    std::cout << " F=" << F << "  L_max=" << L << "  e_zero=" << EZ << "  K=" << L
              << "  =>  K == L_max : ONE fused summation node\n";
    std::cout << "=====================================================================\n\n";

    std::cout << "[1] DECODE  (value = (-1)^sign * sig * 2^exp)\n";
    std::cout << "      k    bits    a_k                  bits    b_k\n";
    for (int k = 0; k < 6; ++k) {
        const decoded_t<FMT::MAN_BITS + 1, W::EXP> da =
            decode_fp8<FMT, W::EXP>(sc_uint<8>(blk.a[k]));
        const decoded_t<FMT::MAN_BITS + 1, W::EXP> db =
            decode_fp8<FMT, W::EXP>(sc_uint<8>(blk.b[k]));
        std::ostringstream oa, ob;
        oa << da;
        ob << db;
        std::cout << "     " << std::setw(3) << k << "    0x" << std::hex << std::setw(2)
                  << std::setfill('0') << (int)blk.a[k] << std::setfill(' ') << std::dec
                  << "   " << std::setw(18) << oa.str() << "   0x" << std::hex << std::setw(2)
                  << std::setfill('0') << (int)blk.b[k] << std::setfill(' ') << std::dec
                  << "   " << std::setw(18) << ob.str() << "\n";
    }
    std::cout << "      ... (k = 6..31 omitted)\n\n";

    term_t t[W::NTERM];
    for (int k = 0; k < L; ++k) {
        const decoded_t<FMT::MAN_BITS + 1, W::EXP> da =
            decode_fp8<FMT, W::EXP>(sc_uint<8>(blk.a[k]));
        const decoded_t<FMT::MAN_BITS + 1, W::EXP> db =
            decode_fp8<FMT, W::EXP>(sc_uint<8>(blk.b[k]));
        t[k] = multiply<FMT::MAN_BITS + 1, W::EXP>(da, db);
    }
    t[L] = decode_accumulator<FC, W::EXP>(sc_uint<W::C_BITS>(FC::IS_FP16 ? (uint32_t)blk.c16
                                                                          : blk.c32));

    std::cout << "[2] PRODUCT  (exact integer significand, integer exponent sum;\n";
    std::cout << "              left DENORMALISED, so sig may be >= 2)\n";
    std::cout << "      k    sig_a*sig_b   exp   ne = exp+kfrac\n";
    for (int k = 0; k < 6; ++k)
        std::cout << "     " << std::setw(3) << k << "    " << std::setw(11)
                  << t[k].sig.to_uint() << "   " << std::setw(4) << t[k].exp.to_int()
                  << "   " << std::setw(5) << normalised_exp<W::TERM_SIG, W::EXP>(t[k], EZ)
                  << "\n";
    std::cout << "      C    " << std::setw(11) << t[L].sig.to_uint() << "   " << std::setw(4)
              << t[L].exp.to_int() << "   "
              << std::setw(5) << normalised_exp<W::TERM_SIG, W::EXP>(t[L], EZ)
              << "   (the accumulator C joins the SAME fused summation)\n\n";

    const uint8_t sp = special_code<W::TERM_SIG, W::EXP, W::NTERM>(t);
    if (sp != SP_NONE) {
        std::cout << "  a NaN/Inf term collapses the whole dot product -> 0x" << std::hex
                  << special_bits(sp, FC::IS_FP16) << std::dec << "\n";
        return;
    }

    int ne_max = EZ;
    for (int i = 0; i < W::NTERM; ++i)
        ne_max = std::max(ne_max, normalised_exp<W::TERM_SIG, W::EXP>(t[i], EZ));

    std::cout << "[3] ALIGN  one global exponent ne_max = " << ne_max
              << ", then TRUNCATE toward zero to F=" << F << " fractional bits\n";
    std::cout << "      quantum = 2^(ne_max-F) = 2^" << (ne_max - F)
              << " ; aligned = trunc(sig * 2^(exp-ne_max+F))\n";
    std::cout << "      k    shift    aligned\n";
    for (int k = 0; k < 6; ++k)
        std::cout << "     " << std::setw(3) << k << "    " << std::setw(5)
                  << (t[k].exp.to_int() - ne_max + F) << "    " << std::setw(9)
                  << align_term<W::ALIGNED, W::TERM_SIG, W::EXP>(t[k], ne_max, F).to_uint()
                  << "\n";
    std::cout << "      C    " << std::setw(5) << (t[L].exp.to_int() - ne_max + F) << "    "
              << std::setw(9)
              << align_term<W::ALIGNED, W::TERM_SIG, W::EXP>(t[L], ne_max, F).to_uint() << "\n";
    std::cout << "      NOTE: a negative shift discards those bits outright -- no sticky bit,\n";
    std::cout << "            no guard digit, no round-to-nearest. This is THE error source.\n\n";

    int64_t v[W::NTERM];
    for (int i = 0; i < W::NTERM; ++i) {
        const int64_t m =
            (int64_t)align_term<W::ALIGNED, W::TERM_SIG, W::EXP>(t[i], ne_max, F).to_uint();
        const bool neg = t[i].sign && t[i].cls == CLS_FINITE && t[i].sig.to_uint() != 0;
        v[i] = neg ? -m : m;
    }
    const int64_t S = adder_tree(v, W::NTERM);
    std::cout << "[4] ACCUMULATE  exact sign-magnitude adder tree over " << W::NTERM
              << " leaves, no intermediate rounding\n";
    std::cout << "      S = " << S << "   =>   exact value = S * 2^" << (ne_max - F) << "\n\n";

    std::cout << "[5] NORMALISE + ROUND  ONE rounding for the whole K loop\n";
    if (FC::IS_FP16) {
        const uint16_t r16 = round_rne_fp16(S, ne_max, F).to_uint();
        std::cout << "      .f16 output  RNE-FP16 keep 1+10 = 11 significant bits, nearest-even\n";
        std::cout << "          0x" << std::hex << std::setw(4) << std::setfill('0') << r16
                  << std::dec << std::setfill(' ') << "\n";
    } else {
        const uint32_t r32 = round_rz_e8m13(S, ne_max, F).to_uint();
        float    vf;
        uint32_t tmp = r32;
        std::memcpy(&vf, &tmp, 4);
        std::cout << "      .f32 output  RZ-E8M13  keep 1+13 = 14 significant bits, truncate\n";
        std::cout << "          0x" << std::hex << std::setw(8) << std::setfill('0') << r32
                  << std::dec << std::setfill(' ') << "  = " << vf << "\n";
        std::cout << "      =>  \"FP32 accumulate\" on Hopper FP8 is really 14 significant bits.\n";
    }
    std::cout << "\n";
}

int sc_main(int argc, char** argv) {
    std::string vfile;
    int  nrand  = 64;
    int  fmt_id = 0;
    bool is_e5m2 = false;
    uint64_t seed = 0x1234567ull;
    bool dump = false, walk = false, walk_f16 = false, finite_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dump") {
            dump = true;
            if (i + 1 < argc) nrand = std::atoi(argv[++i]);
        } else if (arg == "--fmt") {
            const std::string f = argv[++i];
            if (f == "mixed") { fmt_id = -1; }
            else { fmt_of(f.c_str(), fmt_id, is_e5m2); }
        } else if (arg == "--seed") {
            seed = std::strtoull(argv[++i], nullptr, 0);
        } else if (arg == "--finite") {
            finite_only = true;
        } else if (arg == "--walkthrough") {
            walk = true;
        } else if (arg == "--acc-f16") {
            walk_f16 = true;
        } else {
            vfile = arg;
        }
    }

    // --walkthrough needs no SystemC machinery at all: it just prints the
    // datapath stage by stage on one crafted block.
    if (walk) {
        Block blk;
        if (fmt_id == 1) {
            for (int k = 0; k < 32; ++k) {
                blk.a[k] = (uint8_t)(0x3C + (k % 3));
                blk.b[k] = (uint8_t)(0x3C + (k % 2));
            }
            blk.a[0] = 0x7B;  // 57344, the largest E5M2 normal
            blk.a[1] = 0x01;  // smallest E5M2 subnormal 2^-16
            blk.c32  = 0x3F800000u;
            blk.c16  = 0x3C00u;
            if (walk_f16) walkthrough<e5m2, f16>(blk, "e5m2");
            else          walkthrough<e5m2, f32>(blk, "e5m2");
        } else {
            for (int k = 0; k < 32; ++k) {
                blk.a[k] = (uint8_t)(0x38 + (k % 4));
                blk.b[k] = (uint8_t)(0x40 + (k % 3));
            }
            blk.a[0] = 0x7E;  // 448, the largest E4M3 normal
            blk.a[1] = 0x01;  // smallest E4M3 subnormal 2^-9
            blk.c32  = 0x3F800000u;
            blk.c16  = 0x3C00u;
            if (walk_f16) walkthrough<e4m3, f16>(blk, "e4m3");
            else          walkthrough<e4m3, f32>(blk, "e4m3");
        }
        return 0;
    }

    sc_clock clk("clk", 10, SC_NS);
    Dut<e4m3> d43("d43", clk);
    Dut<e5m2> d52("d52", clk);

    auto set_valid = [&](bool v) {
        d43.in_valid.write(v);
        d52.in_valid.write(v);
    };
    auto set_rst = [&](bool v) {
        d43.rst_n.write(v);
        d52.rst_n.write(v);
    };
    auto tick = [&]() { sc_start(10, SC_NS); };

    set_rst(false);
    set_valid(false);
    tick();
    tick();
    set_rst(true);

    // ------------------------- DUMP MODE -------------------------
    if (dump) {
        std::deque<Block> q;
        // Stream: drive one block per cycle, and every cycle that out_valid is
        // high the block that entered the pipeline LATENCY-1 cycles ago retires.
        // The FIFO keeps the emitted blocks paired with their own results.
        auto stream = [&](int id, const char* name) {
            Rng r(seed ^ (id ? 0xE5ull : 0xE4ull));
            q.clear();
            std::cout << "# fmt=" << name << " F=" << DEFAULT_F << " L=" << DEFAULT_L
                      << " e_zero=" << DEFAULT_E_ZERO
                      << " fp32_rho=RZ-E8M13 fp16_rho=RNE-FP16\n";
            int done = 0;
            for (int i = 0; i < nrand + 8 && done < nrand; ++i) {
                const bool driving = (i < nrand);
                if (driving) {
                    Block blk = gen_block(r, id, finite_only);
                    q.push_back(blk);
                    if (id == 0) d43.write(blk); else d52.write(blk);
                }
                set_valid(driving);
                tick();
                const bool v32 = (id == 0) ? d43.out_valid32.read() : d52.out_valid32.read();
                const bool v16 = (id == 0) ? d43.out_valid16.read() : d52.out_valid16.read();
                if (v32 && v16 && !q.empty()) {
                    const Block b = q.front();
                    q.pop_front();
                    const uint32_t o32 = (id == 0) ? d43.d32.read().to_uint()
                                                   : d52.d32.read().to_uint();
                    const uint32_t o16 = ((id == 0) ? d43.d16.read().to_uint()
                                                    : d52.d16.read().to_uint()) & 0xFFFFu;
                    std::ostringstream os;
                    for (int k = 0; k < 32; ++k) os << (int)b.a[k] << " ";
                    for (int k = 0; k < 32; ++k) os << (int)b.b[k] << " ";
                    os << b.c32 << " " << b.c16 << " " << o32 << " " << o16;
                    std::cout << os.str() << "\n";
                    ++done;
                }
            }
            set_valid(false);
        };
        if (fmt_id >= 0) {
            stream(fmt_id, fmt_id ? "e5m2" : "e4m3");
        } else {
            stream(0, "e4m3");
            stream(1, "e5m2");
        }
        for (int i = 0; i < Dut<e4m3>::LATENCY; ++i) tick();
        return 0;
    }

    // ---------------------- GOLDEN VECTORS -----------------------
    // Single-shot: drive one block, then advance until it retires. The
    // datapath is 4 stages deep, so the result is NOT there after one tick.
    int fails = 0;
    auto run_one = [&](int id, const Block& blk, uint32_t& o32, uint32_t& o16) -> bool {
        if (id == 0) d43.write(blk); else d52.write(blk);
        set_valid(true);
        for (int t = 0; t < 8; ++t) {
            tick();
            set_valid(false);  // single-shot: one block in flight
            const bool v32 = (id == 0) ? d43.out_valid32.read() : d52.out_valid32.read();
            const bool v16 = (id == 0) ? d43.out_valid16.read() : d52.out_valid16.read();
            if (v32 && v16) {
                o32 = (id == 0) ? d43.d32.read().to_uint() : d52.d32.read().to_uint();
                o16 = ((id == 0) ? d43.d16.read().to_uint() : d52.d16.read().to_uint()) & 0xFFFFu;
                return true;
            }
        }
        return false;
    };

    std::cout << "=== Hopper FP8 wgmma: golden vectors ===\n";
    std::cout << "    (expected bits from WGMMA(\"Hopper\", ...) in MMA-Sim)\n";
    for (int i = 0; i < NGOLDEN; ++i) {
        const Golden& g = GOLDEN[i];
        int  id; bool e5;
        fmt_of(g.fmt, id, e5);
        Block blk;
        std::memcpy(blk.a, g.a, 32);
        std::memcpy(blk.b, g.b, 32);
        blk.c32 = g.c32;
        blk.c16 = g.c16;
        uint32_t got32 = 0, got16 = 0;
        const bool got = run_one(id, blk, got32, got16);
        const bool ok  = got && got32 == g.exp32 && got16 == g.exp16;
        if (!ok) {
            ++fails;
            std::cout << "  FAIL " << g.name << " (" << g.fmt << ")\n"
                      << "       f32 got 0x" << std::hex << std::setw(8) << std::setfill('0')
                      << got32 << " want 0x" << std::setw(8) << g.exp32 << std::dec
                      << std::setfill(' ') << "\n"
                      << "       f16 got 0x" << std::hex << std::setw(4) << std::setfill('0')
                      << got16 << " want 0x" << std::setw(4) << g.exp16 << std::dec
                      << std::setfill(' ') << "\n";
        }
    }
    std::cout << "  golden vectors: " << (NGOLDEN - fails) << "/" << NGOLDEN << " identical\n";

    // ------------------ PIPELINE vs COMBINATIONAL ----------------
    // Drive back-to-back blocks (one per cycle) and compare each pipelined
    // result against the combinational model of the same block.
    std::cout << "\n=== 4-stage pipeline vs combinational model ===\n";
    {
        const int N = 256;
        Rng       rng(0xC0FFEEull);
        std::vector<Block> blocks;
        for (int i = 0; i < N; ++i) blocks.push_back(gen_block(rng, i & 1, false));

        // Pipeline.
        std::vector<uint32_t> p32[2], p16[2];
        // The two formats have separate DUTs; drive them one format at a time
        // (each with its own clock run) so results stay easy to pair up.
        for (int id = 0; id < 2; ++id) {
            set_valid(false);
            for (int i = 0; i < Dut<e4m3>::LATENCY; ++i) tick();
            for (int i = 0; i < N; ++i) {
                if ((i & 1) != id) continue;
                if (id == 0) d43.write(blocks[i]); else d52.write(blocks[i]);
                set_valid(true);
                tick();
                bool v32 = (id == 0) ? d43.out_valid32.read() : d52.out_valid32.read();
                bool v16 = (id == 0) ? d43.out_valid16.read() : d52.out_valid16.read();
                if (v32) p32[id].push_back((id == 0) ? d43.d32.read().to_uint()
                                                     : d52.d32.read().to_uint());
                if (v16) p16[id].push_back(((id == 0) ? d43.d16.read().to_uint()
                                                      : d52.d16.read().to_uint()) & 0xFFFFu);
            }
            set_valid(false);
            for (int i = 0; i < Dut<e4m3>::LATENCY + 2; ++i) {
                tick();
                bool v32 = (id == 0) ? d43.out_valid32.read() : d52.out_valid32.read();
                bool v16 = (id == 0) ? d43.out_valid16.read() : d52.out_valid16.read();
                if (v32) p32[id].push_back((id == 0) ? d43.d32.read().to_uint()
                                                     : d52.d32.read().to_uint());
                if (v16) p16[id].push_back(((id == 0) ? d43.d16.read().to_uint()
                                                      : d52.d16.read().to_uint()) & 0xFFFFu);
            }
        }

        int bad = 0, cmp = 0, expected = 0;
        for (int id = 0; id < 2; ++id) {
            std::vector<const Block*> bl;
            for (int i = 0; i < N; ++i)
                if ((i & 1) == id) bl.push_back(&blocks[i]);
            expected += (int)bl.size();
            // A pipeline that drops or stalls a beat would otherwise shrink the
            // comparison set and still report PASS, so require exact lengths.
            if (p32[id].size() != bl.size() || p16[id].size() != bl.size()) {
                std::cout << "  LENGTH MISMATCH " << (id ? "e5m2" : "e4m3") << ": drove "
                          << bl.size() << " blocks but captured f32=" << p32[id].size()
                          << " f16=" << p16[id].size() << "\n";
                bad += (int)bl.size();
                continue;
            }
            const size_t m = bl.size();
            for (size_t j = 0; j < m; ++j) {
                ++cmp;
                const uint32_t e32 = (id == 0) ? comb_dot<e4m3, f32>(*bl[j])
                                               : comb_dot<e5m2, f32>(*bl[j]);
                const uint32_t e16 = (id == 0) ? comb_dot<e4m3, f16>(*bl[j])
                                               : comb_dot<e5m2, f16>(*bl[j]);
                if (p32[id][j] != e32 || p16[id][j] != (e16 & 0xFFFFu)) {
                    if (bad < 5)
                        std::cout << "  MISMATCH " << (id ? "e5m2" : "e4m3") << " block " << j
                                  << ": pipe 0x" << std::hex << p32[id][j] << " comb 0x" << e32
                                  << std::dec << "\n";
                    ++bad;
                }
            }
        }
        const int good = cmp - bad + (expected - cmp);
        std::cout << "  pipeline vs combinational: " << good << "/" << expected
                  << " identical (latency " << Dut<e4m3>::LATENCY - 1 << " cycles)\n";
        if (cmp != expected) {
            std::cout << "  COMPARISON SHORTFALL: compared " << cmp << " of " << expected
                      << " blocks\n";
            fails += bad;
        } else {
            fails += bad;
        }
    }

    std::cout << "\n" << (fails == 0 ? "RESULT: PASS" : "RESULT: FAIL") << "\n";
    return fails == 0 ? 0 : 1;
}
