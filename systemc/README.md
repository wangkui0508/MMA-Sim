# SystemC models of NVIDIA tensor-core fused dot products

Two bit-accurate SystemC models of how NVIDIA tensor cores evaluate a fused
dot-product-accumulate. They are deliberately structured the same way (width-
annotated `sc_int`/`sc_uint` datapath in a header, a `_tb.cpp`, a Python
differential verifier against MMA-Sim), so the two architectures can be diffed
side by side.

| model | architecture | instruction | header |
|---|---|---|---|
| **1. MXFP8 fused dot product** | Blackwell B200 (SM100), B300/GB300 (SM103) | `tcgen05.mma.kind::mxf8f6f4.block_scale`, `mma.sync kind::mxf8f6f4` | `mxfp8_stfdpa.h` |
| **2. FP8 wgmma** | Hopper (SM90) | `wgmma.mma_async.sync.aligned.m64n8k32.{f32,f16}.{e4m3,e5m2}` | `hopper_fp8_wgmma.h` |

The headline numerical difference between the two, and the reason they are
worth reading together:

|  | Hopper FP8 wgmma | Blackwell MXFP8 |
|---|---|---|
| fused-summation fractional width `F` | **13** | **25** |
| fused terms per node `L_max` | 32 | 32 |
| block scale in the exponent | no | yes (E8M0, one per 32) |
| result significand | **14 bits** (RZ) | **24 bits** (RZ) |

---

# Model 1 — Blackwell / B300 MXFP8 fused 32-element dot product

Bit-accurate SystemC model of what a Blackwell 5th-gen tensor core does when it
multiplies **32 pairs of FP8 values together and sums them, with no accumulate
term** — i.e. the `D = A * B` (C = 0) case of an MXFP8 block-scaled matrix
multiply on **B200 (SM100)** and **B300 / GB300 (SM103)**:

```
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X   (UTCQMMA)
mma.sync.aligned.m16n8k32...kind::mxf8f6f4.block_scale.scale_vec::1X (QMMA.SF)
```

Files:

| file | what it is |
|---|---|
| `mxfp8_stfdpa.h` | the model: width-annotated `sc_int`/`sc_uint` datapath + 4-stage SystemC pipeline |
| `mxfp8_stfdpa_tb.cpp` | testbench: golden vectors, pipeline-vs-combinational, random stimulus dump |
| `verify_vs_reference.py` | differential test vs `../mxfp8_stfdpa.py` (standalone reference model) |
| `verify_vs_mmasim.py` | differential test vs the MMA-Sim package (`src/mmasim/.../fdpa.py`) |
| `Makefile` | `make`, `make run`, `make verify` |
The fused-adder fractional width is the C++ template parameter `F`
(`mxfp8_dot<F, L, ...>` / `mxfp8_dot_block<F, L, ...>`), **default 25** — the
value MMA-Sim verified for the Blackwell MXFP8 path. The accumulator width
follows from it (`F=24 -> 31 bits`, `F=25 -> 32 bits`, `F=26 -> 33 bits`).

Build and run (SystemC 3.0.2 was used, `/usr/local`):

```shell
cd systemc
make run                 # golden vectors + pipeline check
make verify N=20000      # 3 x 20000 random blocks vs the Python reference
python3 verify_vs_mmasim.py 6000 mxfp8_dump.txt mxfp8_dump.txt.e5m2 mxfp8_dump.txt.mixed

# a non-default adder width: the dump is self-describing, the verifier follows
./mxfp8_stfdpa_tb --dump 3000 --F 24 > f24.txt
python3 verify_vs_reference.py f24.txt          # -> 3000 blocks (F=24), 0 mismatched
```

---

## 1. Is the adder width 25?

**F = 25 is the right default, but 25 is the *fractional* width, not the whole adder.**
`F` is the number of fractional bits the fused adder keeps below the alignment
point (the parameter called `F` in MMA-Sim / ST-FDPA, and the value the model was
verified with for the Blackwell MXFP8 row). The adder register itself is:

```
MAG = F + 1 + ceil(log2(L + 1))
    = 25 + 1 + 6            (L = 32: 32 products + the FP32 accumulator C)
    = 32 bits               (+1 sign bit in a sign-magnitude datapath)
```

| bits | meaning |
|---|---|
| `31..27` | carry headroom for 33 accumulated terms |
| `26..25` | the **2 integer bits** of a denormalised product significand (`s < 3.515625`) |
| `24..0`  | the **F = 25 fractional bits** kept below the alignment point |

The "+1" is not padding: products are never renormalised, so a product
significand can be in `[2,4)` and needs a second integer bit. The 6 carry bits
are `ceil(log2(33))`. Tightness: `32 * 3.515625 + 2 = 114.5 < 128 = 2**7`, so
`|S| < 2**7 * 2**F = 2**32` — 32 magnitude bits are exactly enough, and a
two's-complement accumulator would need 33 bits because bit 31 is data, not
sign. In `mxfp8_stfdpa.h` this is `widths<...>::MAG`, guarded by
`static_assert(W::MAG == 32)`; changing the `F` template parameter changes the
adder accordingly (`F=24` → 31 bits, `F=26` → 33 bits).

## 2. What the model computes

Per 32-element MX block (one `K = 32` instruction, `scale_vec::1X` → one E8M0
scale per operand per block):

1. **MULTIPLY** — exact integer product of the two signed significands; the
   exponents are *summed as integers, including both E8M0 scale exponents*.
   E8M0 has an implicit significand of exactly 1.0, so the block scale enters
   purely in the exponent domain, before alignment:
   `s_k = sig_a * sig_b`, `e_k = exp_a + exp_b + exp_scale_a + exp_scale_b`.
2. **ALIGN** — one single alignment point `e_max = max(e_0..e_31, e_C)`; every
   term is truncated **toward zero** to `F = 25` fractional bits below `e_max`.
   With `C = 0` there is one extra rule: the zero accumulator joins the
   summation at the floor exponent `e_zero = -133` and can therefore *raise*
   `e_max` and flush tiny products to `+0`. This is real hardware behaviour
   (`fdpa.py`: `e[s == 0.0] = e_zero`) and it is what makes extreme E8M0 scales
   (down to `2**-127`) behave as measured.
3. **ACCUMULATE** — exact sign-magnitude sum of the 33 aligned terms, no
   intermediate rounding anywhere.
4. **ROUND** — one single conversion of `S * 2**(e_max - F)` to FP32 with
   **round-toward-zero**; FP32 subnormals are kept down to `2**-149`, underflow
   keeps the sign, exact cancellation gives `+0.0`, finite overflow gives `±Inf`.

Verified parameter row (MMA-Sim, [arXiv:2511.10909](https://arxiv.org/abs/2511.10909),
`nv_ptx/sim.py`): `MXFP8 -> FP32`, `L_max = 32`, `F = 25`, `rho = RZ-FP32`,
`e_zero = -133`.

### Signal widths (all `sc_int` / `sc_uint`, see `widths<>` in the header)

| signal | type | why |
|---|---|---|
| one FP8 significand | `sc_uint<4>` (e4m3) / `sc_uint<3>` (e5m2) | 1 implicit + mantissa bits |
| exact product significand | `sc_uint<8>` / `sc_uint<6>` | `15*15 = 225`, `7*7 = 49` |
| exponent (incl. both E8M0 scales) | `sc_int<10>` | `-282 .. +284` |
| alignment shift count | `sc_int<12>` | `n = (P-F) - (exp - e_max)`, `-587 .. +547` |
| aligned term | `sc_uint<27>` | `< 3.515625 * 2**25 < 2**27` |
| accumulator magnitude | `sc_uint<32>` | `= F + 1 + ceil(log2(L+1))` |
| result | `sc_uint<32>` | FP32 bit pattern |

The `sc_int<12>` shift counter is load-bearing: with an 8-bit counter, a tiny `C`
far below a large `e_max` wraps the right shift that should flush the term into a
huge left shift. That exact bug was caught by the differential test (all 11
mismatching blocks out of 2000 had a non-zero, far-below `C`).

## 3. Why the widths are not IEEE

Products are left denormalised, so the same mathematical product can align
differently depending on how it was formed. Both cases below are `288`, but

* `12 * 24` → significand `2.25` (uses the 2nd integer bit), `exp = 7`
* `18 * 16` → significand `1.125`, `exp = 8`

align one bit apart, and the extra addends of the first case survive into the
result: `0x43900001` vs `0x43900000`. Both are golden vectors (`a4`/`a5`) in the
testbench. This "two algebraically identical products give different results"
behaviour is the headline non-IEEE property of these units.

## 4. Verification status

| check | result |
|---|---|
| golden vectors (16 fixed cases incl. NaN/Inf/subnormal/floor) | 16/16 identical |
| 4-stage pipeline vs combinational model, 256 back-to-back blocks | 256/256 identical, latency 4 |
| SystemC vs `mxfp8_stfdpa.py`, e4m3 / e5m2 / mixed, 3 x 20000 blocks | 60000/60000 identical |
| SystemC vs MMA-Sim package `fdpa.st_fdpa` (F=25, RZ-FP32, e_zero=-133), 3 x 6000 blocks | 18000/18000 identical |
| template parameter sweep F = 24 / 25 / 26 vs the parameterised reference, 6 x 3000 blocks | 18000/18000 identical |

Stimulus mixes fully random bytes (NaN, Inf, subnormal, ±0), finite values with
wide exponent spread, full-range E8M0 scales (stresses the `-133` floor and FP32
subnormal outputs), dominant-plus-tiny-term blocks (alignment truncation), and
all-zero blocks with random `C`.

## 5. Caveats

* Confirmed for the `D = A*B` (no-accumulate) case; `C` is still modelled (and
  participates early, in the same fused summation) so that
  `mxfp8_stfdpa_tb --dump` can exercise the full `D = A*B + C` block — wire
  `c_in = 0` for no-accumulate.
* `B200` (SM100) and `B300` (SM103) share the 5th-gen tensor core FDPA
  datapath; no public documentation suggests a different FP8 dot-product
  behaviour on SM103, but the model is parameterised (`F`, `E_ZERO`, formats) so
  a different operating point can be tested without touching the datapath.
* MMA-Sim itself documents one open corner: for fp16/bf16 inputs the alignment
  exponent is clamped from below, and the fp8 rows of that table list no such
  value, so non-zero products are not clamped here.
* `verify_vs_mmasim.py` calls the module-level `fdpa.st_fdpa`; the
  `MMA_ST_FDPA.__call__` wrapper is not usable on a single block because its
  `B.T[None, :, :]` operand expansion only matches the documented `[..., K]`
  contract when `n == k`.

---

# Model 2 — Hopper (SM90) FP8 wgmma

Bit-accurate SystemC model of the arithmetic datapath one Hopper tensor core
uses for the FP8 wgmma matrix multiply-accumulate:

```
wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3
wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3
wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2
wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e5m2
```

It computes one output element `d = c + sum_{k=0}^{31} a_k * b_k` the way the
hardware does, bit for bit.

Files:

| file | what it is |
|---|---|
| `hopper_fp8_wgmma.h` | the model: width-annotated `sc_int`/`sc_uint` datapath + 4-stage SystemC pipeline |
| `hopper_fp8_wgmma_tb.cpp` | testbench: golden vectors, pipeline-vs-combinational, random stimulus dump, `--walkthrough` |
| `verify_hopper_fp8_vs_mmasim.py` | differential test vs the MMA-Sim package (`WGMMA("Hopper", ...)`) |
| `hopper_fp8_coverage.cpp` | branch-coverage probe over a dump (`make hopper-coverage`) |
| `Makefile` | `make hopper`, `make hopper-run`, `make hopper-verify`, `make hopper-coverage`, `make hopper-walk` |

```shell
cd systemc
make hopper-run                     # golden vectors + pipeline check
make hopper-walk                    # annotated datapath trace, .f32 and .f16 accumulators
make hopper-verify HON=20000        # 2 x 20000 random blocks vs MMA-Sim
make hopper-coverage HON=20000      # which arms of the datapath the stimulus reaches
```

## 6. What the model computes

The accumulator `C` has the **same type as the output**: FP32 for the `.f32`
variants, FP16 for the `.f16` variants (`isa_wgmma.py` sets `c_type = d_type`,
and the hardware agrees — the `.f16` kernel has a `uint16_t` accumulator loaded
by `LOAD_D_M64N8_F16()`). The two variants therefore differ in the accumulator's
significand width (24 vs 11 bits) and subnormal floor (`2^-126` vs `2^-14`), not
just in the final rounding. In the model this is the `FC` template parameter.

Per `K = 32` instruction, for one output element (`C` participates early, in the
same fused summation — this is not a separate later add):

1. **MULTIPLY** — exact integer product of the two signed significands, and an
   integer sum of the exponents. The product is deliberately left
   **denormalised**: `sig_a * sig_b` reaches `15*15 = 225`, i.e. a significand in
   `[1, 4)`, which needs a second integer bit downstream. It is never
   renormalised.
2. **ALIGN** — a single alignment exponent `ne_max = max(exp + kfrac)` over all
   33 terms (32 products + `C`). Every term is shifted onto that one quantum and
   **truncated toward zero** to `F = 13` fractional bits. There is no guard
   digit, no sticky bit and no round-to-nearest: bits shifted out are simply
   discarded. This is the dominant error source of the instruction.
   Zero terms join at `e_zero = -133` (see §9).
3. **ACCUMULATE** — an exact sign-magnitude binary adder tree over the 33
   aligned terms, with no intermediate rounding anywhere. Because the leaves are
   exact integers on a common quantum and the only rounding is in step 4, the
   tree's associativity does not affect the result; the tree in the model is a
   structural illustration, not a numerical requirement.
4. **ROUND** — one single conversion of `S * 2^(ne_max - F)`:
   * `.f32` output → `rho = RZ-E8M13`: keep `1+13 = 14` significant bits,
     truncate toward zero, place in an FP32 container.
   * `.f16` output → `rho = RNE-FP16`: keep `1+10 = 11` significant bits,
     round to nearest even, place in FP16.

Verified parameter row (MMA-Sim, [arXiv:2511.10909](https://arxiv.org/abs/2511.10909),
`nv_ptx/sim.py` lines 94-118, `class WGMMA`):

| parameter | value |
|---|---|
| `F` | 13 |
| `L_max` | 32 (= `K`, so ONE fused node — no chaining) |
| `rho` | `RZ-E8M13` for `.f32`, `RNE-FP16` for `.f16` |
| `e_zero` | -133 |

**`K == L_max` is the single most important structural fact**: the whole 32-term
dot product is one fusion node, so there is no intermediate FP32 rounding inside
the K loop and no accumulation order to infer.

## 7. Signal widths

All datapath registers are `sc_int` / `sc_uint` with documented widths, derived
in `widths<>` and guarded by `static_assert`s (`MAX_ALIGNED < 2^ALIGNED`,
`MAX_SUM < 2^MAG`).

| signal | width | meaning |
|---|---|---|
| one FP8 significand `SIG` | `sc_uint<4>` / `sc_uint<3>` | 1 implicit + mantissa bits |
| exact product significand `PROD_SIG` | `sc_uint<8>` / `sc_uint<6>` | `15*15 = 225`, `7*7 = 49` |
| **term slot `TERM_SIG`** | `sc_uint<24>` / `sc_uint<11>` | must also hold the accumulator `C` significand |
| exponent `EXP` | `sc_int<10>` | `-149 .. +104` (a tiny/huge FP32 `C` dominates) |
| largest alignment shift `SHIFT_MAX` | (compile-time bound) | `F - kfrac`: **+7** E4M3, **+9** E5M2 |
| aligned term `ALIGNED` | `sc_uint<15>` | `< 225 * 2^(F-6) = 28800` |
| accumulator magnitude `MAG` | `sc_uint<20>` | `= F + 1 + ceil(log2(L+1))` |
| accumulator (sign-magnitude) | 20 + 1 sign | `MAX_SUM = 32*28800 + 16383 = 937983 < 2^20` |
| result | `sc_uint<32>` | FP32 bit pattern, or FP16 in `[15:0]` |

The adder is **20 magnitude bits**, and what actually justifies that width is the
bound the code asserts:

```
|S| < 2^F * (L * max_product_significand + max_aligned_C)
    = 2^13 * (32 * 3.515625 + 2)  =  2^13 * 114.5  <  2^20
```

so `MAX_SUM = 32*28800 + 16383 = 937983 < 2^20`. Only the leading-1 headroom and
the `ceil(log2(L+1))` carry bits are structural; `MAG = F + 1 + ceil(log2(L+1))`
is the same closed form the MXFP8 model uses. Do not read the `+1` as "the second
integer bit" on its own -- the real criterion is the bound above.

`TERM_SIG` is the one place the datapath is much wider than the FP8 products
alone would need: `C` converges into the same fused summation, so every term slot
must carry the accumulator's significand — **24 bits for a `.f32` accumulator,
11 bits for `.f16`** (`TERM_SIG = max(PROD_SIG, C_SIG)`).

## 8. Why the result is only 14 bits

`RZ-E8M13` is implemented in MMA-Sim's `ldexp_and_normalize` as: renormalise the
exact sum, truncate the significand toward zero to 13 fractional bits, then place
it in FP32. The exponent range is FP32's (the `-126` renormalisation boundary,
overflow to Inf), *not* an 8-bit range — the "E8M13" name describes the
significand truncation point, so:

* **"FP32 accumulate" on Hopper FP8 delivers 14 significant bits**, not 24.
* Rounding is **RZ**, i.e. a systematic bias rather than unbiased RNE.
* The same instruction's `.f16` output variant *does* round to nearest; only the
  `.f32` variant truncates.

That combination — a single truncating alignment point for 33 terms, plus a
14-bit result — is the structural reason behind the reported Hopper FP8
accumulation-accuracy problems, and it is exactly the `F = 13` vs `F = 25`
difference against the Blackwell MXFP8 model above.

## 9. `e_zero = -133` is inert for this instruction

MMA-Sim pins zero terms to `e_zero = -133` so they still participate in the
`max()`. On Hopper FP8 that cannot change any result. Two quantities are worth keeping
apart here:

* the smallest *value* a product can take is `2^-18` (E4M3) / `2^-32` (E5M2);
* the model's *term exponent* for such a product is `-12` / `-28`, because a
  subnormal FP8 input's `frexp` exponent is clamped to `-6` / `-14` rather than
  its true `-9` / `-16`, and a product doubles that.

Both are far above `-133`, so a zero term can never be the maximum, and every
term is zero only when the whole dot product is zero anyway.

`e_zero` becomes live only for the block-scaled variants (the MXFP8 model),
where an E8M0 scale can be as small as `2^-127` and push products below `-133`.
It is kept in the model because it is part of the verified parameter set.

## 10. Verification status

| check | result |
|---|---|
| golden vectors (24 fixed cases, see below) | 24/24 identical |
| 4-stage pipeline vs combinational model, 256 back-to-back blocks | 256/256 identical, latency 4 (lengths checked exactly -- a dropped beat fails the run) |
| SystemC vs MMA-Sim `WGMMA("Hopper", ...)`, e4m3 + e5m2, 40000 blocks | 40000/40000 identical |
| same, finite-only stimulus (stresses alignment truncation, no Inf/NaN) | 40000/40000 identical |

Branch coverage of the stimulus, from `make hopper-coverage` over the two 40000-
block dumps (each row counts blocks, summing e4m3 + e5m2). Note how little the
`mixed` dump contributes numerically -- 99.9% of its E5M2 blocks are collapsed by
a single Inf/NaN operand -- which is exactly why the `--finite` mode exists:

| path | `mixed` | `--finite` |
|---|---|---|
| blocks reaching the numeric path (f32 / f16 variant) | 1852 / 1844 | 40000 / 40000 |
| FP32 truncating the aligned sum (`drop > 0`) | 729 | 17316 |
| FP32 overflow → Inf | 0 | 0 |
| FP16 RNE actually rounding (`drop > 0`) | 1790 | 39074 |
| FP16 RNE carry out of the significand | 0 | 14 |
| FP16 overflow → Inf | 333 | 22279 |
| `S == 0` (all-zero / exact cancellation) | 0 | 43 |
| blocks with a finite FP32 result | 1852 | 40000 |
| blocks with a finite FP16 result | 1511 | 17721 |
| FP32 / FP16 **subnormal** output | 0 / 0 | 0 / 0 |

The two subnormal-output arms are rare enough that random stimulus never reaches
them (0 hits in 80000 blocks; the FP32 arm needs `ne_max <= -115`, which requires
a subnormal FP32 `C` many orders of magnitude below every product), so they are
pinned by golden vectors instead:

| golden | exercises |
|---|---|
| `C32 tiny subnormal` (`C32 = 2^-139`) | FP32 **subnormal output** → `0x00000400` |
| `C16 min subnormal` (`C16 = 2^-24`) | FP16 **subnormal output** → `0x0001` |
| `f16 RNE carry out of significand` (`C16 = 1.9990234375` + a `2^-11` product) | FP16 **carry** → `0x4000` |

Each variant is checked against its own accumulator type: `WGMMA(...f32...)` is
fed an FP32 `C` and `WGMMA(...f16...)` an FP16 `C`. Feeding an FP32 `C` to the
`.f16` variant instead — the bug described below — mispredicts 983/20000 e4m3
and 6/20000 e5m2 blocks, so this is not a cosmetic distinction.

Golden expected bits were produced by the MMA-Sim oracle itself, not by hand.

Four real bugs were found and fixed by this differential test, plus one in the
testbench itself. Each is worth recording because each is a plausible mistake:

1. **Sticky `±Inf` cancellation.** Counting infinities as `+1/-1` lets one `+Inf`
   and one `-Inf` sum to zero and slip through as a *finite* result. The signs
   must be tracked as two separate flags.
2. **Missing exponent compensation on a left shift.** In the `.f16` path the
   significand is shifted to 11 bits; when the value already has fewer than 11
   significant bits the shift is a *left* shift, and the scale must absorb it
   (`E_res = E_V + drop`) in that direction too — compensating only for the
   right-shift case silently scales the result by a power of two.
3. **A negative shift width when packing.** Clearing the low bits of the
   truncated magnitude leaves it spanning its original bit positions (up to 21
   bits), not 14. Extracting the 13 fractional bits must therefore select
   `At >> (msb-13)` when `msb > 13`, otherwise the shift count goes negative.
4. **The wrong accumulator type for the `.f16` variants.** The first version
   decoded `C` as FP32 unconditionally, and the verifier masked it by feeding
   FP32 `C` to both variants. Because `c_type = d_type`, the `.f16` variants
   actually accumulate into FP16: the significand is 11 bits rather than 24 and
   the subnormal floor is `2^-14` rather than `2^-126`. `FC` is now a template
   parameter (`f32` / `f16`) and `TERM_SIG` follows from it.
5. **A "finite" stimulus that was not finite.** In the testbench, `--finite`
   only guarded `C`, and the finite operand generator drew `e` from
   `below(2*max_e + 1)` -- an exponent *field* value above `max_e`, which the
   `uint8_t` cast folded back into a different field, occasionally all-ones. So
   ~1/60 of "finite" E5M2 operands were Inf/NaN, one special operand collapses
   the dot product, and the `.finite` E5M2 dump turned out to be 100% NaN
   propagation (11 finite results in 20000 blocks). `--finite` now rejects
   special bytes from every generator and the same dump yields 20000 finite
   results. A verification mode that silently tests nothing is worse than no
   mode at all, so the coverage table above is now part of the README.

## 11. Caveats

* Only ONE output element is modelled. Not modelled: the async wgmma
  issue/commit/wait machinery, shared-memory matrix descriptors, the 64x8 output
  tile / thread-data mapping, throughput, and the accumulator's round trip
  through registers between separate instructions. Those are data-movement
  concerns; the numerical behaviour of the instruction is fully captured by the
  datapath here.
* `e5m2` is modelled as IEEE-like (Inf when the exponent field is all ones and
  the mantissa is zero, NaN otherwise) and `e4m3` as OCP (no infinity; only
  `S.1111.111` is NaN, all other exponent-all-ones values are ordinary normals up
  to 448).
* MMA-Sim documents one open corner that this model inherits: for fp16/bf16
  inputs the alignment exponent is clamped from below, and the fp8 rows of that
  table list no such value, so non-zero products are not clamped here.
* The model is parameterised (`F`, `L`, `E_ZERO`, formats), but `F = 13`,
  `L = 32`, `e_zero = -133` are the verified Hopper FP8 operating point; a
  different point is not hardware-checked.

