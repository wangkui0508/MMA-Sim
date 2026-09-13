# Blackwell / B300 MXFP8 fused 32-element dot product — SystemC model

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
