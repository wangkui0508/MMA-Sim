# SystemC testbenches for NVIDIA tensor-core fused dot-product datapaths.
#
# 1) Blackwell / B300 MXFP8 (SM100 / SM103)
#      tcgen05.mma.kind::mxf8f6f4.block_scale / mma.sync kind::mxf8f6f4
#
#      make                build both testbenches
#      make run            golden vectors + pipeline check
#      make verify         differential test vs the Python reference model
#      make dump           emit N random blocks for the verifiers
#
# 2) Hopper FP8 (SM90)
#      wgmma.mma_async.sync.aligned.m64n8k32.{f32,f16}.{e4m3,e5m2}
#
#      make hopper         build ./hopper_fp8_wgmma_tb
#      make hopper-run     golden vectors + pipeline check
#      make hopper-verify  differential test vs the MMA-Sim package
#      make hopper-walk    annotated single-block datapath trace
#
#      make clean

SYSTEMC_HOME ?= /usr/local
CXX          ?= clang++
CXXFLAGS     ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
INC           = -I$(SYSTEMC_HOME)/include -I.
LIB           = -L$(SYSTEMC_HOME)/lib -lsystemc
RPATH         = -Wl,-rpath,$(SYSTEMC_HOME)/lib

MXTB    = mxfp8_stfdpa_tb
HOTB    = hopper_fp8_wgmma_tb
HOCOV   = hopper_fp8_coverage
DUMP    = mxfp8_dump.txt
HODUMP  = hopper_dump.txt
N      ?= 4000
HON    ?= 20000

all: $(MXTB) $(HOTB) $(HOCOV)

$(MXTB): $(MXTB).cpp mxfp8_stfdpa.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< $(LIB) $(RPATH)

$(HOTB): $(HOTB).cpp hopper_fp8_wgmma.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< $(LIB) $(RPATH)

$(HOCOV): $(HOCOV).cpp hopper_fp8_wgmma.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< $(LIB) $(RPATH)

# --- Blackwell / B300 MXFP8 -------------------------------------------------
run: $(MXTB)
	./$(MXTB)

dump: $(MXTB)
	./$(MXTB) --dump $(N) --fmt e4m3 --seed 0x1234567 > $(DUMP)
	./$(MXTB) --dump $(N) --fmt e5m2 --seed 0x89ABCDE > $(DUMP).e5m2
	./$(MXTB) --dump $(N) --fmt mixed --seed 0x5EED    > $(DUMP).mixed

verify: dump
	python3 verify_vs_reference.py $(DUMP) $(DUMP).e5m2 $(DUMP).mixed

# --- Hopper FP8 -------------------------------------------------------------
hopper: $(HOTB)

hopper-run: $(HOTB)
	./$(HOTB)

hopper-walk: $(HOTB)
	./$(HOTB) --walkthrough
	@echo "--- same block, .f16 accumulator ---"
	./$(HOTB) --walkthrough --acc-f16

hopper-dump: $(HOTB)
	./$(HOTB) --dump $(HON) --fmt mixed --seed 0x1234567        > $(HODUMP)
	./$(HOTB) --dump $(HON) --fmt mixed --seed 0x5EED --finite  > $(HODUMP).finite

# The dump is self-describing: verify_hopper_fp8_vs_mmasim.py parses the
# "# fmt=... F=... L=... e_zero=..." header and ASSERTS it against the operating
# point WGMMA("Hopper", ...) uses, so a dump from a different configuration fails
# loudly instead of being silently mis-verified.
hopper-verify: hopper-dump
	python3 verify_hopper_fp8_vs_mmasim.py $(HON) $(HODUMP) $(HODUMP).finite

# Bit-exactness means nothing on blocks that never reach the datapath, so this
# prints how many blocks actually exercise each arm.
hopper-coverage: hopper-dump $(HOCOV)
	./$(HOCOV) $(HODUMP) $(HODUMP).finite

clean:
	rm -f $(MXTB) $(HOTB) $(HOCOV) $(DUMP) $(DUMP).e5m2 $(DUMP).mixed $(HODUMP) $(HODUMP).finite

.PHONY: all run dump verify hopper hopper-run hopper-walk hopper-dump hopper-verify \
        hopper-coverage clean
