# SystemC testbench for the Blackwell/B300 MXFP8 fused 32-element dot product.
#
#   make            build ./mxfp8_stfdpa_tb
#   make run        golden vectors + pipeline check
#   make verify     differential test against the Python reference model
#   make clean

SYSTEMC_HOME ?= /usr/local
CXX          ?= clang++
CXXFLAGS     ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
INC           = -I$(SYSTEMC_HOME)/include -I.
LIB           = -L$(SYSTEMC_HOME)/lib -lsystemc
RPATH         = -Wl,-rpath,$(SYSTEMC_HOME)/lib

TB      = mxfp8_stfdpa_tb
DUMP    = mxfp8_dump.txt
N      ?= 4000

all: $(TB)

$(TB): $(TB).cpp mxfp8_stfdpa.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< $(LIB) $(RPATH)

run: $(TB)
	./$(TB)

dump: $(TB)
	./$(TB) --dump $(N) --fmt e4m3 --seed 0x1234567 > $(DUMP)
	./$(TB) --dump $(N) --fmt e5m2 --seed 0x89ABCDE > $(DUMP).e5m2
	./$(TB) --dump $(N) --fmt mixed --seed 0x5EED    > $(DUMP).mixed

verify: dump
	python3 verify_vs_reference.py $(DUMP) $(DUMP).e5m2 $(DUMP).mixed

clean:
	rm -f $(TB) $(DUMP) $(DUMP).e5m2 $(DUMP).mixed

.PHONY: all run dump verify clean
