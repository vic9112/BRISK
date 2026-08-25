SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

-include local.mk

PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1
HLS_FREQ_MHZ ?= 300
KERNEL_FREQ_MHZ ?= 220
JOBS ?= 8
VITIS_SETUP ?=
XRT_SETUP ?=
PYTHON ?= python3
MODEL_DIR ?=
DEVICE ?= 0
BUILD_DIR ?= $(CURDIR)/build
XCLBIN ?= $(BUILD_DIR)/llama3_fast_attention.hw.xclbin

SOURCE_DIR := $(CURDIR)/source
TCL_DIR := $(SOURCE_DIR)/tcl
XO := $(BUILD_DIR)/llama3_fast_attention.xo
ROUTE_STAMP := $(BUILD_DIR)/.route-complete
DEMO_XCLBIN := $(BUILD_DIR)/.demo-xclbin-ready
TEMP_DIR := $(BUILD_DIR)/temp
LOG_DIR := $(BUILD_DIR)/logs
REPORT_DIR := $(BUILD_DIR)/reports

VIVADO_ARGS := \
	--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_BalanceSLRs \
	--vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=NoTimingRelaxation \
	--vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$(TCL_DIR)/kernel_clock_override.tcl \
	--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=$(TCL_DIR)/lane_floorplan.tcl \
	--vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.TCL.POST=$(TCL_DIR)/slr_physopt.tcl \
	--vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.TCL.PRE=$(TCL_DIR)/route_pre.tcl \
	--vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST=$(TCL_DIR)/route_post.tcl

PROFILE_ARGS := \
	--profile.exec all:all:counters \
	--profile.data llama3_fast_attention:llama3_fast_attention_1:cache_kv0:counters \
	--profile.data llama3_fast_attention:llama3_fast_attention_1:cache_kv1:counters \
	--profile.data llama3_fast_attention:llama3_fast_attention_1:cache_kv2:counters \
	--profile.data llama3_fast_attention:llama3_fast_attention_1:cache_kv3:counters

HLS_SOURCES := $(wildcard $(SOURCE_DIR)/*.hpp) $(SOURCE_DIR)/attention_kernel.cpp
TCL_SOURCES := $(wildcard $(TCL_DIR)/*.tcl)

.PHONY: all hls demo clean $(DEMO_XCLBIN)

all: $(XCLBIN)

hls: $(XO)

$(XO): $(HLS_SOURCES) Makefile
	if [[ -n "$(VITIS_SETUP)" ]]; then source "$(VITIS_SETUP)"; fi
	mkdir -p "$(BUILD_DIR)" "$(TEMP_DIR)/compile" "$(LOG_DIR)/compile" "$(REPORT_DIR)/compile"
	v++ -c -t hw \
	  --platform "$(PLATFORM)" \
	  -k llama3_fast_attention \
	  --kernel_frequency "$(HLS_FREQ_MHZ)" \
	  -I "$(SOURCE_DIR)" \
	  -DFATTN_USE_HLS_AP_UINT=1 \
	  --hls.jobs "$(JOBS)" \
	  --temp_dir "$(TEMP_DIR)/compile" \
	  --log_dir "$(LOG_DIR)/compile" \
	  --report_dir "$(REPORT_DIR)/compile" \
	  -o "$@" \
	  "$(SOURCE_DIR)/attention_kernel.cpp"

$(ROUTE_STAMP): $(XO) Makefile $(SOURCE_DIR)/link.cfg $(TCL_SOURCES)
	if [[ -n "$(VITIS_SETUP)" ]]; then source "$(VITIS_SETUP)"; fi
	mkdir -p "$(TEMP_DIR)/link" "$(LOG_DIR)/link" "$(REPORT_DIR)/link"
	KERNEL_FREQUENCY_MHZ="$(KERNEL_FREQ_MHZ)" \
	FPT26_VITIS_BUILD_DIR="$(BUILD_DIR)" \
	v++ -l -t hw \
	  --platform "$(PLATFORM)" \
	  --kernel_frequency "$(KERNEL_FREQ_MHZ)" \
	  --config "$(SOURCE_DIR)/link.cfg" \
	  --vivado.synth.jobs "$(JOBS)" \
	  --vivado.impl.jobs "$(JOBS)" \
	  --temp_dir "$(TEMP_DIR)/link" \
	  --log_dir "$(LOG_DIR)/link" \
	  --report_dir "$(REPORT_DIR)/link" \
	  --save-temps \
	  --to_step vpl.impl.route_design \
	  $(PROFILE_ARGS) \
	  $(VIVADO_ARGS) \
	  -o "$(XCLBIN)" \
	  "$(XO)"
	touch "$@"

$(XCLBIN): $(ROUTE_STAMP)
	if [[ -n "$(VITIS_SETUP)" ]]; then source "$(VITIS_SETUP)"; fi
	KERNEL_FREQUENCY_MHZ="$(KERNEL_FREQ_MHZ)" \
	FPT26_VITIS_BUILD_DIR="$(BUILD_DIR)" \
	v++ -l -t hw \
	  --platform "$(PLATFORM)" \
	  --kernel_frequency "$(KERNEL_FREQ_MHZ)" \
	  --config "$(SOURCE_DIR)/link.cfg" \
	  --vivado.synth.jobs "$(JOBS)" \
	  --vivado.impl.jobs "$(JOBS)" \
	  --temp_dir "$(TEMP_DIR)/link" \
	  --log_dir "$(LOG_DIR)/link" \
	  --report_dir "$(REPORT_DIR)/link" \
	  --save-temps \
	  --from_step vpl.impl.write_bitstream \
	  $(PROFILE_ARGS) \
	  $(VIVADO_ARGS) \
	  -o "$@" \
	  "$(XO)"
	test -s "$@"

$(DEMO_XCLBIN):
	test -f "$(XCLBIN)" || { echo "local demo xclbin not found: $(XCLBIN); build it with 'make' or set XCLBIN to a local artifact" >&2; exit 2; }

demo: $(DEMO_XCLBIN)
	test -n "$(MODEL_DIR)" || { echo "MODEL_DIR must point to local Llama3-8B weights" >&2; exit 2; }
	if [[ -n "$(XRT_SETUP)" ]]; then source "$(XRT_SETUP)"; fi
	cd demo
	FPT26_MODEL_DIR="$(MODEL_DIR)" \
	FPT26_XCLBIN="$(XCLBIN)" \
	FPT26_DEVICE="$(DEVICE)" \
	"$(PYTHON)" -m jupyter lab demo.ipynb

clean:
	test -n "$(BUILD_DIR)"
	rm -rf -- "$(BUILD_DIR)" "$(CURDIR)/demo/__pycache__" "$(CURDIR)/demo/.ipynb_checkpoints"
