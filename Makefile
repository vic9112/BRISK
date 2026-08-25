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
XCLBIN_WAS_SET := $(if $(filter undefined,$(origin XCLBIN)),0,1)
XCLBIN ?= $(BUILD_DIR)/llama3_fast_attention.hw.xclbin

RELEASE_TAG := v1.0.0
RELEASE_ASSET := BRISK-U50-v1.0.0.hw.xclbin
RELEASE_XCLBIN := $(BUILD_DIR)/$(RELEASE_ASSET)
RELEASE_URL := https://github.com/vic9112/BRISK/releases/download/$(RELEASE_TAG)/$(RELEASE_ASSET)
RELEASE_SHA256 := d928f26881fe59588ff7481f0945fcd857a276dffbe6b0f5dbfd12690de44ded

SOURCE_DIR := $(CURDIR)/source
TCL_DIR := $(SOURCE_DIR)/tcl
BUILD_DIR_ABS := $(abspath $(BUILD_DIR))
BUILD_SENTINEL := $(BUILD_DIR)/.brisk-build-owner
BUILD_SENTINEL_CONTENT := BRISK_BUILD_OWNER=$(abspath $(CURDIR));BUILD_DIR=$(BUILD_DIR_ABS)
XO := $(BUILD_DIR)/llama3_fast_attention.xo
ROUTE_STAMP := $(BUILD_DIR)/.route-complete
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

.PHONY: all hls demo clean prepare-build-dir

all: $(XCLBIN)

hls: $(XO)

prepare-build-dir:
	test "$(BUILD_DIR_ABS)" != "/" || { echo "refusing to use filesystem root as BUILD_DIR" >&2; exit 2; }
	test "$(BUILD_DIR_ABS)" != "$(abspath $(CURDIR))" || { echo "refusing to use repository root as BUILD_DIR" >&2; exit 2; }
	if [[ -e "$(BUILD_DIR)" ]]; then
	  test -d "$(BUILD_DIR)" || { echo "refusing non-directory BUILD_DIR: $(BUILD_DIR)" >&2; exit 2; }
	  test -f "$(BUILD_SENTINEL)" || { echo "refusing sentinel-less build directory: $(BUILD_DIR)" >&2; exit 2; }
	  test "$$(cat "$(BUILD_SENTINEL)")" = "$(BUILD_SENTINEL_CONTENT)" || { echo "refusing build directory with a foreign sentinel: $(BUILD_DIR)" >&2; exit 2; }
	else
	  mkdir -p "$(BUILD_DIR)"
	  printf '%s\n' "$(BUILD_SENTINEL_CONTENT)" > "$(BUILD_SENTINEL)"
	fi

$(XO): $(HLS_SOURCES) Makefile | prepare-build-dir
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

$(ROUTE_STAMP): $(XO) Makefile $(SOURCE_DIR)/link.cfg $(TCL_SOURCES) | prepare-build-dir
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

$(XCLBIN): $(ROUTE_STAMP) | prepare-build-dir
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

demo: prepare-build-dir
	test -n "$(MODEL_DIR)" || { echo "MODEL_DIR must point to local Llama3-8B weights" >&2; exit 2; }
	demo_xclbin="$(XCLBIN)"
	if [[ ! -f "$$demo_xclbin" ]]; then
	  if [[ "$(XCLBIN_WAS_SET)" = "1" ]]; then
	    echo "explicit XCLBIN not found: $$demo_xclbin" >&2
	    exit 2
	  fi
	  demo_xclbin="$(RELEASE_XCLBIN)"
	  if [[ ! -f "$$demo_xclbin" ]]; then
	    tmp="$$(mktemp "$(BUILD_DIR)/.$(RELEASE_ASSET).XXXXXX")"
	    trap 'rm -f -- "$$tmp"' EXIT
	    if command -v curl >/dev/null 2>&1; then
	      curl -fL --retry 3 -o "$$tmp" "$(RELEASE_URL)"
	    elif command -v wget >/dev/null 2>&1; then
	      wget -O "$$tmp" "$(RELEASE_URL)"
	    else
	      echo "make demo requires curl or wget to fetch $(RELEASE_ASSET)" >&2
	      exit 127
	    fi
	    printf '%s  %s\n' "$(RELEASE_SHA256)" "$$tmp" | sha256sum -c -
	    mv -- "$$tmp" "$$demo_xclbin"
	    trap - EXIT
	  fi
	  printf '%s  %s\n' "$(RELEASE_SHA256)" "$$demo_xclbin" | sha256sum -c -
	fi
	if [[ -n "$(XRT_SETUP)" ]]; then source "$(XRT_SETUP)"; fi
	cd demo
	FPT26_MODEL_DIR="$(MODEL_DIR)" \
	FPT26_XCLBIN="$$demo_xclbin" \
	FPT26_DEVICE="$(DEVICE)" \
	"$(PYTHON)" -m jupyter lab demo.ipynb

clean:
	test -n "$(BUILD_DIR)"
	if [[ -e "$(BUILD_DIR)" ]]; then
	  test "$(BUILD_DIR_ABS)" != "/" || { echo "refusing to clean filesystem root" >&2; exit 2; }
	  test "$(BUILD_DIR_ABS)" != "$(abspath $(CURDIR))" || { echo "refusing to clean repository root" >&2; exit 2; }
	  test -f "$(BUILD_SENTINEL)" || { echo "refusing to clean sentinel-less build directory: $(BUILD_DIR)" >&2; exit 2; }
	  test "$$(cat "$(BUILD_SENTINEL)")" = "$(BUILD_SENTINEL_CONTENT)" || { echo "refusing to clean build directory with a foreign sentinel: $(BUILD_DIR)" >&2; exit 2; }
	  rm -rf -- "$(BUILD_DIR)"
	fi
	rm -rf -- "$(CURDIR)/demo/__pycache__" "$(CURDIR)/demo/.ipynb_checkpoints"
