SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

-include local.mk

PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1
HLS_FREQ_MHZ ?= 220
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

SOURCE_DIR := $(CURDIR)/source
TCL_DIR := $(SOURCE_DIR)/tcl
BUILD_DIR_ABS := $(abspath $(BUILD_DIR))
BUILD_SENTINEL := $(BUILD_DIR)/.brisk-build-owner
BUILD_SENTINEL_CONTENT := BRISK_BUILD_OWNER=$(abspath $(CURDIR));BUILD_DIR=$(BUILD_DIR_ABS)
XO := $(BUILD_DIR)/llama3_fast_attention.xo
HLS_CONFIG := $(SOURCE_DIR)/hls_config.cfg
HLS_WORK_DIR := $(BUILD_DIR)/hls_component
ADAPTER_CONFIG := $(CURDIR)/verification/hls_adapter_config.cfg
ADAPTER_WORK_DIR := $(BUILD_DIR)/hls_adapter
IDENTITY_CONFIG := $(CURDIR)/verification/hls_identity_config.cfg
IDENTITY_WORK_DIR := $(BUILD_DIR)/hls_identity
ADAPTER_CHECK := $(CURDIR)/verification/check_adapter_rtl.py
COSIM_STAMP := $(BUILD_DIR)/.cosim-complete
# Keep each independent stateful sequence intact; XSim preloads all per-call
# backing pages, so a monolithic run can exceed the verification host's RAM.
COSIM_SUITES := numeric paper residency isolation identity validation
ROUTE_STAMP := $(BUILD_DIR)/.route-complete
TEMP_DIR := $(BUILD_DIR)/temp
LOG_DIR := $(BUILD_DIR)/logs
REPORT_DIR := $(BUILD_DIR)/reports
ROUTED_CLOCK_REPORT ?= $(BUILD_DIR)/report/hw/link/imp/impl_1_hw_bb_locked_timing_summary_post_route_physopt.rpt

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

HLS_SOURCES := $(wildcard $(SOURCE_DIR)/*.hpp) $(SOURCE_DIR)/attention_kernel.cpp \
	$(wildcard $(CURDIR)/spec/*.hpp)
TCL_SOURCES := $(wildcard $(TCL_DIR)/*.tcl)

.PHONY: all hls hls-csim hls-cosim vivado demo qualify clean prepare-build-dir

all: $(XCLBIN)

hls: $(XO)

vivado: $(XCLBIN)

hls-csim: prepare-build-dir
	if [[ -n "$(VITIS_SETUP)" ]]; then set +u; source "$(VITIS_SETUP)"; set -u; fi
	vitis-run --mode hls --config "$(HLS_CONFIG)" \
	  --work_dir "$(HLS_WORK_DIR)" --freqhz "$(HLS_FREQ_MHZ)Mhz" --csim

hls-cosim: $(COSIM_STAMP)

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

$(XO): $(HLS_SOURCES) verification/brisk_hls_tb.cpp $(HLS_CONFIG) Makefile | prepare-build-dir
	if [[ -n "$(VITIS_SETUP)" ]]; then set +u; source "$(VITIS_SETUP)"; set -u; fi
	mkdir -p "$(BUILD_DIR)" "$(TEMP_DIR)/compile" "$(LOG_DIR)/compile" "$(REPORT_DIR)/compile"
	vitis-run --mode hls --config "$(HLS_CONFIG)" \
	  --work_dir "$(HLS_WORK_DIR)" --freqhz "$(HLS_FREQ_MHZ)Mhz" --csim
	v++ --compile --mode hls --config "$(HLS_CONFIG)" \
	  --work_dir "$(HLS_WORK_DIR)" --freqhz "$(HLS_FREQ_MHZ)Mhz"
	vitis-run --mode hls --config "$(HLS_CONFIG)" \
	  --work_dir "$(HLS_WORK_DIR)" --freqhz "$(HLS_FREQ_MHZ)Mhz" \
	  --hls.package.output.file "$@" --package
	test -s "$@"

$(COSIM_STAMP): $(XO) verification/brisk_hls_tb.cpp $(HLS_CONFIG) $(ADAPTER_CONFIG) $(IDENTITY_CONFIG) $(ADAPTER_CHECK) Makefile | prepare-build-dir
	if [[ -n "$(VITIS_SETUP)" ]]; then set +u; source "$(VITIS_SETUP)"; set -u; fi
	v++ --compile --mode hls --config "$(ADAPTER_CONFIG)" \
	  --work_dir "$(ADAPTER_WORK_DIR)" --freqhz "$(HLS_FREQ_MHZ)Mhz" \
	  --hls.package.output.file "$(BUILD_DIR)/llama3_fast_attention.adapter.xo"
	$(PYTHON) "$(ADAPTER_CHECK)" --production "$(HLS_WORK_DIR)" \
	  --adapter "$(ADAPTER_WORK_DIR)" --report "$(REPORT_DIR)/cosim/rtl-identity.json"
	v++ --compile --mode hls --config "$(IDENTITY_CONFIG)" \
	  --work_dir "$(IDENTITY_WORK_DIR)" --freqhz "$(HLS_FREQ_MHZ)Mhz" \
	  --hls.package.output.file "$(BUILD_DIR)/llama3_fast_attention.identity.xo"
	$(PYTHON) "$(ADAPTER_CHECK)" --production "$(HLS_WORK_DIR)" \
	  --adapter "$(IDENTITY_WORK_DIR)" --report "$(REPORT_DIR)/cosim/identity-rtl-identity.json"
	for suite in $(COSIM_SUITES); do
	  config="$(ADAPTER_CONFIG)"
	  component="$(ADAPTER_WORK_DIR)"
	  if [[ "$$suite" == identity ]]; then
	    config="$(IDENTITY_CONFIG)"
	    component="$(IDENTITY_WORK_DIR)"
	  fi
	  evidence="$(REPORT_DIR)/cosim/$$suite"
	  mkdir -p "$$evidence/dataflow" "$$evidence/native-report"
	  vitis-run --mode hls --config "$$config" \
	    --work_dir "$$component" --freqhz "$(HLS_FREQ_MHZ)Mhz" \
	    --hls.cosim.argv="--suite $$suite" --cosim 2>&1 | tee "$$evidence/cosim.log"
	  cp -a "$$component/hls/sim/report/." "$$evidence/native-report/"
	  find "$$component/hls/.autopilot/db" -maxdepth 1 -name '*.csv' \
	    -exec cp -t "$$evidence/dataflow" {} +
	  for profile in process_stalling_info channel_depth_info loop_performance_info CosimGUI; do
	    if [[ -d "$$component/hls/.autopilot/db/$$profile" ]]; then
	      cp -a "$$component/hls/.autopilot/db/$$profile" "$$evidence/dataflow/"
	    fi
	  done
	done
	touch "$@"

$(ROUTE_STAMP): $(XO) $(COSIM_STAMP) Makefile $(SOURCE_DIR)/link.cfg $(TCL_SOURCES) | prepare-build-dir
	if [[ -n "$(VITIS_SETUP)" ]]; then set +u; source "$(VITIS_SETUP)"; set -u; fi
	mkdir -p "$(TEMP_DIR)/link" "$(LOG_DIR)/link" "$(REPORT_DIR)/link"
	KERNEL_FREQUENCY_MHZ="$(KERNEL_FREQ_MHZ)" \
	BRISK_BUILD_DIR="$(BUILD_DIR)" \
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
	if [[ -n "$(VITIS_SETUP)" ]]; then set +u; source "$(VITIS_SETUP)"; set -u; fi
	KERNEL_FREQUENCY_MHZ="$(KERNEL_FREQ_MHZ)" \
	BRISK_BUILD_DIR="$(BUILD_DIR)" \
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
	if [[ "$(XCLBIN_WAS_SET)" = "0" ]]; then
	  $(MAKE) --no-print-directory all
	fi
	test -s "$(XCLBIN)" || { echo "explicit XCLBIN not found: $(XCLBIN)" >&2; exit 2; }
	if [[ -n "$(XRT_SETUP)" ]]; then set +u; source "$(XRT_SETUP)"; set -u; fi
	brisk_demo_dir="$(BUILD_DIR)/demo/$$(date +%Y%m%d-%H%M%S-%N)"
	"$(PYTHON)" demo/run_notebook.py --model-dir "$(MODEL_DIR)" \
	  --xclbin "$(XCLBIN)" --device "$(DEVICE)" --output "$$brisk_demo_dir"

qualify: prepare-build-dir
	test -n "$(MODEL_DIR)" || { echo "MODEL_DIR must point to local Meta Llama3-8B weights" >&2; exit 2; }
	if [[ "$(XCLBIN_WAS_SET)" = "0" ]]; then
	  $(MAKE) --no-print-directory all
	fi
	test -s "$(XCLBIN)"
	test -s "$(ROUTED_CLOCK_REPORT)"
	if [[ -n "$(XRT_SETUP)" ]]; then set +u; source "$(XRT_SETUP)"; set -u; fi
	brisk_qualification_dir="$(BUILD_DIR)/qualification/$$(date +%Y%m%d-%H%M%S-%N)"
	brisk_artifact_hash="$$(sha256sum "$(XCLBIN)" | awk '{print $$1}')"
	"$(PYTHON)" demo/qualify_u50.py --model-dir "$(MODEL_DIR)" \
	  --xclbin "$(XCLBIN)" --sha256 "$$brisk_artifact_hash" --device "$(DEVICE)" \
	  --routed-clock-mhz "$(KERNEL_FREQ_MHZ)" --routed-clock-report "$(ROUTED_CLOCK_REPORT)" \
	  --report-dir "$$brisk_qualification_dir"

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
