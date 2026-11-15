# Correct the U50 Vitis 2025.2 user-kernel generated-clock constraint.
#
# The link request remains owned by v++ --kernel_frequency. This pre-opt hook
# only repairs the generated-clock ratio when the platform helper emits 1:1
# for a non-100-MHz request. The routed report remains the signoff authority.

proc brisk_gcd {a b} {
    while {$b != 0} {
        set remainder [expr {$a % $b}]
        set a $b
        set b $remainder
    }
    return $a
}

if {![info exists ::env(KERNEL_FREQUENCY_MHZ)]} {
    error "BRISK clock hook requires KERNEL_FREQUENCY_MHZ from the Makefile"
}
set requested_mhz [expr {int($::env(KERNEL_FREQUENCY_MHZ))}]

if {$requested_mhz <= 0} {
    error "KERNEL_FREQUENCY_MHZ must be a positive integer"
}

set clock_root "level0_i/ulp/ulp_ucs/inst/aclk_kernel_00_hierarchy/clkwiz_aclk_kernel_00/inst/CLK_CORE_DRP_I/clk_inst/mmcme4_adv_inst"
set source_pin [get_pins -quiet "${clock_root}/CLKIN1"]
set output_pin [get_pins -quiet "${clock_root}/CLKOUT0"]
if {[llength $source_pin] != 1 || [llength $output_pin] != 1} {
    error "U50 kernel clock source/output pins are not unique"
}

set source_clocks [get_clocks -quiet -of_objects $source_pin]
if {[llength $source_clocks] != 1} {
    error "U50 kernel clock source clock is not unique"
}
set source_period_ns [get_property PERIOD [lindex $source_clocks 0]]
set source_mhz [expr {int(round(1000.0 / $source_period_ns))}]
set divisor [brisk_gcd $requested_mhz $source_mhz]
set multiply_by [expr {$requested_mhz / $divisor}]
set divide_by [expr {$source_mhz / $divisor}]

# Vivado 2025.2 has no clock-removal command.  Re-creating a generated clock
# with the same name is the supported overwrite path used by the platform's
# own _user_impl_clk.xdc file.
create_generated_clock \
    -name clk_kernel_00_unbuffered_net \
    -source $source_pin \
    -multiply_by $multiply_by \
    -divide_by $divide_by \
    $output_pin

set expected_period_ns [expr {1000.0 / $requested_mhz}]
set routed_constraint_period_ns [get_property PERIOD \
    [get_clocks clk_kernel_00_unbuffered_net]]
if {abs($routed_constraint_period_ns - $expected_period_ns) > 0.001} {
    error [format \
        "kernel clock override mismatch: requested %.3f ns, constrained %.3f ns" \
        $expected_period_ns $routed_constraint_period_ns]
}
puts [format \
    "BRISK kernel clock override: source %.3f ns, ratio %d/%d, requested %.3f ns" \
    $source_period_ns $multiply_by $divide_by $expected_period_ns]
