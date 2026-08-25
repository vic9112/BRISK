# Optimize the measured ap_sync_ready control paths that cross between the
# lane-local cache-update dataflows on SLR0 and SLR1. This hook is source- and
# ABI-neutral and runs after the standard post-place phys_opt pass.

set fpt26_kernel_clock [get_clocks -quiet clk_kernel_00_unbuffered_net]
if {[llength $fpt26_kernel_clock] != 1} {
  error "FPT26 SLR control physopt expected one clk_kernel_00_unbuffered_net"
}

puts "FPT26 SLR control physopt: starting -slr_crossing_opt -tns_cleanup"
phys_opt_design -slr_crossing_opt -tns_cleanup
puts "FPT26 SLR control physopt: completed"
