# Run one attributable post-route setup repair only when setup timing is open.
set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
if {[llength $setup_paths] != 1} {
  error "BRISK route post expected one worst setup path"
}

set setup_slack [get_property SLACK [lindex $setup_paths 0]]
puts [format "BRISK route post: post-route setup slack %.3f ns" $setup_slack]
set phys_opt_effective_floor_ns -0.5
if {$setup_slack < 0.0 && $setup_slack >= $phys_opt_effective_floor_ns} {
  puts "BRISK route post: running phys_opt_design -directive AggressiveExplore"
  phys_opt_design -directive AggressiveExplore

  set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
  set setup_slack [get_property SLACK [lindex $setup_paths 0]]
  puts [format "BRISK route post: setup slack after AggressiveExplore %.3f ns" \
    $setup_slack]
  if {$setup_slack < 0.0 && $setup_slack >= $phys_opt_effective_floor_ns} {
    puts "BRISK route post: running phys_opt_design -directive AggressiveFanoutOpt"
    phys_opt_design -directive AggressiveFanoutOpt
  }

  set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
  set setup_slack [get_property SLACK [lindex $setup_paths 0]]
  puts [format "BRISK route post: setup slack after fanout repair %.3f ns" \
    $setup_slack]
  if {$setup_slack < 0.0 && $setup_slack >= $phys_opt_effective_floor_ns} {
    puts "BRISK route post: running phys_opt_design -directive AlternateReplication"
    phys_opt_design -directive AlternateReplication
  }
} elseif {$setup_slack < $phys_opt_effective_floor_ns} {
  puts [format \
    "BRISK route post: skipping ineffective phys_opt_design at %.3f ns" \
    $setup_slack]
}

if {![info exists ::env(BRISK_BUILD_DIR)]} {
  error "BRISK route post requires BRISK_BUILD_DIR from the Makefile"
}
set final_report [file normalize [file join \
  $::env(BRISK_BUILD_DIR) report hw link imp \
  impl_1_hw_bb_locked_timing_summary_post_route_physopt.rpt]]
file mkdir [file dirname $final_report]
if {[file exists $final_report]} {
  file delete -force $final_report
}
report_timing_summary -delay_type min_max -max_paths 10 \
  -file $final_report
puts "BRISK route post: final timing report $final_report"

# Do not let a completed route step authorize bitstream generation when timing
# is still open. Re-read both checks after any post-route optimization above.
set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1 -nworst 1]
if {[llength $setup_paths] != 1 || [llength $hold_paths] != 1} {
  error "BRISK route post requires final setup and hold paths"
}
set setup_slack [get_property SLACK [lindex $setup_paths 0]]
set hold_slack [get_property SLACK [lindex $hold_paths 0]]
if {$setup_slack < 0.0 || $hold_slack < 0.0} {
  error [format "BRISK routed timing is open: setup %.3f ns, hold %.3f ns" \
    $setup_slack $hold_slack]
}

if {![info exists ::env(KERNEL_FREQUENCY_MHZ)]} {
  error "BRISK route post requires the requested kernel frequency"
}
set kernel_clock [get_clocks -quiet clk_kernel_00_unbuffered_net]
if {[llength $kernel_clock] != 1} {
  error "BRISK routed kernel clock is not unique"
}
set expected_period [expr {1000.0 / $::env(KERNEL_FREQUENCY_MHZ)}]
set routed_period [get_property PERIOD $kernel_clock]
if {abs($routed_period - $expected_period) > 0.001} {
  error [format "BRISK routed clock mismatch: requested %.3f ns, actual %.3f ns" \
    $expected_period $routed_period]
}
puts [format "BRISK route accepted: setup %.3f ns, hold %.3f ns, kernel %.3f ns" \
  $setup_slack $hold_slack $routed_period]
