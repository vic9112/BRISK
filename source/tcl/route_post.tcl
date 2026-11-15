# Run one attributable post-route setup repair only when setup timing is open.
set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
if {[llength $setup_paths] != 1} {
  error "FPT26 route post expected one worst setup path"
}

set setup_slack [get_property SLACK [lindex $setup_paths 0]]
puts [format "FPT26 route post: post-route setup slack %.3f ns" $setup_slack]
set phys_opt_effective_floor_ns -0.5
if {$setup_slack < 0.0 && $setup_slack >= $phys_opt_effective_floor_ns} {
  puts "FPT26 route post: running phys_opt_design -directive AggressiveExplore"
  phys_opt_design -directive AggressiveExplore

  set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
  set setup_slack [get_property SLACK [lindex $setup_paths 0]]
  puts [format "FPT26 route post: setup slack after AggressiveExplore %.3f ns" \
    $setup_slack]
  if {$setup_slack < 0.0 && $setup_slack >= $phys_opt_effective_floor_ns} {
    puts "FPT26 route post: running phys_opt_design -directive AggressiveFanoutOpt"
    phys_opt_design -directive AggressiveFanoutOpt
  }

  set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
  set setup_slack [get_property SLACK [lindex $setup_paths 0]]
  puts [format "FPT26 route post: setup slack after fanout repair %.3f ns" \
    $setup_slack]
  if {$setup_slack < 0.0 && $setup_slack >= $phys_opt_effective_floor_ns} {
    puts "FPT26 route post: running phys_opt_design -directive AlternateReplication"
    phys_opt_design -directive AlternateReplication
  }
} elseif {$setup_slack < $phys_opt_effective_floor_ns} {
  puts [format \
    "FPT26 route post: skipping ineffective phys_opt_design at %.3f ns" \
    $setup_slack]
}

if {![info exists ::env(FPT26_VITIS_BUILD_DIR)]} {
  error "FPT26 route post requires FPT26_VITIS_BUILD_DIR"
}
set final_report [file normalize [file join \
  $::env(FPT26_VITIS_BUILD_DIR) report hw link imp \
  impl_1_hw_bb_locked_timing_summary_post_route_physopt.rpt]]
file mkdir [file dirname $final_report]
if {[file exists $final_report]} {
  file delete -force $final_report
}
report_timing_summary -delay_type min_max -max_paths 10 \
  -file $final_report
puts "FPT26 route post: final timing report $final_report"
