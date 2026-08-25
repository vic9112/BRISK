set ucs [get_bd_cells -quiet /ulp_ucs]
if {[llength $ucs] != 1} {
  error "Expected exactly one /ulp_ucs block-design cell"
}

set_property CONFIG.HAS_HBM_CLK 2 $ucs

set actual_mode [get_property CONFIG.HAS_HBM_CLK $ucs]
if {$actual_mode ne "2"} {
  error "Failed to select U50 400 MHz HBM AXI clock mode: ${actual_mode}"
}

validate_bd_design
save_bd_design
puts "FPT26 U50 HBM AXI clock mode: 400 MHz"
