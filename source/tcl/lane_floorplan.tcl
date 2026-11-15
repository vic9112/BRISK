# Keep each lane's ownership, arithmetic and AXI adapters together. The two
# platform SLR regions split the independent lanes evenly; shared actors and
# FIFOs remain available to the placer. No arithmetic or memory layout changes.

proc require_one {objects description} {
    if {[llength $objects] != 1} {
        error "BRISK floorplan expected one $description, found [llength $objects]"
    }
    return [lindex $objects 0]
}

set dataflow [require_one \
    [get_cells -quiet -hierarchical -regexp \
        {.*/grp_p_anonymous_namespace_run_tile_fu_[0-9]+$}] \
    "attention DATAFLOW region"]
set dataflow_name [get_property NAME $dataflow]
set kernel_name [file dirname $dataflow_name]
set regions [list \
    [require_one [get_pblocks -quiet pblock_dynamic_SLR0] "SLR0 region"] \
    [require_one [get_pblocks -quiet pblock_dynamic_SLR1] "SLR1 region"]]

# These are the actual generated actor identities, in source call/lane order.
# Suffixes are compiler uniquification, not algorithm parameters. Fail loudly
# if a later source/tool change alters this mapping instead of misplacing it.
set score_actors [list score_queries_1_U0 score_queries_2_U0 \
    score_queries_3_U0 score_queries_U0]
set value_actors [list accumulate_values_4_U0 accumulate_values_5_U0 \
    accumulate_values_6_U0 accumulate_values_U0]
set store_actors [list p_anonymous_namespace_store_output_U0 \
    p_anonymous_namespace_store_output_7_U0 \
    p_anonymous_namespace_store_output_8_U0 \
    p_anonymous_namespace_store_output_9_U0]

for {set lane 0} {$lane < 4} {incr lane} {
    set actors [list]
    foreach actor [list "serve_kv_lane_${lane}_U0" \
        [lindex $score_actors $lane] [lindex $value_actors $lane] \
        [lindex $store_actors $lane]] {
        lappend actors [require_one \
            [get_cells -quiet "${dataflow_name}/${actor}"] "lane ${lane} ${actor}"]
    }
    foreach bus [list "gmem_qo${lane}" "gmem_kv${lane}"] {
        lappend actors [require_one \
            [get_cells -quiet "${kernel_name}/${bus}_m_axi_U"] \
            "lane ${lane} ${bus} AXI adapter"]
    }
    set region [lindex $regions [expr {$lane / 2}]]
    add_cells_to_pblock $region $actors
    puts "BRISK lane ${lane}: region=${region}, actors=${actors}"
}
