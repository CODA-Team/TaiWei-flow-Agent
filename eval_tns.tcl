# eval_tns.tcl - Re-evaluate TNS under a fixed reference clock period (CP_0).
#
# Usage:  openroad -exit scripts/eval_tns.tcl
#
# Required env vars:
#   LIB_FILES  - space-separated lib glob patterns (e.g. "path/*.lib.gz")
#   ODB_PATH   - path to 6_final.odb
#   SPEF_PATH  - path to 6_final.spef
#   SDC_PATH   - path to SDC file with CP_0 as clk_period

# Load liberty libraries
foreach lib_pattern $::env(LIB_FILES) {
    foreach lib [glob -nocomplain $lib_pattern] {
        read_liberty $lib
    }
}

# Load design
read_db $::env(ODB_PATH)

# Read SDC (contains CP_0 clock period)
read_sdc $::env(SDC_PATH)

# Read parasitics
if {[file exists $::env(SPEF_PATH)]} {
    read_spef $::env(SPEF_PATH)
} else {
    puts "WARNING: SPEF not found: $::env(SPEF_PATH)"
}

# Propagate clocks for accurate STA
set_propagated_clock [all_clocks]

# Report TNS (setup)
report_tns
set tns_result [sta::total_negative_slack_cmd "max"]
puts "tns_eval = $tns_result"

exit
