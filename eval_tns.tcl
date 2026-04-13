# eval_tns.tcl - Re-evaluate TNS under a fixed reference clock period (CP_0).
#
# Usage:  openroad -exit eval_tns.tcl
#
# Required env vars:
#   LIB_FILES  - space-separated lib glob patterns (e.g. "path/*.lib.gz")
#   ODB_PATH   - path to 6_final.odb
#   SPEF_PATH  - path to 6_final.spef
#   SDC_PATH   - path to SDC file with CP_0 as clk_period

# Load design database first (tech + physical design)
read_db $::env(ODB_PATH)

# Load liberty libraries (timing models) AFTER read_db so they are not cleared
foreach lib_pattern $::env(LIB_FILES) {
    foreach lib [glob -nocomplain $lib_pattern] {
        read_liberty $lib
    }
}

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

# Report TNS (setup) — output goes to stdout, captured by eval_tns.sh
report_tns
report_wns

exit
