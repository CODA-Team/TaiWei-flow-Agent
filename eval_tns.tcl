# eval_tns.tcl - Re-evaluate TNS under a fixed reference clock period (CP_0).
#
# Required env vars:
#   LIB_FILES  - space-separated lib glob patterns
#   ODB_PATH   - path to 6_final.odb
#   SPEF_PATH  - path to 6_final.spef
#   SDC_PATH   - path to SDC file with CP_0 as clk_period

puts "DEBUG: Loading ODB from $::env(ODB_PATH)"
read_db $::env(ODB_PATH)

puts "DEBUG: Loading liberty libraries..."
set lib_count 0
foreach lib_pattern $::env(LIB_FILES) {
    foreach lib [glob -nocomplain $lib_pattern] {
        read_liberty $lib
        incr lib_count
    }
}
puts "DEBUG: Loaded $lib_count liberty files"

puts "DEBUG: Reading SDC from $::env(SDC_PATH)"
read_sdc $::env(SDC_PATH)

# Show clocks created by SDC
set clocks [all_clocks]
puts "DEBUG: Clocks found: $clocks"
puts "DEBUG: Number of clocks: [llength $clocks]"

# Read parasitics
if {[file exists $::env(SPEF_PATH)]} {
    puts "DEBUG: Reading SPEF from $::env(SPEF_PATH)"
    read_spef $::env(SPEF_PATH)
} else {
    puts "DEBUG: SPEF not found at $::env(SPEF_PATH), running estimate_parasitics"
    estimate_parasitics -placement
}

# Propagate clocks
set_propagated_clock [all_clocks]

# Show timing path count for debugging
puts "DEBUG: Reporting timing..."
report_tns
report_wns

exit
