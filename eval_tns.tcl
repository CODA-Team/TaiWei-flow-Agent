# eval_tns.tcl - Re-evaluate TNS under a fixed reference clock period (CP_0).
#
# Required env vars:
#   LIB_FILES   - space-separated lib glob patterns
#   ODB_PATH    - path to 6_final.odb
#   SPEF_PATH   - path to 6_final.spef
#   CP0         - new clock period (replaces ODB's existing clock)
#   CLK_PORT    - clock port name (e.g. clk, clk_i)

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

# --- Critical: remove any clocks left over from the original flow run ---
set existing_clocks [all_clocks]
puts "DEBUG: Pre-existing clocks in ODB: $existing_clocks"
foreach clk $existing_clocks {
    set clk_name [get_name $clk]
    puts "DEBUG: Removing existing clock: $clk_name"
    catch {sta::remove_clock $clk_name}
}

# --- Find the clock port ---
set clk_port_name $::env(CLK_PORT)
set clk_port [get_ports -quiet $clk_port_name]
if {$clk_port == "" || [llength $clk_port] == 0} {
    puts "ERROR: clock port '$clk_port_name' not found in design"
    puts "DEBUG: Available input ports:"
    foreach p [get_ports *] {
        if {[$p getSigType] == "POWER" || [$p getSigType] == "GROUND"} continue
        puts "  - [$p getName]"
    }
    exit 1
}
puts "DEBUG: Found clock port '$clk_port_name'"

# --- Create new clock with CP_0 period ---
set cp0 $::env(CP0)
puts "DEBUG: Creating clock 'eval_clk' with period $cp0"
create_clock -name eval_clk -period $cp0 $clk_port

# --- Read parasitics ---
if {[file exists $::env(SPEF_PATH)]} {
    puts "DEBUG: Reading SPEF from $::env(SPEF_PATH)"
    read_spef $::env(SPEF_PATH)
} else {
    puts "DEBUG: SPEF not found at $::env(SPEF_PATH), running estimate_parasitics -placement"
    estimate_parasitics -placement
}

# --- Propagate and report ---
set_propagated_clock [all_clocks]
set active_clocks [all_clocks]
puts "DEBUG: Active clocks after setup: $active_clocks"

# Verify the clock period was actually set to CP_0
foreach clk $active_clocks {
    set period [sta::clock_period $clk]
    puts "DEBUG: Clock [get_name $clk] period = $period"
    set src_pins [sta::clock_source_pins $clk]
    puts "DEBUG: Clock [get_name $clk] source pins = $src_pins"
}

# Count registers / endpoints
set reg_count [llength [sta::all_registers -no_check]]
puts "DEBUG: Register count = $reg_count"

# Show the worst path explicitly (check if any path exists at all)
puts "DEBUG: report_checks -path_delay max -slack_max 1e9 -group_count 1 -endpoint_count 1:"
report_checks -path_delay max -slack_max 1e9 -group_count 1 -endpoint_count 1

puts "DEBUG: Reporting TNS/WNS under CP_0 = $cp0"
report_tns
report_wns

exit
