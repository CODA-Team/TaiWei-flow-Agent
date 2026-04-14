# eval_tns.tcl - Re-evaluate TNS under a fixed reference clock period (CP_0).
#
# Required env vars:
#   LIB_FILES   - space-separated lib glob patterns
#   ODB_PATH    - path to 6_final.odb
#   SPEF_PATH   - path to 6_final.spef
#   CP0         - new clock period (replaces ODB's existing clock)
#   CLK_PORT    - clock port name (e.g. clk, clk_i)

# Load liberty libraries FIRST so that read_db can link the design's
# instances to timing models. If read_db runs before read_liberty,
# the netlist cells stay un-linked and STA reports 0 registers / 0 paths.
puts "DEBUG: Loading liberty libraries..."
set lib_count 0
foreach lib_pattern $::env(LIB_FILES) {
    foreach lib [glob -nocomplain $lib_pattern] {
        read_liberty $lib
        incr lib_count
    }
}
puts "DEBUG: Loaded $lib_count liberty files"

puts "DEBUG: Loading ODB from $::env(ODB_PATH)"
read_db $::env(ODB_PATH)

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
puts "DEBUG: Active clocks after setup: [all_clocks]"

# Inspect each clock (use get_property which is standard Tcl OO)
foreach clk [all_clocks] {
    puts "DEBUG: clk object = $clk"
    if {[catch {get_property $clk name} n]} { set n "(unknown)" }
    if {[catch {get_property $clk period} p]} { set p "(unknown)" }
    if {[catch {get_property $clk sources} s]} { set s "(unknown)" }
    puts "DEBUG:   name=$n  period=$p  sources=$s"
}

# Count registers
if {[catch {llength [all_registers]} reg_count]} { set reg_count "(n/a: $reg_count)" }
puts "DEBUG: Register count = $reg_count"

# Show worst setup path — no fancy options, just defaults
puts "DEBUG: --- report_checks (default = worst setup path) ---"
if {[catch {report_checks} rc_err]} {
    puts "DEBUG: report_checks error: $rc_err"
}

puts "DEBUG: --- report_tns / report_wns under CP_0 = $cp0 ---"
report_tns
report_wns

exit
