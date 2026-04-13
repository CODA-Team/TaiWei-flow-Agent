#!/bin/bash

# Ensure correct number of arguments
if [ $# -ne 4 ]; then
    echo "Usage: $0 <platform> <design> <parallel_runs> <iteration>"
    echo "platform: asap7 or sky130hd or nangate45"
    echo "design: aes, ibex, or jpeg"
    echo "parallel_runs: number of parallel runs"
    echo "iteration: current iteration index (1-based)"
    exit 1
fi

platform=$1
design=$2
parallel_runs=$3
iteration=$4

# Iteration 1 runs one extra "default" baseline task (config_(N+1).mk generated
# by run_sequential.sh from the unperturbed config.mk). Resource split is done
# against this effective count so all 26 tasks share CPUs/RAM evenly.
if [ "$iteration" -eq 1 ]; then
    parallel_runs=$((parallel_runs + 1))
    echo "[run_parallel.sh] Iteration 1: running $parallel_runs tasks (25 perturbed + 1 default baseline)"
fi

export PLATFORM=$platform
export DESIGN=$design

# Validate platform and design
# if [[ ! "$platform" =~ ^(asap7|sky130hd)$ ]]; then
#     echo "Error: platform must be asap7 or sky130hd"
#     exit 1
# fi

if [[ ! "$design" =~ ^(aes|ibex|jpeg)$ ]]; then
    echo "Error: design must be aes, ibex, or jpeg"
    # exit 1
fi

# Get resource limits from environment or use defaults
TIMEOUT=${TIMEOUT:-"120m"}
TOTAL_CPUS=${TOTAL_CPUS:-100}
TOTAL_RAM=${TOTAL_RAM:-200}

# Calculate resources per run
cpus_per_run=$((TOTAL_CPUS / parallel_runs))
ram_per_run=$((TOTAL_RAM / parallel_runs))

# Ensure minimum resources
if [ $cpus_per_run -lt 2 ]; then
    echo "Warning: Not enough CPUs. Reducing parallel runs to $((TOTAL_CPUS / 2))"
    parallel_runs=$((TOTAL_CPUS / 2))
    cpus_per_run=2
fi

if [ $ram_per_run -lt 4 ]; then
    echo "Warning: Not enough RAM. Reducing parallel runs to $((TOTAL_RAM / 4))"
    parallel_runs=$((TOTAL_RAM / 4))
    ram_per_run=4
fi

# Function to run a single task
run_task() {
    local task_id=$1
    local start_cpu=$2
    local end_cpu=$3

    CURRENT_CONFIG="./designs/${platform}/${design}/config_${task_id}.mk"
    # $(info Using config.mk from $(DESIGN_CONFIG))
    echo "[run_parallel.sh ]Using config.mk from ${CURRENT_CONFIG}"

    chmod +x run_make_2d.sh

    taskset -c "${start_cpu}-${end_cpu}" ./run_make_2d.sh \
        "$CURRENT_CONFIG" \
        "$task_id" \
        "$platform" \
        "$design" \
        > "logs/${platform}_${design}_run${task_id}.log" 2>&1 &
    
    echo "Started task $task_id on CPUs $start_cpu-$end_cpu"
}

# Create logs directory
mkdir -p logs

# Start parallel tasks
for ((i=1; i<=$parallel_runs; i++)); do
    start_cpu=$(( (i-1) * cpus_per_run ))
    end_cpu=$(( start_cpu + cpus_per_run - 1 ))
    run_task $i $start_cpu $end_cpu
done

# Wait for all background tasks to complete
wait


echo "All tasks completed" 
