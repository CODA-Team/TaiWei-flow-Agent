#!/bin/bash
# run_all.sh

LOG_FILE="run_summary.csv"

echo "Design,Platform,Duration(seconds),Formatted_Time" > "$LOG_FILE"

pairs=(
  "aes asap7_3D"
  "ibex asap7_3D"
  "jpeg asap7_3D"
  "aes asap7_nangate45_3D"
  "ibex asap7_nangate45_3D"
  "jpeg asap7_nangate45_3D"
  # "aes asap7"
  # "ibex sky130hd"
  # "ibex asap7"
  # "jpeg sky130hd"
  # "jpeg asap7"
  # "aes sky130hd"
)

format_time() {
    local T=$1
    local H=$((T/3600))
    local M=$(( (T%3600)/60 ))
    local S=$((T%60))
    printf "%02d:%02d:%02d" $H $M $S
}

echo "Starting batch run... Logging to $LOG_FILE"

for pair in "${pairs[@]}"; do
  read d p <<< "$pair"
  echo "==========================================="
  echo "Running design: $d, platform: $p"
  echo "==========================================="
  
  start_ts=$(date +%s)
  
  ./maindriver.sh -p "$p" -d "$d" -o DWL
  
  end_ts=$(date +%s)
  duration=$((end_ts - start_ts))
  
  formatted=$(format_time $duration)
  
  echo "$d,$p,$duration,$formatted" >> "$LOG_FILE"
  
  echo "--> Completed in $formatted ($duration seconds)"
  echo
done


echo ""
echo "==========================================="
echo "           EXECUTION SUMMARY               "
echo "==========================================="

if command -v column &> /dev/null; then
    column -t -s, "$LOG_FILE"
else
    cat "$LOG_FILE"
fi
echo "==========================================="
