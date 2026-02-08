#!/bin/bash
# run_make.sh


CURRENT_CONFIG=$1
TASK_ID=$2
PLATFORM=$3
DESIGN=$4


BASE_PATH="${PLATFORM}/${DESIGN}/base_${TASK_ID}"
RES_DIR="./results/${BASE_PATH}"
LOG_DIR="./logs/${BASE_PATH}"
OBJ_DIR="./objects/${BASE_PATH}"
REP_DIR="./reports/${BASE_PATH}"


mkdir -p "$RES_DIR" "$LOG_DIR" "$OBJ_DIR" "$REP_DIR"

# use taskset to run make
make \
    DESIGN_CONFIG="${CURRENT_CONFIG}" \
    INT_PARAM="${TASK_ID}" \
    RESULTS_DIR="${RES_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    OBJECTS_DIR="${OBJ_DIR}" \
    REPORTS_DIR="${REP_DIR}" clean_all
make \
    DESIGN_CONFIG="${CURRENT_CONFIG}" \
    INT_PARAM="${TASK_ID}" \
    RESULTS_DIR="${RES_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    OBJECTS_DIR="${OBJ_DIR}" \
    REPORTS_DIR="${REP_DIR}"