#!/bin/bash
# run_make.sh


CURRENT_CONFIG=$1
TASK_ID=$2
PLATFORM=$3
DESIGN=$4

export NUM_CORES="${NUM_CORES:-16}"


BASE_PATH="${PLATFORM}/${DESIGN}/base_${TASK_ID}"
RES_DIR="./results/${BASE_PATH}"
LOG_DIR="./logs/${BASE_PATH}"
OBJ_DIR="./objects/${BASE_PATH}"
REP_DIR="./reports/${BASE_PATH}"

mkdir -p "$RES_DIR" "$LOG_DIR" "$OBJ_DIR" "$REP_DIR"
export USE_FLOW=openroad
export FLOW_VARIANT=base_${TASK_ID}

make DESIGN_CONFIG=designs/${PLATFORM}/${DESIGN}/config2d.mk clean_all
make DESIGN_CONFIG=${CURRENT_CONFIG} clean_all

make \
    DESIGN_CONFIG="designs/${PLATFORM}/${DESIGN}/config2d.mk" \
    INT_PARAM="${TASK_ID}" \
    RESULTS_DIR="${RES_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    OBJECTS_DIR="${OBJ_DIR}" \
    REPORTS_DIR="${REP_DIR}" \
    ord-3d-flow-2dpart

BASE_PATH="${PLATFORM}/${DESIGN}/base_${TASK_ID}"
RES_DIR="./results/${BASE_PATH}"
LOG_DIR="./logs/${BASE_PATH}"
OBJ_DIR="./objects/${BASE_PATH}"
REP_DIR="./reports/${BASE_PATH}"

make \
    DESIGN_CONFIG="${CURRENT_CONFIG}" \
    INT_PARAM="${TASK_ID}" \
    RESULTS_DIR="${RES_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    OBJECTS_DIR="${OBJ_DIR}" \
    REPORTS_DIR="${REP_DIR}" \
    ord-3d-flow
