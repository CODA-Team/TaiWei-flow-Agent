#!/bin/bash
# run_make.sh

# 接收传入的变量
CURRENT_CONFIG=$1
TASK_ID=$2
PLATFORM=$3
DESIGN=$4

# 定义目录路径（方便复用）
BASE_PATH="${PLATFORM}/${DESIGN}/base_${TASK_ID}"
RES_DIR="./results/${BASE_PATH}"
LOG_DIR="./logs/${BASE_PATH}"
OBJ_DIR="./objects/${BASE_PATH}"
REP_DIR="./reports/${BASE_PATH}"

# 确保必要的目录存在
mkdir -p "$RES_DIR" "$LOG_DIR" "$OBJ_DIR" "$REP_DIR"

# 使用 taskset 执行 make
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