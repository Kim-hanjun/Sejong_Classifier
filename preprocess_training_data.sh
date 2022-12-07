#!/bin/bash

LABEL="소장"
#LABEL="제안서"
#LABEL="의견서"
#LABEL="실사보고서"
#LABEL="변론요지서"
BASE_DATA_SOURCE_DIR="/data/sejong_data/input/$LABEL"
BASE_TARGET_DIR="/data/sejong_data/output/$LABEL"
mkdir -p "${BASE_TARGET_DIR}"

# 세종 개발기
#DOC_EXTRACTOR_CMD="/home/workspace/venv/venv/bin/synap/v4/snf_exe"
# 42MARU 개발기
DOC_EXTRACTOR_CMD="/data/sejong_data/cynap_cmd/snf_exe"

LOG_FILE="$BASE_TARGET_DIR/preprocess_training_data_4_${LABEL}.log"

python preprocess_training_data.py --base_data_source_dir "$BASE_DATA_SOURCE_DIR" \
    --label "$LABEL" --base_target_dir "$BASE_TARGET_DIR" \
    --doc_extractor_cmd "$DOC_EXTRACTOR_CMD" \
   > "$LOG_FILE" 2>&1