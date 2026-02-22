#!/bin/bash
MODEL_TYPE="LiteTracker"
SHOW_VIS=0
DATA_DIR="$1"
WEIGHTS_PATH="$2"

uv run src/eval/stirc_2024/flow2d.py \
    --num_data -1 \
    --modeltype $MODEL_TYPE \
    --showvis $SHOW_VIS \
    --datadir $DATA_DIR \
    --weights_path $WEIGHTS_PATH

uv run src/eval/stirc_2024/write2dgtjson.py \
    --num_data -1 \
    --jsonsuffix test \
    --datadir $DATA_DIR

uv run src/eval/stirc_2024/calculate_error_from_json2d.py \
    --startgt results/gt_positions_end_all_test.json \
    --endgt results/gt_positions_end_all_test.json \
    --datadir $DATA_DIR \
    --model_predictions results/positions_allLiteTracker.json