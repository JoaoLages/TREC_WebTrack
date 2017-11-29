#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

DATA_CONFIGS=configs/data
MODEL_CONFIGS=configs/model
OUTPUT=model_outputs
DATA=train09_10_12_14_val11_test13
MODEL=repacrr

printf "Testing \033[94m%s\033[0m on \033[94m%s\033[0m\n" ${DATA} ${MODEL}
python scripts/test.py \
    --data-config  ${DATA_CONFIGS}/${DATA}.yml \
    --model-folder DATA/${OUTPUT}/${DATA}/${MODEL} \
    --metrics NDCG20 ERR20 \
    --results-folder DATA/${OUTPUT}/${DATA}/${MODEL}
