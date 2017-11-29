#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

DATA_CONFIGS=configs/data
MODEL_CONFIGS=configs/model
OUTPUT=model_outputs
DATA=train09_11_12_14_val10_test13
MODEL=repacrr

printf "Training \033[94m%s\033[0m on \033[94m%s\033[0m\n" ${DATA} ${MODEL}
python scripts/train.py \
    --data-config ${DATA_CONFIGS}/${DATA}.yml \
    --model-config ${MODEL_CONFIGS}/${MODEL}.yml \
    --model-folder DATA/${OUTPUT}/${DATA}/${MODEL} \
    --metrics NDCG20 ERR20\
    --overload model.gpu_device=${1}

printf "\nTesting \033[94m%s\033[0m on \033[94m%s\033[0m\n" ${DATA} ${MODEL}
python scripts/test.py \
    --data-config  ${DATA_CONFIGS}/${DATA}.yml \
    --model-folder DATA/${OUTPUT}/${DATA}/${MODEL} \
    --metrics NDCG20 ERR20 \
    --results-folder DATA/${OUTPUT}/${DATA}/${MODEL}
