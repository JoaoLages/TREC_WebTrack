#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

DATA_CONFIGS=configs/data
MODEL_CONFIGS=configs/model
OUTPUT=model_outputs
DATA=customdata
MODEL=pacrr_customdata

printf "\nTesting \033[94m%s\033[0m on \033[94m%s\033[0m\n" ${DATA} ${MODEL}
python scripts/test.py \
    --data-config  ${DATA_CONFIGS}/${DATA}.yml \
    --model-folder DATA/${OUTPUT}/${DATA}/${MODEL} \
    --results-folder DATA/${OUTPUT}/${DATA}/${MODEL}

printf "Relevance scores saved under %s/test.probs" DATA/${OUTPUT}/${DATA}/${MODEL}