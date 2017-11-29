#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

printf "Extracting text from HTML files\n"
python corpus_construction/article_extractor.py \
    --qrels-files qrels/2013qrels.all.txt qrels/2014qrels.all.txt \
    --use-diffbot True \
    --re-extract True
