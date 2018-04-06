#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

printf "Extracting text from HTML files\n"
python corpus_construction/article_extractor.py \
    --qrels-files qrels/2009qrels.adhoc.txt qrels/2010qrels.adhoc.txt \
    qrels/2011qrels.adhoc.txt qrels/2012qrels.adhoc.txt \
    qrels/2013qrels.adhoc.txt qrels/2014qrels.adhoc.txt \
    --use-diffbot True \
    --re-extract True
