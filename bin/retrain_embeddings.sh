#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

TREC_QUERY_FILES='topic_files/trec2009-topics.xml topic_files/trec2010-topics.xml topic_files/trec2011-topics.xml
topic_files/trec2012-topics.xml topic_files/trec2013-topics.xml topic_files/trec2014-topics.xml'

printf "Retraining word2vec embeddings\n"
python utils/retrain_embeddings.py \
    -d DATA/corpora/texts/ \
    -q ${TREC_QUERY_FILES} \
    -o DATA/embeddings/ \
    -g DATA/embeddings/GoogleNews-vectors-negative300.bin
