#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

TREC_QUERY_FILES='topic_files/trec2009-topics.xml topic_files/trec2010-topics.xml topic_files/trec2011-topics.xml
topic_files/trec2012-topics.xml topic_files/trec2013-topics.xml topic_files/trec2014-topics.xml'

printf "Constructing embedding matrix\n"
python utils/construct_embedding_matrix.py \
    -d DATA/corpora/texts/ \
    -q ${TREC_QUERY_FILES} \
    -o DATA/embeddings/retrained-w2v-clueweb12
