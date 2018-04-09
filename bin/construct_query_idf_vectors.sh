#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

QREL_FILES='qrels/2009Bqrels.adhoc.txt qrels/2010Bqrels.adhoc.txt qrels/2011Bqrels.adhoc.txt qrels/2012Bqrels.adhoc.txt
qrels/2013Bqrels.adhoc.txt qrels/2014Bqrels.adhoc.txt'
TREC_QUERY_FILES='topic_files/trec2009-topics.xml topic_files/trec2010-topics.xml topic_files/trec2011-topics.xml
topic_files/trec2012-topics.xml topic_files/trec2013-topics.xml topic_files/trec2014-topics.xml'

printf "Constructing query_idf vectors\n"
python data/construct_query_idf_vectors.py \
    --qrel-files ${QREL_FILES}\
    --topic-files ${TREC_QUERY_FILES} \
    --corpus-folder DATA/corpora/texts \
    --outdir DATA/corpora/query_idf/
