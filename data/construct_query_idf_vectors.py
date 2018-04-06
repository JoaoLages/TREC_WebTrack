import sys
import os
import multiprocessing as mp
import numpy as np
import argparse
import itertools
from utils.utils import read_file, read_qrels, read_query, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def process(q_recv, q_send, corpus_folder, remove_stopwords):
    articles = []
    while True:
        cwid = q_recv.get()

        # Check end condition
        if cwid is None:
            break

        # Article is already encoded in UTF-8
        article = read_file("%s/%s" % (corpus_folder, cwid))

        # PREPROCESS
        articles += [preprocess_text(article, tokenize=True, all_lower=True, stopw=remove_stopwords).split()]

    # Send info back
    q_send.put((articles))


def argument_parser(sys_argv):
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(
        prog='Construct query IDF vectors with all vocabulary',
    )
    parser.add_argument(
        '--qrel-files',
        help="qrel files paths",
        required=True,
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--topic-files',
        help="XML topic files paths",
        required=True,
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--corpus-folder',
        help="Directory of raw texts",
        required=True,
        type=str
    )
    parser.add_argument(
        '--remove-stopwords',
        help="Bool variable ro remove stopwords",
        default=True,
        type=bool
    )
    parser.add_argument(
        '--outdir', '-o',
        help="Output directory",
        required=True,
        type=str
    )

    return parser.parse_args(sys_argv)


if __name__ == '__main__':

    # Argument handling
    args = argument_parser(sys.argv[1:])

    # Read QRels files
    qrels = []
    for qrel in args.qrel_files:
        qrels += read_qrels(qrel)

    # Read topics and unique cwids
    query_id2text = read_query(args.topic_files)
    cwids = list({qrel[2] for qrel in qrels})

    # Initialize pool and queues
    q_process_recv = mp.Queue(maxsize=mp.cpu_count())
    q_process_send = mp.Queue(maxsize=mp.cpu_count())
    pool = mp.Pool(
        mp.cpu_count(),
        initializer=process,
        initargs=(q_process_recv, q_process_send, args.corpus_folder, args.remove_stopwords)
    )

    # Send qrels
    for cwid in tqdm(cwids, desc='Reading all corpora'):
        q_process_recv.put(cwid)  # blocks until q below its max size

    # Tell workers we're done
    for _ in range(mp.cpu_count()):
        q_process_recv.put(None)

    # Receive info
    article_words = []
    for _ in range(mp.cpu_count()):
        for x in zip(*q_process_send.get()):
            article_words += x

    # Close pool
    pool.close()
    pool.join()

    # Get topics and descriptions vocab
    topics = [
        preprocess_text(query_id2text[qid]['query'], tokenize=True, all_lower=True, stopw=args.remove_stopwords).split()
        for qid in sorted(query_id2text.keys())
    ]
    descriptions = [
        preprocess_text(query_id2text[qid]['description'], tokenize=True, all_lower=True, stopw=args.remove_stopwords).split()
        for qid in sorted(query_id2text.keys())
    ]

    for queries, f_name in zip([topics, descriptions], ['topic', 'description']):
        vocabulary = article_words + list(itertools.chain.from_iterable(queries))

        # Construct vectorizer
        vectorizer = TfidfVectorizer(
            use_idf=True,
            norm='l2',
            smooth_idf=False,
            sublinear_tf=False,  # tf = 1+ln(tf)
            binary=False,
            max_features=None,
            token_pattern=r"(?u)\b\w+\b"
        )

        # Get IDF dict
        vectorizer.fit_transform(vocabulary)
        idf = vectorizer.idf_
        word2idf = dict(zip(vectorizer.get_feature_names(), idf))

        if not os.path.isdir("%s/%s" % (args.outdir, f_name)):
            os.makedirs("%s/%s" % (args.outdir, f_name))

        # Get query IDF vectors
        for query, qid in tqdm(zip(queries, sorted(query_id2text.keys())), total=len(queries), desc='Saving IDF vectors'):
            query_idf = np.array(list(map(lambda x: word2idf[x], query)))
            # Save IDF array
            np.save("%s/%s/%s.npy" % (args.outdir, f_name, qid), query_idf)
