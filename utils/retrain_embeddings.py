import os
import sys
from random import shuffle
import gensim
from gensim.models import Word2Vec
import argparse
from utils import read_file, read_query, preprocess_text
from tqdm import tqdm

'''
based on the examples from https://radimrehurek.com/gensim/models/word2vec.html
'''


def argument_parser(sys_argv):
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(
        prog='Retrain embeddings',
    )
    parser.add_argument(
        '--datadir', '-d',
        help="Data directory path",
        required=True,
        type=str
    )
    parser.add_argument(
        '--trec-query-file', '-q',
        help="Query XML file path",
        required=True,
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--outdir', '-o',
        help="Output directory",
        required=True,
        type=str
    )
    parser.add_argument(
        '--googlepretrain', '-g',
        help="Word2vec binary file",
        required=True,
        type=str
    )

    args = parser.parse_args(sys_argv)

    return args.datadir, args.trec_query_file, args.outdir, args.googlepretrain


if __name__ == '__main__':

    # Argument handling
    cwid_txt_dir, query_xml_file, outdir, g_pretrain_bin_file = \
        argument_parser(sys.argv[1:])

    query_id2text = read_query(query_xml_file)

    # each line corresponds to one document
    alldocs = []  # will hold all docs in original order
    alltags = []
    for _, _, files in os.walk(cwid_txt_dir):
        for file in tqdm(files, desc="Reading all texts"):
            alldocs.append(
                preprocess_text(
                    read_file("%s/%s" % (cwid_txt_dir, file)), tokenize=True, all_lower=True
                ).split()
            )
    for qid in query_id2text:
        # Add topic
        alldocs.append(
            preprocess_text(
                query_id2text[qid]['query'], tokenize=True, all_lower=True
            ).split()
        )
        # Add description
        alldocs.append(
            preprocess_text(
                query_id2text[qid]['description'], tokenize=True, all_lower=True
            ).split()
        )

    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    print("Building vocabulary ...")
    model = Word2Vec(size=300, window=10, min_count=1, workers=32)
    model.build_vocab(alldocs)
    # only train for unseen words, retain the existing term vector in google pre-trained
    model.intersect_word2vec_format(g_pretrain_bin_file, binary=True, lockf=0.0)

    alpha, min_alpha, passes = 0.025, 0.001, 20
    alpha_delta = (alpha - min_alpha) / passes

    for epoch in tqdm(range(passes), desc="Training w2v"):
        shuffle(alldocs)  # shuffling gets best results
        model.alpha, model.min_alpha = alpha, min_alpha
        model.train(alldocs, total_examples=len(alldocs), epochs=model.iter)
        alpha -= alpha_delta

    print("INFO: all passes completed with %d terms " % len(model.wv.vocab))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    model.save(outdir + "/retrained-w2v-clueweb12")
