import pickle
import os
import sys
# Run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from keras.preprocessing.text import Tokenizer
import argparse
from utils import read_file, read_query, preprocess_text
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec


def argument_parser(sys_argv):
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(
        prog='Constructing embedding matrix',
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
        '--embeddings-path', '-o',
        required=True,
        type=str
    )
    args = parser.parse_args(sys_argv)

    return args.datadir, args.trec_query_file, args.embeddings_path


if __name__ == '__main__':

    # Argument handling
    cwid_txt_dir, query_xml_file, embeddings_path = \
        argument_parser(sys.argv[1:])

    query_id2text = read_query(query_xml_file)

    # each line corresponds to one document
    alltexts = []  # will hold all docs in original order
    for _, _, files in os.walk(cwid_txt_dir):
        for file in tqdm(files, desc="Reading all texts"):
            alltexts.append(
                preprocess_text(
                    read_file("%s/%s" % (cwid_txt_dir, file)), tokenize=True, all_lower=True
                )
            )
    for qid in query_id2text:
        # Add topic
        alltexts.append(
            preprocess_text(
                query_id2text[qid]['query'], tokenize=True, all_lower=True
            )
        )
        # Add description
        alltexts.append(
            preprocess_text(
                query_id2text[qid]['description'], tokenize=True, all_lower=True
            )
        )

    text_lens = [len(x.split()) for x in alltexts]

    # Encode & pad texts
    t = Tokenizer()
    t.fit_on_texts(alltexts)
    vocab_size = len(t.word_index) + 1

    # Construct embedding matrix
    embeddings = Word2Vec.load(embeddings_path)
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        if word in embeddings:
            embedding_matrix[i] = embeddings[word]

    # Save text_lengths
    with open('%s/text_lengths.pickle' % os.path.dirname(embeddings_path), 'wb') as handle:
        pickle.dump(text_lens, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save embeddings matrix
    np.save('%s/embeddings_matrix.npy' % os.path.dirname(embeddings_path), embedding_matrix)

    # Save tokenizer
    with open('%s/tokenizer_vocabsize.pickle' % os.path.dirname(embeddings_path), 'wb') as handle:
        pickle.dump([t, vocab_size], handle, protocol=pickle.HIGHEST_PROTOCOL)
