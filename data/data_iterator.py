import os
from data.template import Data as DataTemplate, DataIterator
from data.pos_methods import *
from utils.utils import read_file, read_query, read_qrels, split_in_sentences, preprocess_text, EMPTY_TOKEN
from gensim.models import Word2Vec
from gensim.matutils import unitvec
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import warnings
import pickle
import math
import time
import gc


def process(q_recv, q_send, query_id2text, label2tlabel, corpus_folder,
            remove_stopwords, use_topic, use_description, use_static_matrices, select_pos_func,
            sim_matrix_config, query_idf_config, embeddings, n_grams, include_spam):
    qids, cwids, labels, ngram_mats, query_idfs, context_vecs, docs, queries, ngram_masks = [], [], [], [], [], [], [], [], []

    while True:
        qrel = q_recv.get()

        # Check end condition
        if qrel is None:
            break

        # Filter spam
        if not include_spam:
            if int(qrel[3]) == -2:
                continue

        # Initialize variables
        exception = None
        query_idf, ngram_mat, context_vec, q_terms, document, ngram_mask = None, dict(), None, None, None, dict()
        qid, cwid, label = int(qrel[0]), qrel[2], label2tlabel[int(qrel[3])]

        # Get preprocessed query and/or its description
        query = {}
        if use_topic:
            query['topic'] = preprocess_text(query_id2text[qid]['query'],
                                             tokenize=True, all_lower=True, stopw=remove_stopwords).split()
        if use_description:
            # Check if description not empty, replace with query if so (and if using description only)
            description = preprocess_text(query_id2text[qid]['description'],
                                          tokenize=True, all_lower=True, stopw=remove_stopwords).split()
            if description == '' and not use_topic:
                description = query_id2text[qid]['query']
            query['description'] = description

        # Initialize article variable but don't read it yet (might not be needed)
        article = None

        desc_idx = None
        aux_s = []
        if use_topic:
            aux_s.append('topic')
        if use_description:
            aux_s.append('description')

        if query_idf_config['use_query_idf'] or (use_topic and use_description):
            # Load Query IDF
            q_idfs = []
            for x in aux_s:
                if os.path.isfile('%s/%s.npy' % (query_idf_config['idf_vectors_path'][x], qrel[0])):
                    # File exists, load it
                    q_idf = np.load('%s/%s.npy' % (query_idf_config['idf_vectors_path'][x], qrel[0]))

                    if len(q_idf) > query_idf_config['max_query_len']:
                        # Send exception to main process
                        exception = "Increase max_query_len. %s/%s.npy has %s length" % \
                                    (query_idf_config['idf_vectors_path'][x], qrel[0], len(q_idf))
                        qids.append(exception)
                        cwids.append(None)
                        labels.append(None)
                        ngram_mats.append(None)
                        query_idfs.append(None)
                        context_vecs.append(None)
                        docs.append(None)
                        queries.append(None)
                        ngram_masks.append(None)
                        break

                    if q_idfs:
                        # when using topic+description
                        max_len = query_idf_config['max_query_len'] - len(q_idfs[0])
                        desc_idx = np.sort(np.argsort(q_idf)[::-1][:max_len])
                        q_idfs.append(q_idf[desc_idx])
                    else:
                        q_idfs.append(q_idf)
                else:
                    # Send exception to main process
                    exception = "Could not find file for IDF vector under %s/%s.npy. " \
                                "Please run bin/construct_query_idf_vectors.sh accordingly." \
                                % (query_idf_config['idf_vectors_path'][x], qrel[0])
                    qids.append(exception)
                    cwids.append(None)
                    labels.append(None)
                    ngram_mats.append(None)
                    query_idfs.append(None)
                    context_vecs.append(None)
                    docs.append(None)
                    queries.append(None)
                    ngram_masks.append(None)
                    break

            # Leave main loop if Exception occurred
            if exception:
                continue

            if query_idf_config['use_query_idf']:
                # Join topic+description vectors
                query_idf = np.concatenate(q_idfs, axis=0).astype(np.float32)
                # Pad to max length with zeros
                query_idf = np.pad(
                    query_idf,
                    pad_width=((0, sim_matrix_config['max_query_len'] - len(query_idf))),
                    mode='constant',
                    constant_values=-np.inf
                )
        if not use_static_matrices:
            # Don't build similarity matrix, it will be be with retrained embeddings
            document = read_file("%s/%s" % (corpus_folder, qrel[2]))
            document = preprocess_text(document, tokenize=True, all_lower=True,
                                       stopw=remove_stopwords)

            # Check document length
            if sim_matrix_config['min_doc_len'] > len(document.split()):
                continue  # skip to next iteration

            # Build context vector
            if len(aux_s) > 1:
                # Join topic + description terms
                q_terms = []
                for k in aux_s:
                    q_terms += query[k]
            else:
                q_terms = query[aux_s[0]]
            q_terms = ' '.join(q_terms)

        if sim_matrix_config['use_static_matrices']:
            sim_matrices = []
            contexts = []
            for x in aux_s:
                # Load sim_matrix
                if os.path.isfile('%s/%s/%s.npy' % (sim_matrix_config['matrices_path'][x], qrel[0], qrel[2])):
                    # File exists, load it
                    sim_matrix = np.load('%s/%s/%s.npy' % (sim_matrix_config['matrices_path'][x], qrel[0], qrel[2]))
                else:
                    # File doesn't exist, create matrix
                    if embeddings is None:
                        # Send exception to main process
                        exception = 'Matrix file does not exist under %s/%s/%s.npy ' \
                                    'and could not load embeddings to construct it. ' \
                                    'Please provide embeddings_path in data config' \
                                    % (sim_matrix_config['matrices_path'][x], qrel[0], qrel[2])
                        qids.append(exception)
                        cwids.append(None)
                        labels.append(None)
                        ngram_mats.append(None)
                        query_idfs.append(None)
                        context_vecs.append(None)
                        docs.append(None)
                        queries.append(None)
                        ngram_masks.append(None)
                        break

                    if article is None:
                        if corpus_folder is None:
                            # Send exception to main process
                            exception = 'Matrix file does not exist under %s/%s/%s.npy ' \
                                        'and could not load raw text files to construct it. ' \
                                        'Please provide corpus_folder in data config' \
                                        % (sim_matrix_config['matrices_path'][x], qrel[0], qrel[2])
                            qids.append(exception)
                            cwids.append(None)
                            labels.append(None)
                            ngram_mats.append(None)
                            query_idfs.append(None)
                            context_vecs.append(None)
                            docs.append(None)
                            queries.append(None)
                            ngram_masks.append(None)
                            break

                        article = read_file("%s/%s" % (corpus_folder, qrel[2]))
                        article = preprocess_text(article, tokenize=True, all_lower=True,
                                                  stopw=remove_stopwords).split()

                    sim_matrix = build_sim_matrix(query[x], article, embeddings, sim_matrix_config['matrix_type'])
                    # Save matrix
                    if not os.path.exists('%s/%s' % (sim_matrix_config['matrices_path'][x], qrel[0])):
                        try:
                            os.makedirs('%s/%s' % (sim_matrix_config['matrices_path'][x], qrel[0]))
                        except FileExistsError:
                            # Another worker created at the same time
                            pass
                    np.save(
                        '%s/%s/%s.npy' % (sim_matrix_config['matrices_path'][x], qrel[0], qrel[2]),
                        sim_matrix
                    )
                if sim_matrices:
                    # when using topic+description
                    assert desc_idx is not None, "When using topic AND description, query_idf_config is mandatory"

                    # Append previously selected ids to the matrix
                    sim_matrices.append(sim_matrix[desc_idx])
                else:
                    sim_matrices.append(sim_matrix)

            # Leave main loop if Exception occurred
            if exception:
                continue

            # Join topic+description matrices/vectors
            sim_matrix = np.concatenate(sim_matrices, axis=0).astype(np.float32)

            # Add context
            if sim_matrix_config['use_context']:
                x = '_'.join(aux_s)
                # Load context vector
                if os.path.isfile('%s/%s/%s.npy' % (sim_matrix_config['context_path'][x], qrel[0], qrel[2])):
                    # File exists, load it
                    context_vec = np.load('%s/%s/%s.npy' % (sim_matrix_config['context_path'][x], qrel[0], qrel[2]))
                else:
                    # File doesn't exist, create matrix
                    if embeddings is None:
                        # Send exception to main process
                        exception = 'Vector file does not exist under %s/%s/%s.npy ' \
                                    'and could not load embeddings to construct it. ' \
                                    'Please provide embeddings_path in data config' \
                                    % (sim_matrix_config['context_path'][x], qrel[0], qrel[2])
                        qids.append(exception)
                        cwids.append(None)
                        labels.append(None)
                        ngram_mats.append(None)
                        query_idfs.append(None)
                        context_vecs.append(None)
                        docs.append(None)
                        queries.append(None)
                        ngram_masks.append(None)
                        continue

                    if article is None:
                        if corpus_folder is None:
                            # Send exception to main process
                            exception = 'Vector file does not exist under %s/%s/%s.npy ' \
                                        'and could not load raw text files to construct it. ' \
                                        'Please provide corpus_folder in data config' \
                                        % (sim_matrix_config['context_path'][x], qrel[0], qrel[2])
                            qids.append(exception)
                            cwids.append(None)
                            labels.append(None)
                            ngram_mats.append(None)
                            query_idfs.append(None)
                            context_vecs.append(None)
                            docs.append(None)
                            queries.append(None)
                            ngram_masks.append(None)
                            continue

                        article = read_file("%s/%s" % (corpus_folder, qrel[2]))
                        article = preprocess_text(article, tokenize=True, all_lower=True,
                                                  stopw=remove_stopwords).split()

                    # Build context vector
                    if len(aux_s) > 1:
                        # Join topic + description terms
                        terms = []
                        for k in aux_s:
                            terms += query[k]
                    else:
                        terms = query[x]
                    context_vec = build_context_vector(terms, article, sim_matrix_config['context_window'],
                                                       embeddings)

                    # Save vector
                    if not os.path.exists('%s/%s' % (sim_matrix_config['context_path'][x], qrel[0])):
                        try:
                            os.makedirs('%s/%s' % (sim_matrix_config['context_path'][x], qrel[0]))
                        except FileExistsError:
                            # Another worker create the folder at the same time
                            pass
                    np.save(
                        '%s/%s/%s.npy' % (sim_matrix_config['context_path'][x], qrel[0], qrel[2]),
                        context_vec
                    )

            # Query and doc lengths
            len_doc, len_query = sim_matrix.shape[1], sim_matrix.shape[0]
            if sim_matrix_config['min_doc_len'] > len_doc:
                continue  # skip to next iteration
            if sim_matrix_config['use_context']:
                assert len_doc == context_vec.shape[0], "Redo matrices and context vectors, some have different size"
            if sim_matrix_config['use_masking']:
                # Build mask: zeros to padded values and ones otherwise
                mask = np.ones((len_query, len_doc))

            for n_gram in n_grams:
                if len_doc > sim_matrix_config['max_doc_len']:
                    # Document too long -> choose positions to keep
                    # FIXME: change (0,1) to (0,0) ?
                    rmat = np.pad(
                        sim_matrix,
                        pad_width=((0, sim_matrix_config['max_query_len'] - len_query), (0, 1)),
                        mode='constant',
                        constant_values=0
                    ).astype(np.float32)

                    # Cut some positions with given pos_method
                    selected_inds = select_pos_func(sim_matrix, sim_matrix_config['max_doc_len'], n_gram)
                    rmat = rmat[:, selected_inds]

                    if sim_matrix_config['use_masking']:
                        rmask = np.pad(
                            mask,
                            pad_width=((0, sim_matrix_config['max_query_len'] - len_query),
                                       (0, 1)),
                            mode='constant',
                            constant_values=0
                        ).astype(np.float32)
                        rmask = rmask[:, selected_inds]

                    if n_gram == min(n_grams) and sim_matrix_config['use_context']:
                        context_vec = context_vec[selected_inds]

                else:
                    # Just pad document
                    rmat = np.pad(
                        sim_matrix,
                        pad_width=((0, sim_matrix_config['max_query_len'] - len_query),
                                   (0, sim_matrix_config['max_doc_len'] - len_doc)),
                        mode='constant',
                        constant_values=0
                    ).astype(np.float32)

                    if sim_matrix_config['use_masking']:
                        rmask = np.pad(
                            mask,
                            pad_width=((0, sim_matrix_config['max_query_len'] - len_query),
                                       (0, sim_matrix_config['max_doc_len'] - len_doc)),
                            mode='constant',
                            constant_values=0
                        ).astype(np.float32)

                    if n_gram == min(n_grams) and sim_matrix_config['use_context']:
                        context_vec = np.pad(
                            context_vec,
                            pad_width=((0, sim_matrix_config['max_doc_len'] - len_doc),),
                            mode='constant',
                            constant_values=0
                        )

                # Save matrix
                ngram_mat[n_gram] = rmat
                if sim_matrix_config['use_masking']:
                    ngram_mask[n_gram] = rmask

            if sim_matrix_config['use_context']:
                # hack so that we have the same shape as the sim matrices
                context_vec = np.array([context_vec for _ in range(sim_matrix_config['max_query_len'])], dtype=np.float32)

        # Save
        qids.append(qid)
        cwids.append(cwid)
        labels.append(label)
        ngram_mats.append(ngram_mat)
        query_idfs.append(query_idf)
        context_vecs.append(context_vec)
        docs.append(document)
        queries.append(q_terms)
        ngram_masks.append(ngram_mask)

    # Send info back
    q_send.put((qids, cwids, labels, ngram_mats, query_idfs, context_vecs, docs, queries, ngram_masks))


def build_sim_matrix(query, document, embeddings, mode='cosine'):
    """
    Build similarity matrix, cosine similarity or word mover's distance
    """

    assert mode in ['cosine', 'wmd'], 'Only cosine and wmd allowed'

    matrix = np.zeros((len(query), len(document)))
    for i, w_q in enumerate(query):
        for j, w_d in enumerate(document):
            if mode == 'cosine':
                matrix[i][j] = embeddings.similarity(w_q, w_d)
            elif mode == 'wmd':
                matrix[i][j] = embeddings.wmdistance(w_q, w_d)
    return matrix


def build_context_vector(query, document, context_window, embeddings):
    """
    Build context vector
    """

    # Get query embeddings and average them into a single unit vector
    query_vector = list(map(lambda x: embeddings[x], query))
    query_vector = unitvec(np.mean(query_vector, axis=0))

    context = []
    for i in range(len(document)):
        begin = i - context_window
        if begin < 0:
            begin = 0

        end = i + context_window + 1
        if end > len(document):
            end = len(document)

        # Build doc vector
        doc_vector = list(map(lambda x: embeddings[x], document[begin:end]))
        doc_vector = unitvec(np.mean(doc_vector, axis=0))

        # Append context
        context.append(np.dot(query_vector, doc_vector))
    return np.array(context)


def read_corpus(dset_files, topics_files, corpus_folder, dset_folder,
                use_label_encoder=True, use_topic=True, use_description=True,
                sim_matrix_config=None, query_idf_config=None,
                num_negative=1, embeddings_path=None, remove_stopwords=False,
                include_spam=True, shuffle_seed=None, custom_loss=False):
    """
    Reads files needed to build a corpus
    """

    # Assertions
    assert use_topic or use_description
    assert not include_spam, "'include_spam' has to be set to False"
    if sim_matrix_config:
        assert 'use_static_matrices' in sim_matrix_config, "Need to provide 'use_static_matrices'"
        assert 'matrices_path' in sim_matrix_config, "Provide path to load/store similarity matrices"
        if use_topic:
            assert 'topic' in sim_matrix_config['matrices_path'].keys()
        if use_description:
            assert 'description' in sim_matrix_config['matrices_path'].keys()
        assert 'use_masking' in sim_matrix_config, "Need 'use_masking' option when building sim_matrix"
        assert 'ngrams' in sim_matrix_config, "Need 'ngrams' when building sim_matrix"
        assert 'max_doc_len' in sim_matrix_config and 'max_query_len' in sim_matrix_config, \
            "'max_doc_len' and 'max_query_len' has to be provided when building sim_matrix"
        if 'min_doc_len' not in sim_matrix_config:
            sim_matrix_config['min_doc_len'] = 0
        else:
            assert sim_matrix_config['min_doc_len'] >= 0
        assert 'pos_method' in sim_matrix_config, "Need to provide 'pos_method' to deal with too long documents"
        assert sim_matrix_config['pos_method'] == 'firstk', "Models only available for firstk pos_method"
        assert 'use_context' in sim_matrix_config, "Need to provide 'use_context' when building sim_matrix"
        assert sim_matrix_config['pos_method'] == 'firstk' or not sim_matrix_config['use_context'], \
            "context is misaligned if we aren't using firstk"
        if sim_matrix_config['use_context']:
            assert not sim_matrix_config['use_static_matrices'], "Context + sim_matrix_config['use_static_matrices'] not supported, yet"
            assert 'context_path' in sim_matrix_config, "Need to provide 'context_path' when using context"
            assert 'context_window' in sim_matrix_config, "Need to provide 'context_window' when using context"
            if use_topic and use_description:
                assert 'topic_description' in sim_matrix_config['context_path'].keys()
            elif use_topic:
                assert 'topic' in sim_matrix_config['context_path'].keys()
            elif use_description:
                assert 'description' in sim_matrix_config['context_path'].keys()
    else:
        raise Exception('sim_matrix_config has to be provided now')

    if query_idf_config['use_query_idf']:
        assert 'max_query_len' in query_idf_config, \
            "'max_query_len' has to be provided when building query_idf"
        assert 'idf_vectors_path' in query_idf_config, \
            "Need to provide path for IDF query vectors, or run bin/construct_query_idf_vectors.py to build them"
        if use_topic:
            assert 'topic' in query_idf_config['idf_vectors_path']
        if use_description:
            assert 'description' in query_idf_config['idf_vectors_path']

    if sim_matrix_config and query_idf_config['use_query_idf']:
        assert sim_matrix_config['max_query_len'] == query_idf_config['max_query_len']

    if not sim_matrix_config['use_static_matrices']:
        assert embeddings_path is not None and corpus_folder is not None, \
            'Please provide embeddings and corpus_folder when NOT using "use_static_matrices"'
        assert 'tokenizer_vocabsize.pickle' in os.listdir(os.path.dirname(embeddings_path)), "Construct matrix embeddings first"
        assert 'embeddings_matrix.npy' in os.listdir(os.path.dirname(embeddings_path)), "Construct matrix embeddings first"

        # Get tokenizer
        with open('%s/tokenizer_vocabsize.pickle' % os.path.dirname(embeddings_path), 'rb') as handle:
            tokenizer, _ = pickle.load(handle)

        from keras.preprocessing.sequence import pad_sequences

    if custom_loss:
        def labels2gains(labels):
            denominator = sum([2. ** x - 1. for x in labels])
            return [(2. ** x - 1.) / denominator for x in labels]

    # Shuffle seed
    if shuffle_seed:
        np.random.seed(shuffle_seed)

    # Dict to hold dataset
    dset = {
        'input': defaultdict(list),
        'output': defaultdict(list)
    }

    # Convert labels
    if use_label_encoder:
        label2tlabel = {
            4: 2,
            3: 2,
            2: 2,
            1: 1,
            0: 0,
            -2: 0
        }
    else:
        label2tlabel = {
            4: 4,
            3: 3,
            2: 2,
            1: 1,
            0: 0,
            -2: -2
        }

    # Read QRels files
    qrels = []
    for file in dset_files:
        # Pass files for rerank
        if not isinstance(file, dict):
            qrels += read_qrels(file)

    # Read topics
    query_id2text = read_query(topics_files)

    # Embeddings
    select_pos_func = None
    embeddings = None
    n_grams = None
    if sim_matrix_config:
        if sim_matrix_config['pos_method'] == 'firstk':
            n_grams = [max(sim_matrix_config['ngrams'])]
        else:
            n_grams = sim_matrix_config['ngrams']

        # Try to load embeddings (may not be necessary so don't throw exception)
        if embeddings_path and os.path.isfile(embeddings_path):
            embeddings = Word2Vec.load(embeddings_path)

        qid_ngram_cwid_mat = defaultdict(lambda: defaultdict(dict))  # dict[qid][ngram][cwid] = sim_mat

        if sim_matrix_config['use_context']:
            qid_cwid_context = defaultdict(dict)

        if sim_matrix_config['use_masking']:
            qid_ngram_cwid_mask = defaultdict(lambda: defaultdict(dict))

        # Function to select positions when document too long
        select_pos_func = eval('select_pos_%s' % sim_matrix_config['pos_method'])

    if query_idf_config['use_query_idf']:
        query_idfs = dict()

    # Variable for sim_matrix_config['use_static_matrices'] = False
    qid_cwid2query_doc = defaultdict(lambda: defaultdict())

    # Iterate through all the qrels to build input/output
    qid_label_cwids = defaultdict(lambda: defaultdict(list))  # dict[qid][l] = [cw1,cw2,...]
    qid_cwid_label = defaultdict(lambda: defaultdict())  # dict[qid][cw1] = l
    labels = []

    # Initialize pool and queues
    q_process_recv = mp.Queue(maxsize=mp.cpu_count())
    q_process_send = mp.Queue(maxsize=mp.cpu_count())
    pool = mp.Pool(
        mp.cpu_count(),
        initializer=process,
        initargs=(q_process_recv, q_process_send, query_id2text, label2tlabel, corpus_folder,
                  remove_stopwords, use_topic, use_description, sim_matrix_config['use_static_matrices'], select_pos_func,
                  sim_matrix_config, query_idf_config, embeddings, n_grams, include_spam)
    )

    # Send qrels
    for qrel in tqdm(qrels, desc="%s%s%s data for %s" % ('\033[1;33m', "Collecting", '\033[0;0m', dset_folder)):
        q_process_recv.put(qrel)  # blocks until q below its max size

    # Tell workers we're done
    for _ in range(mp.cpu_count()):
        q_process_recv.put(None)

    # Receive info
    for _ in range(mp.cpu_count()):
        for qid, cwid, label, ngram_mat, query_idf, context_vec, doc, query_terms, ngram_mask in zip(*q_process_send.get()):
            # Check for exceptions
            if cwid is None:
                raise Exception(qid)  # Exception comes in the qid variable

            labels.append(label)
            qid_label_cwids[qid][label].append(cwid)
            qid_cwid_label[qid][cwid] = label

            # Save query/doc text
            if not sim_matrix_config['use_static_matrices']:
                qid_cwid2query_doc[qid][cwid] = (query_terms, doc)
            else:
                # Save matrices
                for ng in ngram_mat:
                    qid_ngram_cwid_mat[qid][ng][cwid] = ngram_mat[ng]
                    if sim_matrix_config['use_masking']:
                        qid_ngram_cwid_mask[qid][ng][cwid] = ngram_mask[ng]
                if sim_matrix_config['use_context']:
                    qid_cwid_context[qid][cwid] = context_vec

            # Save Query IDF
            if query_idf_config['use_query_idf'] and qid not in query_idfs:
                query_idfs[qid] = query_idf

    # Close pool
    pool.close()
    pool.join()

    # FIXME: Needs to be more general to check if is for train
    if dset_folder.split('/')[-1][:5] == 'train':
        """ Logic to construct softmax with 1 positive doc and num_negative negative docs for every training point"""

        label_count = Counter(labels)  # label counter

        # Get percentage of positive (> 0) labels
        total_count = sum([label_count[l] for l in label_count if l > 0])
        sample_label_prob = {l: label_count[l] / float(total_count) for l in label_count if l > 0}

        label_qid_prob = dict()  # dict[l][qid] = n labels l for that qid / total labels l
        for l in sample_label_prob:
            label_qid_prob[l] = {
                qid: len(qid_label_cwids[qid][l]) / float(label_count[l])
                for qid in qid_label_cwids
                if l in qid_label_cwids[qid]
            }

        # dict[l] = [n labels l for qid_1 / total labels l, ..., n labels l for qid_n / total labels l]
        sample_label_qid_prob = {l: [label_qid_prob[l][qid] if qid in label_qid_prob[l] else 0
                                     for qid in qid_label_cwids]
                                 for l in label_qid_prob}

        # Contruct positive and negative batch
        pos_batch = defaultdict(list)  # 1 positive example per softmax
        pos_batch_masks = defaultdict(list)  # 1 positive example per softmax
        pos_batch_texts = defaultdict(list)  # 1 positive example per softmax
        neg_batch = defaultdict(lambda: defaultdict(list))  # num_negative examples per softmax
        neg_batch_masks = defaultdict(lambda: defaultdict(list))  # num_negative examples per softmax
        neg_batch_texts = defaultdict(lambda: defaultdict(list))  # num_negative examples per softmax
        positive_cwids, negative_cwids = [], []
        for _ in range(num_negative):
            negative_cwids.append([])  # Append list for every new negative sample
        pos_context_batch = []
        neg_context_batch = defaultdict(list)
        qids, qidf_batch, ys = [], [], []

        # Select every label
        selected_labels = np.array([l for l in label_count if l > 0])

        # Choose N positive samples (N is the total number of training samples in here)
        selected_labels = np.concatenate(
            [selected_labels,
             np.random.choice([l for l in sorted(sample_label_prob)], size=len(labels), replace=True,
                              p=[sample_label_prob[l] for l in sorted(sample_label_prob)])]
        )
        label_counter = Counter(selected_labels)
        for label in label_counter:
            # Select every qid, if label selected exists for that qid
            selected_qids = np.array([qid for qid in qid_label_cwids.keys() if qid in label_qid_prob[label]])

            size_left = label_counter[label] - len(selected_qids)
            # For the rest, select random qids equal to the number of the selected label
            # (this is necessary to train each of the selected labels)
            if size_left > 0:
                selected_qids = np.concatenate(
                    [selected_qids,
                     np.random.choice(list(qid_label_cwids.keys()),
                                      size=size_left, replace=True,
                                      p=sample_label_qid_prob[label])]
                )
            qid_counter = Counter(selected_qids)
            for qid in qid_counter:
                if qid_counter[qid] == 0:
                    continue

                # Check if qid+label has inferior document to pair with (2-1, 1-0)
                neg_labels = [nl for nl in reversed(range(label)) if nl in qid_label_cwids[qid]]
                # If it doesnt exist, leave it
                if not neg_labels:
                    continue

                # Positive and negative CWids
                pos_cwids = qid_label_cwids[qid][label]
                if custom_loss:
                    neg_cwids = []
                    for nl in neg_labels:
                        neg_cwids += qid_label_cwids[qid][nl]
                else:
                    neg_cwids = qid_label_cwids[qid][neg_labels[0]]
                n_pos, n_neg = len(pos_cwids), len(neg_cwids)

                # Add all and then randomly picked positive matrices of that qid to positive batch
                idx_poses = np.array(list(range(n_pos)))
                size_left = qid_counter[qid] - len(idx_poses)
                if size_left > 0:
                    idx_poses = np.concatenate(
                        [idx_poses, np.random.choice(list(range(n_pos)), size=size_left, replace=True)]
                    )

                if sim_matrix_config['use_static_matrices']:
                    min_ngram = min(qid_ngram_cwid_mat[qid])
                    ngrams = qid_ngram_cwid_mat[qid]
                else:
                    min_ngram = max(n_grams)
                    ngrams = [min_ngram]

                for ngram in ngrams:
                    for pi in idx_poses:
                        p_cwid = pos_cwids[pi]
                        if sim_matrix_config['use_static_matrices']:
                            pos_batch[ngram].append(qid_ngram_cwid_mat[qid][ngram][p_cwid])
                            if sim_matrix_config['use_masking']:
                                pos_batch_masks[ngram].append(qid_ngram_cwid_mask[qid][ngram][p_cwid])

                        if ngram == min_ngram:
                            if not sim_matrix_config['use_static_matrices']:
                                pos_batch_texts[max(n_grams)].append(qid_cwid2query_doc[qid][p_cwid])

                            if sim_matrix_config['use_context']:
                                pos_context_batch.append(qid_cwid_context[qid][p_cwid])
                            if not custom_loss:
                                # When using custom loss, append labels at the end
                                ys.append(1)  # positive label
                            qids.append(qid)  # save qid
                            positive_cwids.append(p_cwid)  # save cwid

                # Add num_negative randomly picked negative matrices
                # per positive example of that qid to negative batch
                for neg_ind in range(num_negative):
                    idx_negs = np.random.choice(list(range(n_neg)), size=len(idx_poses), replace=True)
                    for ngram in ngrams:
                        for ni in idx_negs:
                            n_cwid = neg_cwids[ni]
                            if sim_matrix_config['use_static_matrices']:
                                neg_batch[ngram][neg_ind].append(qid_ngram_cwid_mat[qid][ngram][n_cwid])
                                if sim_matrix_config['use_masking']:
                                    neg_batch_masks[ngram][neg_ind].append(qid_ngram_cwid_mask[qid][ngram][n_cwid])

                            if ngram == min_ngram:
                                if not sim_matrix_config['use_static_matrices']:
                                    neg_batch_texts[max(n_grams)][neg_ind].append(qid_cwid2query_doc[qid][n_cwid])
                                negative_cwids[neg_ind].append(n_cwid)
                                if sim_matrix_config['use_context']:
                                    neg_context_batch[neg_ind].append(qid_cwid_context[qid][n_cwid])

                # Add Query IDF to its batch
                if query_idf_config['use_query_idf']:
                    qidf_batch.append(
                        query_idfs[qid].reshape((1, query_idf_config['max_query_len'], 1)).repeat(len(idx_poses),
                                                                                                  axis=0)
                    )

        # INPUT
        dset['input']['meta-data'] = {
            'qids': qids,
            'pos_cwids': positive_cwids,
            'neg_cwids': list(zip(*negative_cwids))
        }

        if custom_loss:
            # When using custom loss, append labels at the end
            for qid, pos_cwid, neg_cwids in zip(qids, positive_cwids, dset['input']['meta-data']['neg_cwids']):
                pos_label = 1. * qid_cwid_label[qid][pos_cwid]
                neg_labels = [1. * qid_cwid_label[qid][x] for x in neg_cwids]
                ys.append(
                    labels2gains([pos_label] + neg_labels)
                )
        if not sim_matrix_config['use_static_matrices']:
            # Encode & pad texts
            queries, documents = [], []
            for sample in pos_batch_texts[max(n_grams)]:
                queries.append(sample[0])
                documents.append(sample[1])
            dset['input']['pos_query'] = pad_sequences(tokenizer.texts_to_sequences(queries),
                                                       maxlen=sim_matrix_config['max_doc_len'],
                                                       padding='post', truncating='post')
            dset['input']['pos_doc'] = pad_sequences(tokenizer.texts_to_sequences(documents),
                                                     maxlen=sim_matrix_config['max_doc_len'],
                                                     padding='post', truncating='post')

            for neg_ind in range(num_negative):
                queries, documents = [], []
                for sample in neg_batch_texts[max(n_grams)][neg_ind]:
                    queries.append(sample[0])
                    documents.append(sample[1])
                dset['input']['neg%d_query' % neg_ind] = pad_sequences(tokenizer.texts_to_sequences(queries),
                                                                       maxlen=sim_matrix_config['max_doc_len'],
                                                                       padding='post', truncating='post')
                dset['input']['neg%d_doc' % neg_ind] = pad_sequences(tokenizer.texts_to_sequences(documents),
                                                                     maxlen=sim_matrix_config['max_doc_len'],
                                                                     padding='post', truncating='post')

        if sim_matrix_config['use_static_matrices']:
            for ngram in pos_batch:
                dset['input']['pos_ngram_%d' % ngram] = np.array(pos_batch[ngram])
                if sim_matrix_config['use_masking']:
                    dset['input']['pos_ngram_mask_%d' % ngram] = np.array(pos_batch_masks[ngram])

                for neg_ind in range(num_negative):
                    dset['input']['neg%d_ngram_%d' % (neg_ind, ngram)] = \
                        np.array(np.array(neg_batch[ngram][neg_ind]))
                    if sim_matrix_config['use_masking']:
                        dset['input']['neg%d_ngram_mask_%d' % (neg_ind, ngram)] = np.array(neg_batch_masks[ngram][neg_ind])

            if sim_matrix_config['use_context']:
                dset['input']['pos_context'] = np.array(pos_context_batch)
                for neg_ind in range(num_negative):
                    dset['input']['neg%d_context' % neg_ind] = np.array(np.array(neg_context_batch[neg_ind]))

        if query_idf_config['use_query_idf']:
            dset['input']['query_idf'] = np.concatenate(qidf_batch, axis=0)

        # OUTPUT
        dset['output']['tags'] = np.array(ys)

    else:
        # Dev/Test
        doc_vec, doc_texts, doc_masks, q_idfs, contexts = \
            defaultdict(list), defaultdict(list), defaultdict(list), list(), list()
        qids, cwids = list(), list()
        ys = list()
        for qid in qid_label_cwids:
            if sim_matrix_config['use_static_matrices']:
                min_ngram = min(qid_ngram_cwid_mat[qid])
                ngrams = qid_ngram_cwid_mat[qid]
            else:
                min_ngram = max(n_grams)
                ngrams = [min_ngram]

            for ngram in ngrams:
                if sim_matrix_config['use_static_matrices']:
                    aux_cwids = qid_ngram_cwid_mat[qid][ngram]
                else:
                    aux_cwids = qid_cwid2query_doc[qid]

                for cwid in aux_cwids:
                    if sim_matrix_config['use_static_matrices']:
                        doc_vec[ngram].append(qid_ngram_cwid_mat[qid][ngram][cwid])
                        if sim_matrix_config['use_masking']:
                            doc_masks[ngram].append(qid_ngram_cwid_mask[qid][ngram][cwid])
                        if sim_matrix_config['use_context'] and ngram == min_ngram:
                            contexts.append(qid_cwid_context[qid][cwid])

                    if ngram == min_ngram:
                        if not sim_matrix_config['use_static_matrices']:
                            doc_texts[max(n_grams)].append(qid_cwid2query_doc[qid][cwid])

                        # Add query IDF in last iteration
                        if query_idf_config['use_query_idf']:
                            q_idfs.append(query_idfs[qid].reshape((1, query_idfs[qid].shape[0], 1)))

                        # Add label in last iteration
                        ys.append(qid_cwid_label[qid][cwid])

                        # Save qid and cwid
                        qids.append(qid)
                        cwids.append(cwid)

        # INPUT
        dset['input']['meta-data'] = {
            'qids': qids,
            'cwids': cwids
        }
        if not sim_matrix_config['use_static_matrices']:
            # Encode & pad texts
            queries, documents = [], []
            for sample in doc_texts[max(n_grams)]:
                queries.append(sample[0])
                documents.append(sample[1])
            dset['input']['query'] = pad_sequences(tokenizer.texts_to_sequences(queries),
                                                       maxlen=sim_matrix_config['max_doc_len'], padding='post')
            dset['input']['doc'] = pad_sequences(tokenizer.texts_to_sequences(documents),
                                                     maxlen=sim_matrix_config['max_doc_len'], padding='post')

        else:
            for ngram in doc_vec:
                dset['input']['doc_ngram_%d' % ngram] = np.array(np.array(doc_vec[ngram]))
                if sim_matrix_config['use_masking']:
                    dset['input']['doc_ngram_mask_%d' % ngram] = np.array(np.array(doc_masks[ngram]))

            if sim_matrix_config['use_context']:
                dset['input']['doc_context'] = np.array(np.array(contexts))

        if query_idf_config['use_query_idf']:
            dset['input']['query_idf'] = np.concatenate(q_idfs, axis=0)

        # OUTPUT
        # NOTE: These labels don't matter in NDCG20/ERR20 as the model will be evaluated
        # by ranking probabilities, not by comparing true label vs predicted label
        dset['output']['tags'] = np.array(ys)

    return dset


class Data(DataTemplate):
    """
    Web Track data
    """

    def __init__(self, config):
        # Inherited
        # self.config, self.datasets
        DataTemplate.__init__(self, config)

        assert "topics_files" in config, \
            "Need to provide 'topics_files' in config"

        assert False not in [os.path.isfile(file) for file in config['topics_files']], \
            "Not all files exist in %s" % config['topics_files']

        # Init variables for retrain
        self.qid2cwid_label, self.qid_cwid_label = None, None
        self.qid_cwid_data = None
        if 'retrain_mode' not in config:
            self.retrain_mode = 1
        else:
            self.retrain_mode = config['retrain_mode']
            assert self.retrain_mode in [0, 1], \
                "'retrain_mode' has to be either 0 (negative samples can have different labels) " \
                "or 1 (negative samples are always 1 rank less than positive sample)"

        if 'corpus_folder' not in config:
            config['corpus_folder'] = None

        if 'embeddings_path' not in config:
            config['embeddings_path'] = None

        if 'include_spam' not in config:
            config['include_spam'] = True

        if 'use_label_encoder' not in config:
            config['use_label_encoder'] = True

        # LOAD DATA
        for dset in config['datasets'].keys():
            dset_files = config['datasets'][dset]

            assert isinstance(dset_files, list), \
                "Expected a list for each type of dataset"

            # SANITY CHECKS
            # Assert files exist
            assert False not in [os.path.isfile(file) for file in dset_files
                                 if not isinstance(file, dict)], \
                "Not all files exist in %s" % dset_files

            self.datasets[dset] = read_corpus(
                dset_files,
                config['topics_files'],
                config['corpus_folder'],
                use_topic=config['use_topic'],
                use_label_encoder=config['use_label_encoder'],
                use_description=config['use_description'],
                sim_matrix_config=config['sim_matrix_config'],
                embeddings_path=config['embeddings_path'],
                query_idf_config=config['query_idf_config'],
                num_negative=config['num_negative'],
                remove_stopwords=config['remove_stopwords'],
                dset_folder="%s/%s" % (config['name'], dset),
                include_spam=config['include_spam'],
                shuffle_seed=config['shuffle_seed'],
                custom_loss=config['custom_loss']
            )
            self.nr_samples[dset] = \
                len(self.datasets[dset]['output']['tags'])

    def size(self, set_name):
        return self.nr_samples[set_name]

    def batches(self, set_names=None, batch_size=1, features_model=None,
                no_output=False, shuffle_seed=None):
        """
        Returns iterator over batches of features as

        {'input': { 'feat1': [[]], ... } 'output': { 'feat-a': [[]], ... }}

        If a model instance is provided, the features are extracted instead
        """

        assert isinstance(set_names, list), "Provide list of set_names, even if an one-element list"

        # Get all datasets
        dsets, nr_examples = [], 0
        for set_name in set_names:
            dsets.append(self.datasets[set_name])
            nr_examples += self.nr_samples[set_name]

        dset = {side: {k:v for k, v in dsets[0][side].items()} for side in ['input', 'output']}
        for side in ['input', 'output']:
            d_keys = dset[side].keys()

            # Assert that datasets are of the same type and can be joined
            for d in dsets[1:]:
                assert all(x in d[side] for x in d_keys)

            # Join datasets
            for d in dsets[1:]:
                for k in d_keys:
                    if isinstance(d[side][k], np.ndarray):
                        dset[side][k] = np.concatenate((dset[side][k], d[side][k]))
                    elif isinstance(d[side][k], list):
                        dset[side][k] += d[side][k]
                    elif isinstance(d[side][k], dict):
                        for x in d[side][k]:
                            if isinstance(d[side][k][x], np.ndarray):
                                dset[side][k][x] = np.concatenate((dset[side][k][x], d[side][k][x]))
                            elif isinstance(d[side][k][x], list):
                                dset[side][k][x] += d[side][k][x]
                            else:
                                raise Exception("Could not merge datasets")
                    else:
                        raise Exception("Could not merge datasets")

        if batch_size is None:
            # Get all data
            nr_batch = 1
            batch_size = nr_examples
        else:
            nr_batch = int(np.ceil(nr_examples * 1. / batch_size))

        if shuffle_seed:
            # Shuffle data
            idx = list(range(len(dset['output']['tags'])))
            random_state = np.random.RandomState(shuffle_seed)
            random_state.shuffle(idx)

            # Sort
            for key, value in dset['input'].items():
                if key != 'meta-data':
                    if isinstance(value, np.ndarray):
                        dset['input'][key] = value[idx]
                    else:
                        dset['input'][key] = [value[i] for i in idx]
                else:
                    for mkey, mvalue in dset['input']['meta-data'].items():
                        if isinstance(value, np.ndarray):
                            dset['input'][mkey] = mvalue[idx]
                        else:
                            dset['input']['meta-data'][mkey] = [
                                mvalue[i] for i in idx
                            ]
            for key, value in dset['output'].items():
                if isinstance(value, np.ndarray):
                    dset['input'][key] = value[idx]
                else:
                    dset['output'][key] = [value[i] for i in idx]

        data = []
        # Ignore output when solicited
        if no_output:
            data_sides = [key for key in dset.keys() if key != 'output']
        else:
            data_sides = dset.keys()
        for batch_n in range(nr_batch):

            # Collect data for this batch
            data_batch = {}
            for side in data_sides:
                data_batch[side] = {}
                for feat_name, values in dset[side].items():
                    if feat_name == 'meta-data':
                        data_batch[side][feat_name] = {}
                        for key in values.keys():
                            data_batch[side][feat_name][key] = values[key][
                                                               batch_n * batch_size:(batch_n + 1) * batch_size
                                                               ]
                    else:
                        data_batch[side][feat_name] = values[batch_n * batch_size:(batch_n + 1) * batch_size]

            # If feature extractors provided, return features instead
            if features_model is not None:
                feat_batch = features_model(**data_batch)
            else:
                feat_batch = data_batch

            data.append(feat_batch)

        return DataIterator(data, nr_samples=nr_examples)

    def retrain_batches(self, data, predict_function, retrain_qrel_files, batch_size=1,
                        features_model=None, no_output=False, shuffle_seed=None):
        """
        Returns DataIterator with only examples wrong classified
        This function was only made for train data

        :param data: DataIterator, given by Data.batches function
        :param predict_function: model function to get predictions of batch input
        :param retrain_qrel_files: list of original qrel files to get original label
        """

        assert isinstance(data, DataIterator)

        print("%sConstructing retrain dataset%s" % ('\033[1;33m', '\033[0;0m'), end='', flush=True)
        start_time = time.time()

        # Remove previous retrain
        if 'retrain' in self.datasets:
            del self.datasets['retrain']
            gc.collect()

        if self.qid2cwid_label is None:
            # Read QRels files
            qrels = []
            for file in retrain_qrel_files:
                # Pass files for rerank
                qrels += read_qrels(file)

            # Construct qid2cwid_label and qid_cwid_label
            self.qid2cwid_label = defaultdict(list)
            self.qid_cwid_label = defaultdict(dict)
            for qrel in qrels:
                if int(qrel[3]) != -2 or self.config['include_spam']:
                    self.qid2cwid_label[int(qrel[0])].append((qrel[2], int(qrel[3])))  # (cwid, label)
                    self.qid_cwid_label[int(qrel[0])][qrel[2]] = int(qrel[3])

        # Get sorted documents (cwids) for each qid by its score and save data for later
        qid2sorted_cwids = defaultdict(list)
        if self.qid_cwid_data is None:
            self.qid_cwid_data = defaultdict(lambda: defaultdict(dict))  # will save all (qid, cwid) data pairs
            for batch in data:
                # Put all document under new batch
                new_input_batch = {
                    key.replace('pos', 'doc'): [] for key in batch['input'] if 'pos' in key
                }
                new_input_batch['query_idf'] = []
                new_input_batch['meta-data'] = {
                    'qids': [],
                    'cwids': []
                }

                for i, (qid, pos_cwid, neg_cwids) in enumerate(zip(batch['input']['meta-data']['qids'],
                                                                   batch['input']['meta-data']['pos_cwids'],
                                                                   batch['input']['meta-data']['neg_cwids'])):

                    # Get positive matrices
                    if pos_cwid not in self.qid_cwid_data[qid]:
                        for key in new_input_batch:
                            if key == 'meta-data':
                                # Save meta-data
                                new_input_batch[key]['qids'].append(qid)
                                new_input_batch[key]['cwids'].append(pos_cwid)
                            elif 'doc' in key:
                                new_input_batch[key].append(batch['input'][key.replace('doc', 'pos')][i])
                                self.qid_cwid_data[qid][pos_cwid][key] = new_input_batch[key][-1]
                            else:
                                new_input_batch[key].append(batch['input'][key][i])
                                self.qid_cwid_data[qid][pos_cwid][key] = new_input_batch[key][-1]

                    # Get negative matrices
                    for neg_ind in range(self.config['num_negative']):
                        if neg_cwids[neg_ind] not in self.qid_cwid_data[qid]:
                            for key in new_input_batch:
                                if key == 'meta-data':
                                    # Save meta-data
                                    new_input_batch[key]['qids'].append(qid)
                                    new_input_batch[key]['cwids'].append(neg_cwids[neg_ind])
                                elif 'doc' in key:
                                    new_input_batch[key].append(batch['input'][key.replace('doc', 'neg%d' % neg_ind)][i])
                                    self.qid_cwid_data[qid][neg_cwids[neg_ind]][key] = new_input_batch[key][-1]
                                else:
                                    new_input_batch[key].append(batch['input'][key][i])
                                    self.qid_cwid_data[qid][neg_cwids[neg_ind]][key] = new_input_batch[key][-1]

                if new_input_batch['query_idf']:  # condition to check if new_input_batch isnt empty
                    # Pass to numpy arrays
                    for key in new_input_batch:
                        if key != 'meta-data':  # No need for meta-data
                            new_input_batch[key] = np.array(new_input_batch[key])

                    # Get predictions of new batch and save them for each (qid, cwid) pair
                    scores = predict_function(new_input_batch)['probs']
                    assert len(new_input_batch['meta-data']['qids']) == len(scores)
                    for qid, cwid, score \
                            in zip(new_input_batch['meta-data']['qids'], new_input_batch['meta-data']['cwids'], scores):
                        qid2sorted_cwids[qid].append((cwid, score))  # add (cwid, score) to sort later

        else:
            # Collect all data into a batch
            new_input_batch = defaultdict(list)
            qids, cwids = [], []
            for qid in self.qid_cwid_data:
                for cwid in self.qid_cwid_data[qid]:
                    for key in self.qid_cwid_data[qid][cwid]:
                        new_input_batch[key].append(self.qid_cwid_data[qid][cwid][key])
                    qids.append(qid)
                    cwids.append(cwid)

            # Get predictions of new batch and save them for each (qid, cwid) pair
            scores = []
            for i in range(math.ceil(len(new_input_batch['query_idf'])/batch_size)):
                # Pass to numpy arrays
                mini_batch = dict()
                for key in new_input_batch:
                    mini_batch[key] = np.array(new_input_batch[key][i*batch_size:i*batch_size+batch_size])
                scores.extend(predict_function(mini_batch)['probs'])
            assert len(qids) == len(scores)
            for qid, cwid, score in zip(qids, cwids, scores):
                qid2sorted_cwids[qid].append((cwid, score))  # add (cwid, score) to sort later
        
        for qid in qid2sorted_cwids:
            # sort and prune score
            qid2sorted_cwids[qid] = sorted(qid2sorted_cwids[qid], key=lambda x: -x[1])
            qid2sorted_cwids[qid] = list(map(lambda x: x[0], qid2sorted_cwids[qid]))

        if self.config['include_spam']:
            lowest_label = -2
        else:
            lowest_label = 0

        # For each query, pick documents ordered by their (descending) score
        # Then pick documents with lower label that scored higher
        retrain_qid_poscwid_negcwids = defaultdict(lambda: defaultdict(list))
        for qid in qid2sorted_cwids:
            for cwid, label in sorted(self.qid2cwid_label[qid], key=lambda x: -x[1]):
                if label == lowest_label:
                    # Lowest label, wont find any higher
                    continue

                try:
                    # Get position of doc
                    cwid_rank_index = qid2sorted_cwids[qid].index(cwid)
                except ValueError:
                    # Document wasnt loaded during training, since they are selected randomly
                    warnings.warn(
                        'Query-Document pair %s-%s, with label %s, was not selected while loading train data'
                        % (qid, cwid, label)
                    )
                    continue

                # Check whether if there are docs with lower labels ranked higher
                for h_cwid in qid2sorted_cwids[qid][:cwid_rank_index]:
                    # Check if label found is inferior
                    if self.qid_cwid_label[qid][h_cwid] < label:
                        if self.retrain_mode == 1:
                            if self.qid_cwid_label[qid][h_cwid] == label - 1:
                                # Save this pair to retrain
                                retrain_qid_poscwid_negcwids[qid][cwid].append(h_cwid)
                        else:
                            # Save this pair to retrain
                            retrain_qid_poscwid_negcwids[qid][cwid].append(h_cwid)

        # Construct retrain corpus
        dset = {
            'input': defaultdict(list),
            'output': defaultdict(list)
        }
        for qid in retrain_qid_poscwid_negcwids:
            for pos_cwid in retrain_qid_poscwid_negcwids[qid]:
                # Pick chosen neg_cwids
                neg_cwids = retrain_qid_poscwid_negcwids[qid][pos_cwid]

                # Logic to rearrange them in batches of 'num_negative' neg_docs vs 1 pos_doc
                if len(neg_cwids) <= self.config['num_negative']:
                    # less docs than necessary => duplicate examples randomly
                    neg_docs_left = self.config['num_negative'] - len(neg_cwids)
                    neg_cwids += list(np.random.choice(neg_cwids, neg_docs_left))

                    # Add negative docs
                    for neg_ind, neg_cwid in enumerate(neg_cwids):
                        for key in self.qid_cwid_data[qid][neg_cwid]:
                            # Add input
                            if 'doc' in key:
                                dset['input'][key.replace('doc', 'neg%d' % neg_ind)].append(
                                    self.qid_cwid_data[qid][neg_cwid][key])

                    # Add positive doc
                    for key in self.qid_cwid_data[qid][pos_cwid]:
                        dset['input'][key.replace('doc', 'pos')].append(self.qid_cwid_data[qid][pos_cwid][key])

                    # Add output
                    dset['output']['tags'].append(1)

                else:
                    # More docs than necessary => divide in more than 1 sample
                    num_samples = math.ceil(1.*len(neg_cwids)/self.config['num_negative'])

                    for sample_no in range(int(num_samples)):
                        start = sample_no*self.config['num_negative']
                        end = start+self.config['num_negative']

                        # Check if sample is not finished (will only occur in last sample)
                        if len(neg_cwids) < end + 1:
                            # Duplicate examples randomly
                            neg_docs_left = end + 1 - len(neg_cwids)
                            neg_cwids += list(np.random.choice(neg_cwids, neg_docs_left))

                        # Add negative docs
                        for neg_ind, neg_cwid in enumerate(neg_cwids[start:start+self.config['num_negative']]):
                            for key in self.qid_cwid_data[qid][neg_cwid]:
                                # Add input
                                if 'doc' in key:
                                    dset['input'][key.replace('doc', 'neg%d' % neg_ind)].append(
                                        self.qid_cwid_data[qid][neg_cwid][key])

                        # Add positive doc
                        for key in self.qid_cwid_data[qid][pos_cwid]:
                            dset['input'][key.replace('doc', 'pos')].append(
                                self.qid_cwid_data[qid][pos_cwid][key])

                        # Add output
                        dset['output']['tags'].append(1)

        # Pass to numpy arrays
        for side in dset:
            for key in dset[side]:
                if key != 'meta-data':  # No need for meta-data
                    dset[side][key] = np.array(dset[side][key])

        print(" took %.2f seconds" % (time.time() - start_time))

        # Return DataIterator over this new dset
        self.datasets['retrain'] = dset
        self.nr_samples['retrain'] = \
            len(dset['output']['tags'])
        return self.batches('retrain', batch_size=batch_size, features_model=features_model,
                            no_output=no_output, shuffle_seed=shuffle_seed)
