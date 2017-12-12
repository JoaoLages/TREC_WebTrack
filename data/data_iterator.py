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


def process(q_recv, q_send, query_id2text, label2tlabel, corpus_folder,
            remove_stopwords, use_topic, use_description, select_pos_func,
            sim_matrix_config, query_idf_config, embeddings, n_grams, include_spam):

    qids, cwids, labels, ngram_mats, query_idfs, context_vecs = [], [], [], [], [], []

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
        query_idf, ngram_mat,  context_vec = None, dict(), None
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

        if query_idf_config:
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
                        q_send.put([exception], [None], [None], [None], [None])
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
                                "Please run bin/construct_query_idf_vectors.sh accordingly."\
                                % (query_idf_config['idf_vectors_path'][x], qrel[0])
                    q_send.put([exception], [None], [None], [None], [None])
                    break

            # Join topic+description vectors
            query_idf = np.concatenate(q_idfs, axis=0).astype(np.float32)
            # Pad to max length with zeros
            query_idf = np.pad(
                query_idf,
                pad_width=((0, sim_matrix_config['max_query_len'] - len(query_idf))),
                mode='constant',
                constant_values=-np.inf
            )

        if sim_matrix_config:
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
                        q_send.put([exception], [None], [None], [None], [None])
                        break

                    if article is None:
                        if corpus_folder is None:
                            # Send exception to main process
                            exception = 'Matrix file does not exist under %s/%s/%s.npy ' \
                                        'and could not load raw text files to construct it. ' \
                                        'Please provide corpus_folder in data config'\
                                        % (sim_matrix_config['matrices_path'][x], qrel[0], qrel[2])
                            q_send.put([exception], [None], [None], [None], [None])
                            break

                        article = read_file("%s/%s" % (corpus_folder, qrel[2]))
                        article = preprocess_text(article, tokenize=True, all_lower=True, stopw=remove_stopwords).split()

                    sim_matrix = build_sim_matrix(query[x], article, embeddings)
                    # Save matrix
                    if not os.path.exists('%s/%s' % (sim_matrix_config['matrices_path'][x], qrel[0])):
                        os.makedirs('%s/%s' % (sim_matrix_config['matrices_path'][x], qrel[0]))
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
                break

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
                                    'Please provide embeddings_path in data config'\
                                    % (sim_matrix_config['context_path'][x], qrel[0], qrel[2])
                        q_send.put([exception], [None], [None], [None], [None])
                        break

                    if article is None:
                        if corpus_folder is None:
                            # Send exception to main process
                            exception = 'Vector file does not exist under %s/%s/%s.npy ' \
                                        'and could not load raw text files to construct it. ' \
                                        'Please provide corpus_folder in data config' \
                                        % (sim_matrix_config['context_path'][x], qrel[0], qrel[2])
                            q_send.put([exception], [None], [None], [None], [None])
                            break

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
                        os.makedirs('%s/%s' % (sim_matrix_config['context_path'][x], qrel[0]))
                    np.save(
                        '%s/%s/%s.npy' % (sim_matrix_config['context_path'][x], qrel[0], qrel[2]),
                        context_vec
                    )

            # Query and doc lengths
            len_doc, len_query = sim_matrix.shape[1], sim_matrix.shape[0]
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

                    # FIXME: Why for only last n_gram??
                    if sim_matrix_config['use_context']:
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

                    if sim_matrix_config['use_context']:
                        context_vec = np.pad(
                            context_vec,
                            pad_width=((0, sim_matrix_config['max_doc_len'] - len_doc),),
                            mode='constant',
                            constant_values=0
                        )

                # Save matrix
                ngram_mat[n_gram] = rmat

            if sim_matrix_config['use_context']:
                # hack so that we have the same shape as the sim matrices
                context_vec = np.array([context_vec for _ in sim_matrix_config['max_query_len']], dtype=np.float32)

        # Save
        qids.append(qid)
        cwids.append(cwid)
        labels.append(label)
        ngram_mats.append(ngram_mat)
        query_idfs.append(query_idf)
        context_vecs.append(context_vec)

    # Send info back
    q_send.put((qids, cwids, labels, ngram_mats, query_idfs, context_vecs))


def build_sim_matrix(query, document, embeddings):
    """
    Build similarity matrix
    """
    matrix = np.zeros((len(query), len(document)))
    for i, w_q in enumerate(query):
        for j, w_d in enumerate(document):
            matrix[i][j] = embeddings.similarity(w_q, w_d)
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
                use_topic=True, use_description=True, sim_matrix_config=None,
                query_idf_config=None, num_negative=1, embeddings_path=None,
                remove_stopwords=False, include_spam=True, shuffle_seed=None):
    """
    Reads files needed to build a corpus
    """

    # Assertions
    assert use_topic or use_description
    if sim_matrix_config:
        assert 'matrices_path' in sim_matrix_config, "Provide path to load/store similarity matrices"
        if use_topic:
            assert 'topic' in sim_matrix_config['matrices_path'].keys()
        if use_description:
            assert 'description' in sim_matrix_config['matrices_path'].keys()
        assert 'ngrams' in sim_matrix_config, "Need 'ngrams' when building sim_matrix"
        assert 'max_doc_len' in sim_matrix_config and 'max_query_len' in sim_matrix_config, \
            "'max_doc_len' and 'max_query_len' has to be provided when building sim_matrix"
        assert 'pos_method' in sim_matrix_config, "Need to provide 'pos_method' to deal with too long documents"
        assert 'use_context' in sim_matrix_config, "Need to provide 'use_context' when building sim_matrix"
        assert sim_matrix_config['pos_method'] == 'firstk' or not sim_matrix_config['use_context'], \
            "context is misaligned if we aren't using firstk"
        if sim_matrix_config['use_context']:
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

    if query_idf_config:
        assert 'max_query_len' in query_idf_config, \
            "'max_query_len' has to be provided when building query_idf"
        assert 'idf_vectors_path' in query_idf_config, \
            "Need to provide path for IDF query vectors, or run bin/construct_query_idf_vectors.py to build them"
        if use_topic:
            assert 'topic' in query_idf_config['idf_vectors_path']
        if use_description:
            assert 'description' in query_idf_config['idf_vectors_path']

    if sim_matrix_config and query_idf_config:
        assert sim_matrix_config['max_query_len'] == query_idf_config['max_query_len']

    # Shuffle seed
    if shuffle_seed:
        np.random.seed(shuffle_seed)

    # Dict to hold dataset
    dset = {
        'input': defaultdict(list),
        'output': defaultdict(list)
    }

    # Convert labels
    label2tlabel = {
        4: 2,
        3: 2,
        2: 2,
        1: 1,
        0: 0,
        -2: 0
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

        # Function to select positions when document too long
        select_pos_func = eval('select_pos_%s' % sim_matrix_config['pos_method'])

    if query_idf_config:
        query_idfs = dict()

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
                  remove_stopwords, use_topic, use_description, select_pos_func,
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
        for qid, cwid, label, ngram_mat, query_idf, context_vec in zip(*q_process_send.get()):
            # Check for exceptions
            if cwid is None:
                raise Exception(qid)  # Exception comes in the qid variable

            labels.append(label)
            qid_label_cwids[qid][label].append(cwid)
            qid_cwid_label[qid][cwid] = label

            # Save matrices
            if sim_matrix_config:
                for ng in ngram_mat:
                    qid_ngram_cwid_mat[qid][ng][cwid] = ngram_mat[ng]
                if sim_matrix_config['use_context']:
                    qid_cwid_context[qid][cwid] = context_vec

            # Save Query IDF
            if query_idf_config and qid not in query_idfs:
                query_idfs[qid] = query_idf

    # Close pool
    pool.close()
    pool.join()

    # FIXME: Needs to be more general
    if 'train' in dset_folder.split('/')[-1]:
        """ Logic to construct softmax with 1 positive doc and num_negative negative docs for every training point"""
    
        label_count = Counter(labels)  # label counter
    
        # Get percentage of positive (1 and 2) labels
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
        neg_batch = defaultdict(lambda: defaultdict(list))  # num_negative examples per softmax
        pos_context_batch = []
        neg_context_batch = defaultdict(list)
        qidf_batch = list()
        ys = list()  # holds labels
    
        # Choose N samples of 1/2 (N is the total number of training samples in here)
        selected_labels = np.random.choice([l for l in sorted(sample_label_prob)], size=len(labels), replace=True,
                                           p=[sample_label_prob[l] for l in sorted(sample_label_prob)])
        label_counter = Counter(selected_labels)
        for label in label_counter:
            # select random qids equal to the number of the selected label
            # (this is necessary to train each of the selected labels)
            selected_qids = np.random.choice(list(qid_label_cwids.keys()),
                                             size=label_counter[label], replace=True, p=sample_label_qid_prob[label])
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
                neg_cwids = qid_label_cwids[qid][neg_labels[0]]
                n_pos, n_neg = len(pos_cwids), len(neg_cwids)
    
                # Add randomly picked positive matrices of that qid to positive batch
                idx_poses = np.random.choice(list(range(n_pos)), size=qid_counter[qid], replace=True)
                if sim_matrix_config:
                    min_ngram = min(qid_ngram_cwid_mat[qid])
                    for ngram in qid_ngram_cwid_mat[qid]:
                        for pi in idx_poses:
                            p_cwid = pos_cwids[pi]
                            pos_batch[ngram].append(qid_ngram_cwid_mat[qid][ngram][p_cwid])
                            if ngram == min_ngram:
                                if sim_matrix_config['use_context']:
                                    pos_context_batch.append(qid_cwid_context[qid][p_cwid])
                                ys.append(1)  # positive label
    
                    # Add num_negative randomly picked negative matrices
                    # per positive example of that qid to negative batch
                    for neg_ind in range(num_negative):
                        idx_negs = np.random.choice(list(range(n_neg)), size=qid_counter[qid], replace=True)
                        min_ngram = min(qid_ngram_cwid_mat[qid])
                        for ngram in qid_ngram_cwid_mat[qid]:
                            for ni in idx_negs:
                                n_cwid = neg_cwids[ni]
                                neg_batch[ngram][neg_ind].append(qid_ngram_cwid_mat[qid][ngram][n_cwid])
                                if ngram == min_ngram and sim_matrix_config['use_context']:
                                    neg_context_batch[neg_ind].append(qid_cwid_context[qid][n_cwid])

                # Add Query IDF to its batch
                if query_idf_config:
                    qidf_batch.append(
                        query_idfs[qid].reshape((1, query_idf_config['max_query_len'], 1)).repeat(qid_counter[qid], axis=0)
                    )
    
        # INPUT
        if sim_matrix_config:
            for ngram in pos_batch:
                dset['input']['pos_ngram_%d' % ngram] = np.array(pos_batch[ngram])
                for neg_ind in range(num_negative):
                    dset['input']['neg%d_ngram_%d' % (neg_ind, ngram)] = \
                        np.array(np.array(neg_batch[ngram][neg_ind]))
    
            if sim_matrix_config['use_context']:
                dset['input']['pos_context'] = np.array(pos_context_batch)
                for neg_ind in range(num_negative):
                    dset['input']['neg%d_context' % neg_ind] = np.array(neg_context_batch[neg_ind])

        if query_idf_config:
            dset['input']['query_idf'] = np.concatenate(qidf_batch, axis=0)

        # OUTPUT
        dset['output']['tags'] = np.array(ys)
    
    else:
        # Dev/Test
        doc_vec, q_idfs, contexts = defaultdict(list), list(), list()
        qids, cwids = list(), list()
        ys = list()
        for qid in qid_label_cwids:
            if sim_matrix_config:
                min_ngram = min(qid_ngram_cwid_mat[qid])
                for ngram in qid_ngram_cwid_mat[qid]:
                    for cwid in qid_ngram_cwid_mat[qid][ngram]:
                        doc_vec[ngram].append(qid_ngram_cwid_mat[qid][ngram][cwid])
                        if ngram == min_ngram:
                            # Add query IDF in last iteration
                            if query_idf_config:
                                q_idfs.append(query_idfs[qid].reshape((1, query_idfs[qid].shape[0], 1)))

                            if sim_matrix_config['use_context']:
                                contexts.append(qid_cwid_context[qid][cwid])

                            # Add label in last iteration
                            ys.append(qid_cwid_label[qid][cwid])

                            # Save qid and cwid
                            qids.append(qid)
                            cwids.append(cwid)

        # INPUT
        if sim_matrix_config:
            dset['input'] = {'doc_ngram_%d' % ngram: np.array(np.array(doc_vec[ngram])) for ngram in doc_vec}
            dset['input']['meta-data'] = {
                'qids': qids,
                'cwids': cwids
            }
            if sim_matrix_config['use_context']:
                dset['input']['doc_context'] = np.array(contexts)

        if query_idf_config:
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

        assert "topics_files" in config,\
            "Need to provide 'topics_files' in config"

        assert False not in [os.path.isfile(file) for file in config['topics_files']], \
            "Not all files exist in %s" % config['topics_files']

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

            if 'corpus_folder' not in config:
                config['corpus_folder'] = None

            if 'embeddings_path' not in config:
                config['embeddings_path'] = None

            if 'include_spam' not in config:
                config['include_spam'] = True

            self.datasets[dset] = read_corpus(
                dset_files,
                config['topics_files'],
                config['corpus_folder'],
                use_topic=config['use_topic'],
                use_description=config['use_description'],
                embeddings_path=config['embeddings_path'],
                sim_matrix_config=config['sim_matrix_config'],
                query_idf_config=config['query_idf_config'],
                num_negative=config['num_negative'],
                remove_stopwords=config['remove_stopwords'],
                dset_folder="%s/%s" % (config['name'], dset),
                include_spam=config['include_spam'],
                shuffle_seed=config['shuffle_seed']
            )
            self.nr_samples[dset] = \
                len(self.datasets[dset]['output']['tags'])

    def size(self, set_name):
        return self.nr_samples[set_name]

    def batches(self, set_name=None, batch_size=1, features_model=None,
                no_output=False, shuffle_seed=None):
        """
        Returns iterator over batches of features as

        {'input': { 'feat1': [[]], ... } 'output': { 'feat-a': [[]], ... }}

        If a model instance is provided, the features are extracted instead
        """

        dset = self.datasets[set_name]
        nr_examples = self.nr_samples[set_name]

        if batch_size is None:
            # Get all data
            nr_batch = 1
            batch_size = nr_examples
        else:
            nr_batch = int(np.ceil(nr_examples*1./batch_size))

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
                                batch_n*batch_size:(batch_n+1)*batch_size
                            ]
                    else:
                        data_batch[side][feat_name] = values[
                            batch_n*batch_size:(batch_n+1)*batch_size
                        ]

            # If feature extractors provided, return features instead
            if features_model is not None:
                feat_batch = features_model(**data_batch)
            else:
                feat_batch = data_batch

            data.append(feat_batch)

        return DataIterator(data, nr_samples=self.nr_samples[set_name])
