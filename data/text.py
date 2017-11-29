# -*- coding: utf-8 -*-
import re
from collections import Counter, defaultdict
import itertools
import pickle as cPickle
import numpy as np


SPECIAL_TOKENS = {
    'unknown-token': '__unknown__',
    'sentence-start': '__start__',
    'sentence-end': '__end__',
    'padding': '__pad__',
    'null-token': '__null__'
}

POLYGLOT_MAP = {
    '__unknown__': u'<UNK>',
    '__start__': u'<S>',
    '__end__': u'</S>',
    '__pad__': '<PAD>'
}


def get_indexer(vocabulary, special_tokens):
    """
    Returns a function that transforms text tokens into indices to a vocabulary
    """
    word2index = dict(zip(vocabulary, range(len(vocabulary))))

    def get_index(word):
        if word in word2index:
            return word2index[word]
        elif 'unknown-token' not in special_tokens:
            raise Exception("Unknown token %s" % word)
        else:
            return word2index[special_tokens['unknown-token']]
    return get_index


def flatten_list(input_list):
    """ Flattens list of lists into a single list """
    if all(isinstance(el, list) for el in input_list):
        # Return flatten list
        return list(itertools.chain.from_iterable(input_list))
    else:
        return input_list


def get_voc(texts):
    SPECIAL_WORDS = list(SPECIAL_TOKENS.values())

    word_count = Counter([
        token
        for tokens in texts
        for token in flatten_list(tokens)
    ])
    vocabulary = SPECIAL_WORDS + sorted(word_count.keys())
    return set(vocabulary), word_count


def vocabulary_from_polyglot(vocabulary, embeddings):

    # Polyglot Embeddings
    emb_words, emb_vectors = cPickle.load(open(embeddings, 'rb'), encoding='latin1')
    # Construct vocabulary
    embeddings = []
    emb_size = emb_vectors.shape[1]
    for word in vocabulary:
        word = POLYGLOT_MAP[word] if word in POLYGLOT_MAP else word
        if word in emb_words:
            embeddings.append(emb_vectors[emb_words.index(word), :][None, :])
        else:
            embeddings.append(np.zeros((1, emb_size)))
    return vocabulary, np.concatenate(embeddings), SPECIAL_TOKENS


def construct_vocabulary(data, selector_query, selector_subtopic,
                         selector_article, embeddings):
    """
    Constructs vocabulary for source and target
    """

    # Embeddings
    vocabulary = set(SPECIAL_TOKENS.values())

    for sset in ['train', 'dev', 'test']:
        data_batches = data.batches(sset)

        texts = []
        for data_batch in data_batches:
            texts.extend(selector_query(data_batch))
            texts.extend(selector_subtopic(data_batch))
            texts.extend(selector_article(data_batch))

        # Add to vocabulary sets
        vocabulary |= get_voc(texts)[0]

    vocabulary = sorted(vocabulary)
    return vocabulary_from_polyglot(vocabulary, embeddings)


def initialization(data, emb_path):

    # INPUT
    vocabulary, emb, special_tokens = construct_vocabulary(
        data,
        lambda x: x['input']['query'],
        lambda x: x['input']['subtopic'],
        lambda x: x['input']['article'],
        emb_path
    )

    encoding = {
        'vocabulary': vocabulary,
        'emb': emb,
        'special_tokens': special_tokens
    }

    # OUTPUT
    unique_tags = set()
    for sset in ['train', 'dev', 'test']:
        data_batches = data.batches(sset)
        tags = [tag
                for x in data_batches
                for tag in x['output']['tags']]
        # Add to unique_tags set
        unique_tags |= set(tags)

    unique_tags_binary = {0, 1}

    return encoding, unique_tags, unique_tags_binary
