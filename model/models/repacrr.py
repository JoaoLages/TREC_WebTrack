import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Permute, Activation, Dense, Flatten, Input, \
    Lambda, Reshape, Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras import backend
import os
import string
import random


def get_name(nx, ny):
    return '%dx%d' % (nx, ny)


def _multi_kmax_concat(x, top_k, poses):
    # Get p first entries of each line and then top k values
    slice_mats = [tf.nn.top_k(tf.slice(x, [0, 0, 0], [-1, -1, p]), k=top_k, sorted=True, name=None)[0]
                  for p in poses]
    # Join the different slices
    concat_topk_max = tf.concat(slice_mats, -1, name='concat')
    return concat_topk_max


def _multi_kmax_context_concat(inputs, top_k, poses):
    x, context_input = inputs
    idxes, topk_vs = list(), list()
    for p in poses:
        val, idx = tf.nn.top_k(tf.slice(x, [0, 0, 0], [-1, -1, p]), k=top_k, sorted=True, name=None)
        topk_vs.append(val)
        idxes.append(idx)
    concat_topk_max = tf.concat(topk_vs, -1, name='concat_val')
    concat_topk_idx = tf.concat(idxes, -1, name='concat_idx')
    # hack that requires the context to have the same shape as similarity matrices
    # https://stackoverflow.com/questions/41897212/how-to-sort-a-multi-dimensional-tensor-using-the-returned-indices-of-tf-nn-top-k
    shape = tf.shape(x)
    mg = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape[:(x.get_shape().ndims - 1)]) + [top_k * len(poses)])],
                     indexing='ij')
    val_contexts = tf.gather_nd(context_input, tf.stack(mg[:-1] + [concat_topk_idx], axis=-1))

    return backend.concatenate([concat_topk_max, val_contexts])


def add_2dfilters(ngram_filter, filters2add):
    for nx2a, ny2a in filters2add:
        # Init list if it doesnt exist
        if ny2a not in ngram_filter:
            ngram_filter[ny2a] = list()

        # Get existing filternames
        filternames = [get_name(nx, ny) for nx, ny in ngram_filter[ny2a]]

        # Append new filter
        if get_name(nx2a, ny2a) not in filternames:
            ngram_filter[ny2a].append((nx2a, ny2a))


def add_ngram_nfilter(ngram_filter, ngrams):
    filters2add = list()
    for ngram in ngrams:
        filters2add.append((ngram, ngram))
    add_2dfilters(ngram_filter, filters2add)


def add_proximity_filters(ngram_filter, proximity=0, len_query=16):
    if proximity > 0:
        add_2dfilters(ngram_filter, [(len_query, proximity)])


def parse_more_filter(ngram_filter, more_filters):
    '''
    convert the input string to a list of 2d filter sizes in form of tuple.
    input format: axb.cxd. ...
    for example: 1x100,3x1 => [(1,100), (3,1)]
    '''
    filters2add = list()
    tups = more_filters.split('.')
    for tup in tups:
        if tup != '':
            a_b = tup.split('x')
            if len(a_b) != 2:
                raise ValueError("malformed of the input filter sizes %s" % more_filters)
            filters2add.append((int(a_b[0]), int(a_b[1])))

    # Add filters
    add_2dfilters(ngram_filter=ngram_filter, filters2add=filters2add)


def get_ngram_nfilter(ngrams, proximity, len_query, more_filter_str):
    # Build dictionary
    ngram_filter = dict()

    # Main N-Gram filter
    add_ngram_nfilter(ngram_filter, ngrams)

    # Proximity filter
    add_proximity_filters(ngram_filter, proximity=proximity, len_query=len_query)

    # Extra filters
    parse_more_filter(ngram_filter, more_filter_str)

    return ngram_filter


def pos_softmax(pos_neg_scores):
    exp_pos_neg_scores = [tf.exp(s) for s in pos_neg_scores]
    denominator = tf.add_n(exp_pos_neg_scores)
    return exp_pos_neg_scores[0] / denominator


class REPACRR:
    """
    RE-PACRR model
    """
    def __init__(self, config):
        # Initialise hyperparameters
        self.p = {
            'seed': config['seed'],
            'batch_size': config['batch_size'],
            'plot_model': config['plot_model'],

            'max_doc_len': config['sim_matrix_config']['max_doc_len'],  # length of document dimension
            'max_query_len': config['sim_matrix_config']['max_query_len'],  # maximum query length
            'ngrams': config['sim_matrix_config']['ngrams'],  # n-grams to use for windows in convolutions
            'pos_method': config['sim_matrix_config']['pos_method'],  # similarity matrix distillation method
            'use_context': config['sim_matrix_config']['use_context'],  # include match contexts? (boolean)

            'num_negative': config['num_negative'],  # number of non-relevant docs in softmax
            'shuffle': config['shuffle'],  # 'permute',
            # shuffle input to the combination layer? (i.e., LSTM or feedforward layer)
            'filter_size': config['filter_size'],  # number of filters to use for the n-gram convolutions
            'top_k': config['top_k'],  # Get top_k after maxpooling to add to scores
            'combine': config['combine'],  # type of combination layer to use. 0 for an LSTM,
            # otherwise the number of feedforward layer dimensions

            # TODO:??
            'qproximity': 0,  # additional NxN proximity filter to include (0 to disable)
            'ek': 10,  # topk expansion terms to use when enhance=qexpand or enhance=both

            # configure the sizes of extra filters with format: axb.cxd.<more>
            # for example: 1x100.3x1:> [(1,100), (3,1)]
            # to turn off, set to an empty string
            'xfilters': "",

            # configure the cascade mode of the max-pooling
            # Namely, pool over [first10, first20, ..., first100, wholedoc]
            # instead of only pool on complete document
            # the input is a list of relative positions for pooling
            # for example, 25.50.75.100:> [25,50,75,100] on a doc with length 100
            # to turn off, set to an empty string
            'cascade': "",

            # To save for data config when loading model
            'sim_matrix_config': config['sim_matrix_config'],
            'query_idf_config': config['query_idf_config']
        }

        # Model folder
        self.output_dir = config['model_folder']

        # {1: [(1, 1)], 2: [(2, 2)], 3: [(3, 3)]}
        self.ngram_filter = get_ngram_nfilter(
            self.p['ngrams'], self.p['qproximity'], self.p['max_query_len'], self.p['xfilters']
        )

        # Metric
        self.metric = config['metric']

        # Build train/predict model later
        self.train_model = None
        self.predict_model = None
        self.callback = None

    def forward(self, r_query_idf, permute_idxs):
        """
        Build forward function
        :param r_query_idf: Input((max_query_len, 1), float)
        :param permute_idxs: Input((max_query_len, 2), int)
        :return: forward function
        """

        # maxpool positions ([800])
        maxpool_poses = self._cascade_poses()

        # Get filter sizes
        filter_sizes = [
            (n_x, n_y)
            for ng in sorted(self.ngram_filter)
            for n_x, n_y in self.ngram_filter[ng]
        ]

        # Get 2D Conv Layers, k_top values, k_top_context values, MaxPool Layers,
        # Query permutation layers and axis=2 removal layers
        conv_layers = dict()
        maxpool_layer = dict()
        pool_top_k_layer = dict()
        pool_top_k_layer_context = dict()
        squeeze = dict()
        for n_query, n_doc in filter_sizes:
            subsample_docdim = 1
            if self.p['pos_method'] in ['strides']:
                subsample_docdim = n_doc
            dim_name = get_name(n_query, n_doc)

            # 2D Convolution Layer
            conv_layers[dim_name] = Conv2D(
                self.p['filter_size'], kernel_size=(n_query, n_doc), strides=(1, subsample_docdim), padding="same",
                name='conv_%s' % dim_name, activation='relu', weights=None
            )

            # MaxPooling Layer
            maxpool_layer[dim_name] = \
                MaxPooling2D(pool_size=(1, self.p['filter_size']), name='maxpool_%s' % dim_name)

            # K-MaxPooling Layer
            pool_top_k_layer[dim_name] = Lambda(lambda x: _multi_kmax_concat(x, self.p['top_k'], maxpool_poses),
                                                name='pool_%s_top%d' % (dim_name, self.p['top_k']))

            # K-MaxPooling Layer with context
            pool_top_k_layer_context[dim_name] = \
                Lambda(lambda x: _multi_kmax_context_concat(x, self.p['top_k'], maxpool_poses),
                       name='pool_%s_top%d_context' % (dim_name, self.p['top_k']))

            # Remove axis=2
            squeeze[dim_name] = \
                Lambda(lambda t: backend.squeeze(t, axis=2), name='squeeze_%s' % (dim_name))

        # Reshape Layer
        re_input = Reshape((self.p['max_query_len'], self.p['max_doc_len'], 1), name='ql_ds_doc')

        # Query normalized IDF score
        query_idf_score = Reshape((self.p['max_query_len'], 1))(
            Activation('softmax', name='softmax_q_idf')(
                Flatten()(r_query_idf)
            )
        )

        if self.p['combine'] < 0:
            raise RuntimeError("combine should be 0 (LSTM) or the number of feedforward dimensions")
        elif self.p['combine'] == 0:
            rnn_layer = LSTM(1, dropout=0.0, recurrent_regularizer=None, recurrent_dropout=0.0, unit_forget_bias=True,
                             name="lstm_merge_score_idf", recurrent_activation="hard_sigmoid", bias_regularizer=None,
                             activation="tanh", recurrent_initializer="orthogonal", kernel_regularizer=None,
                             kernel_initializer="glorot_uniform")

        else:
            # FF Dense Layers
            dout = Dense(1, name='dense_output')
            d1 = Dense(self.p['combine'], activation='relu', name='dense_1')
            d2 = Dense(self.p['combine'], activation='relu', name='dense_2')
            rnn_layer = lambda x: dout(d1(d2(Flatten()(x))))

        def _permute_scores(inputs):
            scores, idxs = inputs
            return tf.gather_nd(scores, backend.cast(idxs, 'int32'))

        # Aux variables
        ngram_filter = self.ngram_filter
        pos_method = self.p['pos_method']
        use_context = self.p['use_context']

        def _scorer(doc_inputs):
            doc_qts_scores = [query_idf_score]
            for ng in sorted(ngram_filter):
                if pos_method == 'firstk':
                    input_ng = max(ngram_filter)
                else:
                    input_ng = ng

                for n_x, n_y in ngram_filter[ng]:
                    dim_name = get_name(n_x, n_y)
                    if n_x == 1 and n_y == 1:
                        # No Convolution for 1-gram
                        re_doc_cov = doc_inputs[input_ng]
                    else:
                        # Pass input by Convolution Layer
                        doc_cov = conv_layers[dim_name](re_input(doc_inputs[input_ng]))
                        # Pass Conv output through MaxPooling Layer (after permuting) and remove axis=2
                        re_doc_cov = squeeze[dim_name](maxpool_layer[dim_name](Permute((1, 3, 2))(doc_cov)))

                    # Get top_k max values for each row in the matrix (K-MaxPooling Layer)
                    if use_context:
                        ng_signal = pool_top_k_layer_context[dim_name]([re_doc_cov, doc_inputs['context']])
                    else:
                        ng_signal = pool_top_k_layer[dim_name](re_doc_cov)

                    doc_qts_scores.append(ng_signal)

            # Concatenate scores for each query term
            if len(doc_qts_scores) == 1:
                doc_qts_score = doc_qts_scores[0]
            else:
                doc_qts_score = Concatenate(axis=2)(doc_qts_scores)

            # Permute query positions
            if permute_idxs is not None:
                doc_qts_score = Lambda(_permute_scores)([doc_qts_score, permute_idxs])

            # Get a final score
            doc_score = rnn_layer(doc_qts_score)

            return doc_score

        return _scorer

    def write_log(self, logs, num_log, names=['loss', 'accuracy']):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, num_log)
            self.callback.writer.flush()

    def _cascade_poses(self):
        '''
        initialize the cascade positions, over which
        we max-pool after the cnn filters.
        the outcome is a list of document positions.
        when the list only includes the SIM_DIM, it
        is equivalent to max-pool over the whole document
        '''
        doc_poses = list()
        pos_arg = str(self.p['cascade'])
        if len(pos_arg) > 0:
            poses = pos_arg.split('.')
            for p in poses:
                if len(p) > 0:
                    p = int(p)
                    if p <= 0 or p > 100:
                        raise ValueError("Cascade positions are outside (0,100]: %s" % pos_arg)
            doc_poses.extend([int((int(p) / 100) * self.p['max_doc_len']) for p in poses if len(p) > 0])

        if self.p['max_doc_len'] not in doc_poses:
            doc_poses.append(self.p['max_doc_len'])

        return doc_poses

    def _create_inputs(self, prefix):
        if self.p['pos_method'] == 'firstk':
            ng = max(self.p['ngrams'])
            inputs = {ng: Input(shape=(self.p['max_query_len'], self.p['max_doc_len']),
                                name='%s_ngram_%d' % (prefix, ng))}
        else:
            inputs = {}
            for ng in self.p['ngrams']:
                inputs[ng] = Input(shape=(self.p['max_query_len'], self.p['max_doc_len']),
                                   name='%s_ngram_%d' % (prefix, ng))

        return inputs

    def build_train(self):
        # Build train model
        r_query_idf = Input(shape=(self.p['max_query_len'], 1), name='query_idf')
        if self.p['shuffle']:
            permute_input = Input(shape=(self.p['max_query_len'], 2), name='permute', dtype='int32')
        else:
            permute_input = None

        # Get forward scorer
        doc_scorer = self.forward(r_query_idf, permute_input)

        # {3: Input(shape=(16,800))}
        pos_inputs = self._create_inputs('pos')
        if self.p['use_context']:
            pos_inputs['context'] = Input(shape=(self.p['max_query_len'], self.p['max_doc_len']), name='pos_context')

        neg_inputs = {}
        # {0: {3: Input(shape=(16,800))}, ...} 0-> indice of non relevant doc
        # nunmeg -> non relevant docs in softmax
        for neg_ind in range(self.p['num_negative']):
            neg_inputs[neg_ind] = self._create_inputs('neg%d' % neg_ind)
            if self.p['use_context']:
                neg_inputs[neg_ind]['context'] = Input(shape=(self.p['max_query_len'], self.p['max_doc_len']),
                                                       name='neg%d_context' % neg_ind)

        # Get score of positive doc and negative doc(s)
        pos_score = doc_scorer(pos_inputs)
        neg_scores = [doc_scorer(neg_inputs[neg_ind]) for neg_ind in range(self.p['num_negative'])]

        # Join pos/neg(s) scores, do softmax and take prob of pos doc
        pos_neg_scores = [pos_score] + neg_scores
        pos_prob = Lambda(pos_softmax, name='pos_softmax_loss')(pos_neg_scores)

        # Construct training inputs
        pos_input_list = [pos_inputs[name] for name in pos_inputs]
        neg_input_list = [neg_inputs[neg_ind][ng] for neg_ind in neg_inputs for ng in neg_inputs[neg_ind]]
        inputs = pos_input_list + neg_input_list + [r_query_idf]
        if self.p['shuffle']:
            inputs.append(permute_input)

        # Compile model
        self.train_model = Model(inputs=inputs, outputs=[pos_prob])
        self.train_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Initialize TensorBoard callback
        self.callback = TensorBoard("%s/logs" % self.output_dir)
        self.callback.set_model(self.train_model)
        self.batch_nr = 0
        self.val_nr = 0

        # Initialize loss history
        self.loss_history = []

        # Plot model
        if self.p['plot_model']:
            plot_model(self.train_model, to_file="%s/train_model_plot.pdf" % self.output_dir, show_shapes=True)

    def build_predict(self):
        # Build predict model
        r_query_idf = Input(shape=(self.p['max_query_len'], 1), name='query_idf')
        doc_inputs = self._create_inputs('doc')
        if self.p['use_context']:
            doc_inputs['context'] = Input(shape=(self.p['max_query_len'], self.p['max_doc_len']), name='doc_context')

        # Get forward scorer
        doc_scorer = self.forward(r_query_idf, permute_idxs=None)

        # Get doc score
        doc_score = doc_scorer(doc_inputs)
        doc_input_list = [doc_inputs[name] for name in doc_inputs]

        self.predict_model = Model(inputs=doc_input_list + [r_query_idf], outputs=[doc_score])

    def get_features(self, input, output):
        if self.p['shuffle']:
            # Add permute
            input['permute'] = np.array(
                [[(bi, qi)
                  for qi in np.random.permutation(input['query_idf'].shape[1])]
                 for bi in range(input['query_idf'].shape[0])],
                dtype=np.int
            )
        return {'input': input, 'output': output}

    def update(self, input, output):
        if not self.train_model:
            # Build train model
            self.build_train()

        # Remove meta-data
        input = {key: input[key] for key in input if key != 'meta-data'}

        # Forward + Backprop
        logs = self.train_model.train_on_batch(input, output['tags'])

        # Write logs to TensorBoard
        self.write_log(logs, self.batch_nr)
        self.batch_nr += 1

        # FIXME: to remove - Save loss history
        self.loss_history.append(logs[0])

        return self.loss_history[-1]

    def predict(self, input, output):
        # Build predict model
        if not self.predict_model:
            self.build_predict()

        # Check that train model exists
        if self.train_model:
            # Save train weights to temp file and load pred weights from it
            random_fn = ''.join(random.choice(string.ascii_lowercase) for _ in range(7))

            # Get weights from train model
            self.train_model.save_weights('%s.h5' % random_fn)
            self.predict_model.load_weights('%s.h5' % random_fn, by_name=True)
            os.remove("%s.h5" % random_fn)

        # Remove meta-data
        input = {key: input[key] for key in input if key != 'meta-data'}

        # Get probabilities
        probs = self.predict_model.predict_on_batch(input)
        return probs

    def load(self, path):
        # Build model to predict
        if not self.predict_model:
            self.build_predict()

        # Load weights into new model
        self.predict_model.load_weights("%s" % path, by_name=True)

    def save(self):
        path = '%s/%s' % (self.output_dir, type(self).__name__.lower())
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            os.system('mkdir -p {}'.format(parent))
        # Save
        self.train_model.save_weights("%s.h5" % path)
