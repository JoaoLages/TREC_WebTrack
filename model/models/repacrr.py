import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Permute, Activation, Dense, Flatten, Input, \
    Lambda, Reshape, Conv2D, MaxPooling2D, Dropout, Embedding, Conv1D, Multiply
from keras.initializers import Constant
from keras.layers.merge import Concatenate, Dot
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model, multi_gpu_model
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras import backend as K, regularizers
from model.models.keras_toolkit import AttentionMechanism, Capsule
import os
import string
import random
import pickle


def get_name(nx, ny):
    return '%dx%d' % (nx, ny)


def _multi_kmax_concat(x, top_k, poses):
    # Get p first entries of each line and then top k values
    if len(x.shape) == 3:
        slice_mats = [tf.nn.top_k(tf.slice(x, [0, 0, 0], [-1, -1, p]), k=top_k, sorted=True, name=None)[0]
                      for p in poses]
    elif len(x.shape) == 4:
        # (?, channel, n, m)
        slice_mats = [tf.nn.top_k(tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, p]), k=top_k, sorted=True, name=None)[0]
                      for p in poses]
    else:
        raise Exception("Invalid shape %s" % x.shape)

    # Join the different slices
    return tf.concat(slice_mats, -1, name='concat')


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

    return K.concatenate([concat_topk_max, val_contexts])


def softmax(pos_neg_scores):
    exp_pos_neg_scores = [tf.exp(s) for s in pos_neg_scores]
    denominator = tf.add_n(exp_pos_neg_scores)
    return tf.concat(exp_pos_neg_scores, axis=1) / denominator


def pos_softmax(pos_neg_scores):
    exp_pos_neg_scores = [tf.exp(s) for s in pos_neg_scores]
    denominator = tf.add_n(exp_pos_neg_scores)
    return exp_pos_neg_scores[0] / denominator


# convenience l2_norm function
def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    return K.sqrt(K.maximum(square_sum, K.epsilon()))


def pairwise_cosine_sim(A, B, normalize=True):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=2)
    num = K.batch_dot(A, K.permute_dimensions(B, (0, 2, 1)))
    den = (A_mag * K.permute_dimensions(B_mag, (0, 2, 1)))
    dist_mat = num / den if normalize else num

    return dist_mat


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
            'use_context': config['sim_matrix_config']['use_context'],  # include match contexts? (boolean)
            'use_maxpooling': config['sim_matrix_config']['use_maxpooling'],
            # use maxpooling or 1x1 conv to reduce dimension
            'custom_loss': config['custom_loss'],  # use binary crossentropy or categorical
            'use_static_matrices': config['sim_matrix_config']['use_static_matrices'],
            # Use static matrices or create them with embedding layer always
            'use_masking': config['sim_matrix_config']['use_masking'],  # multiply output of 2D convs by mask

            'use_convs': config['use_convs'],  # whether to use convs or not
            'permute': config['permute'],  # permute the input to the classification head
            'num_negative': config['num_negative'],  # number of non-relevant docs in softmax
            'filter_size': config['filter_size'],  # number of filters to use for the n-gram convolutions
            'top_k': config['top_k'],  # Get top_k after maxpooling to add to scores
            'combine': config['combine'],  # type of combination layer to use. 0 for an LSTM,
            # otherwise the number of feedforward layer dimensions

            # Regularizers
            'l2_lambda': config['l2_lambda'],
            'drop_rate': config['drop_rate'],
            'batch_norm': config['batch_norm'],

            # configure the cascade mode of the max-pooling
            # Namely, pool over [first10, first20, ..., first100, wholedoc]
            # instead of only pool on complete document
            # the input is a list of relative positions for pooling
            # for example, 25.50.75.100:> [25,50,75,100] on a doc with length 100
            'cascade': config['cascade'],

            # To save for data config when loading model
            'sim_matrix_config': config['sim_matrix_config'],
            'query_idf_config': config['query_idf_config'],
            'use_topic': config['use_topic'],
            'use_description': config['use_description']
        }

        # Model folder
        self.output_dir = config['model_folder']

        # Sanity checks
        if not self.p['use_static_matrices']:

            """
            Mode 0 - build sim matrix from cosine sim of embedding layers
            Mode 1 - build sim matrix by passing embeddings through FF and adding them in a 3D matrix
            Mode 2 - build sim matrix by passing embeddings through Conv1D and compute 2D similarity matrix
            Mode 3 - build sim matrix without normalization
            """
            assert config['sim_matrix_config']['matrix_build_mode'] in [0, 1, 2, 3]
            self.p['matrix_build_mode'] = config['sim_matrix_config']['matrix_build_mode']

            # Load embeddings matrix
            assert 'embeddings_path' in config, "Need 'embeddings_path' to be provided to load embedding matrix"
            assert 'embeddings_matrix.npy' in os.listdir(
                os.path.dirname(config['embeddings_path'])), "Construct embeddings matrix first"
            assert 'tokenizer_vocabsize.pickle' in os.listdir(
                os.path.dirname(config['embeddings_path'])), "Construct embeddings matrix first"

            # Get vocab_size & embeddings matrix
            with open('%s/tokenizer_vocabsize.pickle' % os.path.dirname(config['embeddings_path']), 'rb') as handle:
                _, self.vocab_size = pickle.load(handle)
            self.embedding_matrix = np.load('%s/embeddings_matrix.npy' % os.path.dirname(config['embeddings_path']))

        # Metric
        self.metric = config['metric']

        # Build train/predict model later
        self.train_model = None
        self.predict_model = None
        self.intermediate_layers = None
        self.train_mode = False
        self.callback = None

        # For multi GPU
        if 'gpu_device' in config:
            self.num_gpus = len(config['gpu_device'])
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))
        else:
            self.num_gpus = 0

    def forward(self, r_query_idf, permute_idxs):
        """
        Build forward function
        :param r_query_idf: Input((max_query_len, 1), float)
        :param permute_idxs: Input((max_query_len, 2), int)
        :return: forward function
        """
        extra_matrix = None
        if not self.p['use_static_matrices']:
            # Embedding Layer
            embedding_layer = Embedding(
                self.vocab_size, 300, weights=[self.embedding_matrix],
                input_length=self.p['max_doc_len'], trainable=False, name='embedding_layer'
            )

            # Linear projection layer of the embeddings (reducing their size)
            ff_size = 20
            ff = Dense(ff_size)
            drop_matrix = Dropout(0.2)

            # 1D Convolution Layer
            conv1D_layer = Conv1D(
                self.p['filter_size'], kernel_size=3, padding="same",
                name='conv1D', activation='relu', kernel_initializer='he_uniform',
                bias_initializer=Constant(0.01), kernel_regularizer=regularizers.l2(self.p['l2_lambda'])
            )

            def embeddings2channels(x):
                # Converts embedding dimension into smaller one by applying feed forward
                return Concatenate(axis=1)(
                    [drop_matrix(ff(row)) for row in tf.split(x, num_or_size_splits=x.shape[1], axis=1)]
                )

            # Build similarity matrix

            if self.p['matrix_build_mode'] == 0:
                # Build cosine sim matrix after passing through LP layer
                build_matrix = Lambda(lambda x: pairwise_cosine_sim(embeddings2channels(x[0]), embeddings2channels(x[1])), name='build_sim_matrix')

            elif self.p['matrix_build_mode'] == 1:
                def join_channels(x, y):
                    """
                    Receives 2 matrices with 1 common dimension (the last axis)
                    and computes the 3d combination of both by summing
                    :param x: (?, m, k)
                    :param y: (?, n, k)
                    :return: (?, m, n, k)
                    """
                    return tf.stack([tf.add(row, y) for row in tf.split(x, num_or_size_splits=x.shape[1], axis=1)], axis=1)

                build_matrix = Lambda(lambda x: join_channels(embeddings2channels(x[0]), embeddings2channels(x[1])), name='build_sim_matrix')

            elif self.p['matrix_build_mode'] == 2:
                # Build cosine sim matrix normally and after passing through Conv1D layer
                build_matrix = Lambda(lambda x: pairwise_cosine_sim(x[0], x[1]), name='build_sim_matrix')
                extra_matrix = Lambda(lambda x: pairwise_cosine_sim(drop_matrix(conv1D_layer(x[0])), drop_matrix(conv1D_layer(x[1]))), name='build_extra_matrix')
            
            elif self.p['matrix_build_mode'] == 3:
                build_matrix = Lambda(lambda x: pairwise_cosine_sim(x[0], x[1], normalize=False), name='build_sim_matrix')

        # NOTE: Deprecated versions for k_max = 0
        # if not self.p['top_k']:
        #     # Build Attention Mechanism
        #     attention_layer = AttentionMechanism((300, 8), self.p['max_doc_len'], 1)  # 64 is hyperparameter, 300 is embedding dim, 1 is fixed

        # maxpool positions ([800])
        maxpool_poses = self._cascade_poses()

        # Get 2D Conv Layers, k_top values, k_top_context values, MaxPool Layers,
        # Query permutation layers, axis=2 removal layers and regularizers
        conv_layers = dict()
        extra_conv_layers = dict()
        conv1x1_layers = dict()
        extra_conv1x1_layers = dict()
        batch_norms = dict()
        batch_norms1x1 = dict()
        maxpool_layer = dict()
        extra_maxpool_layer = dict()
        dropouts = dict()
        pool_top_k_layer = dict()
        pool_top_k_layer_context = dict()
        squeeze = dict()
        drop_row = Dropout(0.1)
        padding = 'valid' if self.p['top_k'] == 0 else 'same'
        ff_row = Dense(4, activation='relu', kernel_initializer='he_uniform',
                     bias_initializer=Constant(0.01), kernel_regularizer=regularizers.l2(self.p['l2_lambda']))
        for ngram in self.p['ngrams']:
            dim_name = get_name(ngram, ngram)

            if self.p['use_convs']:
                # 2D Convolution Layer
                conv_layers[dim_name] = Conv2D(
                    self.p['filter_size'], kernel_size=(ngram, ngram), strides=(1, 1), padding='same',
                    name='conv_%s' % dim_name, activation='relu', kernel_initializer='he_uniform',
                    bias_initializer=Constant(0.01), kernel_regularizer=regularizers.l2(self.p['l2_lambda'])
                )
                if self.p['top_k'] == 0:
                    extra_conv_layers[dim_name] = Conv2D(
                        self.p['filter_size'], kernel_size=(self.p['max_query_len'], self.p['max_query_len']),
                        strides=(1, 1), padding=padding,
                        name='extra_conv_%s' % dim_name, activation='relu', kernel_initializer='he_uniform',
                        bias_initializer=Constant(0.01), kernel_regularizer=regularizers.l2(self.p['l2_lambda'])
                    )

                # Batch Normalization after non linearity
                if self.p['batch_norm']:
                    batch_norms[dim_name] = BatchNormalization()

            if self.p['use_maxpooling']:
                if not self.p['use_static_matrices'] and dim_name == get_name(1, 1):
                    maxpool_filter = ff_size  # For build_matrix_mode 1
                else:
                    maxpool_filter = self.p['filter_size']
                # MaxPooling Layer
                maxpool_layer[dim_name] = \
                    MaxPooling2D(pool_size=(1, maxpool_filter), name='maxpool_%s' % dim_name)
                if self.p['top_k'] == 0:
                    extra_maxpool_layer[dim_name] = \
                        MaxPooling2D(pool_size=(1, self.p['filter_size']), name='extra_maxpool_%s' % dim_name)
            else:
                # 1x1 Convolution
                conv1x1_layers[dim_name] = Conv2D(
                    1, kernel_size=(1, 1), strides=(1, 1), padding=padding,
                    name='conv1x1_%s' % dim_name, activation='relu', kernel_initializer='he_uniform',
                    bias_initializer=Constant(0.01), kernel_regularizer=regularizers.l2(self.p['l2_lambda'])
                )
                # Batch Normalization after non linearity
                if self.p['batch_norm']:
                    batch_norms1x1[dim_name] = BatchNormalization()

                if self.p['top_k'] == 0:
                    extra_conv1x1_layers[dim_name] = Conv2D(
                        1, kernel_size=(1, 1), strides=(1, 1), padding=padding,
                        name='extra_conv1x1_%s' % dim_name, activation='relu', kernel_initializer='he_uniform',
                        bias_initializer=Constant(0.01), kernel_regularizer=regularizers.l2(self.p['l2_lambda'])
                    )

            # Remove axis=2
            squeeze[dim_name] = \
                Lambda(lambda t: K.squeeze(t, axis=2), name='squeeze_%s' % (dim_name))

            # Dropout Layer
            dropouts[dim_name] = Dropout(self.p['drop_rate'])

            if self.p['top_k']:
                # K-MaxPooling Layer
                pool_top_k_layer[dim_name] = Lambda(lambda x: _multi_kmax_concat(x, self.p['top_k'], maxpool_poses),
                                                    name='pool_%s_top%d' % (dim_name, self.p['top_k']))

                # K-MaxPooling Layer with context
                pool_top_k_layer_context[dim_name] = \
                    Lambda(lambda x: _multi_kmax_context_concat(x, self.p['top_k'], maxpool_poses),
                           name='pool_%s_top%d_context' % (dim_name, self.p['top_k']))

            # NOTE: Deprecated versions for k_max = 0
            # else:
            #     def hidden_layer(x, num_split):
            #         return Concatenate(axis=1)([drop_row(ff_row(row)) for row in tf.split(x, num_or_size_splits=num_split, axis=1)])
            #     pool_top_k_layer[dim_name] = Lambda(
            #         lambda x: hidden_layer(x, self.p['max_query_len']),
            #         name='hidden_layer_%s' % dim_name
            #     )

                # Use Attention Mechanism
                #def attention_head(x_emb, y_emb, matrix, num_split):
                #    return Concatenate(axis=2)(
                #            [drop_row(attention_layer([x_emb, y_emb_row, row]))
                #             for y_emb_row, row in zip(tf.split(y_emb, num_or_size_splits=num_split, axis=1),
                #                                      tf.split(matrix, num_or_size_splits=num_split, axis=1))])
                #pool_top_k_layer[dim_name] = Lambda(
                #    lambda x: attention_head(x[0], x[1], x[2], self.p['max_query_len']), name='attention_mechanism_%s' % dim_name
                #)

        # Reshape Layer to have 3 dimensions
        re_input = Reshape((self.p['max_query_len'], self.p['max_doc_len'], 1), name='ql_ds_doc')

        if self.p['query_idf_config']['use_query_idf']:
            # Query normalized IDF score
            query_idf_score = Reshape((self.p['max_query_len'], 1))(
                Activation('softmax', name='softmax_q_idf')(
                    Flatten()(r_query_idf)
                )
            )

        if self.p['combine'] < 0:
            raise RuntimeError("combine should be 0 (LSTM) or the number of feedforward dimensions")
        elif self.p['combine'] == 0:
            head_layer = LSTM(1, dropout=0.0, recurrent_regularizer=None, recurrent_dropout=0.0, unit_forget_bias=True,
                             name="lstm_merge_score_idf", recurrent_activation="hard_sigmoid", bias_regularizer=None,
                             activation="tanh", recurrent_initializer="orthogonal", kernel_regularizer=None,
                             kernel_initializer="glorot_uniform")

        else:
            # FF Dense Layers
            dout = Dense(1, name='dense_output')
            d1 = Dense(2 * self.p['combine'], activation='relu', kernel_initializer='he_uniform',
                       bias_initializer=Constant(0.01), kernel_regularizer=regularizers.l2(self.p['l2_lambda']), name='dense_1')
            drop1 = Dropout(self.p['drop_rate'])
            d2 = Dense(self.p['combine'], activation='relu', kernel_initializer='he_uniform',
                       bias_initializer=Constant(0.01), kernel_regularizer=regularizers.l2(self.p['l2_lambda']), name='dense_2')
            drop2 = Dropout(self.p['drop_rate'])
            if self.p['batch_norm']:
                # batch normalization after activations
                b1 = BatchNormalization()
                b2 = BatchNormalization()
                head_layer = lambda x: dout(drop2(b2(d2(drop1(b1(d1(Flatten()(x))))))))
            else:
                head_layer = lambda x: dout(drop2(d2(drop1(d1(Flatten()(x))))))

        def _permute_scores(inputs):
            scores, idxs = inputs
            return tf.gather_nd(scores, K.cast(idxs, 'int32'))

        # Aux variables
        ngrams = self.p['ngrams']
        use_context = self.p['use_context']
        top_k = self.p['top_k']
        batch_norm = self.p['batch_norm']
        use_maxpooling = self.p['use_maxpooling']
        use_query_idf_config = self.p['query_idf_config']['use_query_idf']
        use_static_matrices = self.p['use_static_matrices']
        use_convs = self.p['use_convs']
        use_masking = self.p['use_masking']

        def _scorer(doc_inputs):
            # Pass to another variable not to override doc_inputs in the future
            inputs = doc_inputs
            if not use_static_matrices:
                query, doc = embedding_layer(inputs['query']), embedding_layer(inputs['doc'])

                # Cut query dimension
                query = Lambda(lambda x: x[:, :self.p['max_query_len'], :])(query)

                # Build similarity matrix
                if not use_static_matrices:
                    inputs = {max(ngrams): build_matrix([query, doc])}
                    if extra_matrix is not None:
                        inputs['extra_matrix'] = extra_matrix([query, doc])

            # Add query IDF vector directly to FF layers
            if use_query_idf_config:
                doc_qts_scores = [query_idf_score]
            else:
                doc_qts_scores = []

            matrices = [inputs[max(ngrams)]] if extra_matrix is None \
                else [inputs[max(ngrams)], inputs['extra_matrix']]

            for n in ngrams:
                dim_name = get_name(n, n)
                for i, matrix in enumerate(matrices):
                    if i == 1 and n != 1:
                        continue  # dont use convs for extra_matrix
                    if n == 1:
                        # No Convolution for 1-gram
                        re_doc_cov = matrix

                    elif use_convs:
                        # Add channel dimension (1st dimension is batch size)
                        if len(matrix.shape) == 3:
                            matrix = re_input(matrix)
                        else:
                            matrix = matrix

                        # Pass input by Convolution Layer
                        re_doc_cov = conv_layers[dim_name](matrix)

                        if use_masking:
                            mask = inputs["%s_mask" % max(ngrams)]
                            # Add channel dimension (1st dimension is batch size)
                            if len(mask.shape) == 3:
                                mask = re_input(mask)
                                re_doc_cov = Multiply()([re_doc_cov, mask])

                        # Batch normalization after non-linearity
                        if batch_norm:
                            re_doc_cov = batch_norms[dim_name](re_doc_cov)
                    else:
                        # Don't use conv
                        continue

                    if len(re_doc_cov.shape) == 4:
                        # Reduce channels to 1
                        if use_maxpooling:
                            # Pass Conv output through MaxPooling Layer (after permuting) and remove axis=2
                            re_doc_cov = maxpool_layer[dim_name](Permute((1, 3, 2))(re_doc_cov))
                        else:
                            # Pass Conv output through Conv1x1, permute and remove axis=2
                            re_doc_cov = Permute((1, 3, 2))(conv1x1_layers[dim_name](re_doc_cov))
                        
                        if top_k != 0:
                            re_doc_cov = squeeze[dim_name](dropouts[dim_name](re_doc_cov))
                        else:
                            re_doc_cov = Permute((1, 3, 2))(dropouts[dim_name](re_doc_cov))

                        # Batch normalization after non-linearity
                        if batch_norm:
                            re_doc_cov = batch_norms1x1[dim_name](re_doc_cov)

                    else:
                        re_doc_cov = re_input(re_doc_cov) if top_k == 0 else re_doc_cov

                    # Get top_k max values for each row in the matrix (K-MaxPooling Layer)
                    # Or use attention mechanism
                    if use_context:
                        ng_signal = pool_top_k_layer_context[dim_name]([re_doc_cov, inputs['context']])
                    else:
                        # # Attention mechanism
                        # ng_signal = Permute((2, 1))(pool_top_k_layer[dim_name]([doc, query, re_doc_cov]))

                        if top_k != 0:
                            # K-Maxpooling layer
                            ng_signal = pool_top_k_layer[dim_name](re_doc_cov)
                        else:
                            # Multiple convs

                            # Pass input by Convolution Layer
                            re_doc_cov = extra_conv_layers[dim_name](re_doc_cov)
                            if use_maxpooling:
                                # Pass Conv output through MaxPooling Layer (after permuting) and remove axis=2
                                ng_signal = squeeze[dim_name](dropouts[dim_name](extra_maxpool_layer[dim_name](Permute((1, 3, 2))(re_doc_cov))))
                            else:
                                # Pass Conv output through Conv1x1, permute and remove axis=2
                                ng_signal = squeeze[dim_name](dropouts[dim_name](Permute((1, 3, 2))(extra_conv1x1_layers[dim_name](re_doc_cov))))

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
            doc_score = head_layer(doc_qts_score)

            return doc_score

        return _scorer

    def build_cosine_similarity_matrix(self, v1, v2):
        """
            v1 batch_size x [n x d] tensor of n rows with d dimensions
            v2 batch_size x [m x d] tensor of n rows with d dimensions

            returns:
            v3 batch_size x [n x m] tensor of cosine similarity scores between each point i<n, j<m
        """

        v1_l2norm = tf.norm(v1)
        v2_l2norm = tf.norm(v2)

        dist_mat = tf.matmul(v1, v2) / (v1_l2norm * v2_l2norm)

        return dist_mat

    def get_kmax_input(self, input, output):
        if self.intermediate_layers is None:
            assert self.predict_model is not None
            layer_names = ['pool_%s_top%d' % (get_name(n, n), self.p['top_k']) for n in self.p['ngrams']]

            self.intermediate_layers = {x: Model(inputs=self.predict_model.input,
                                                 outputs=self.predict_model.get_layer(x).input)
                                        for x in layer_names}

        return {k: v.predict_on_batch(input) for k, v in self.intermediate_layers.items()}

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
        if self.p['cascade']:
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
        ng = max(self.p['ngrams'])
        inputs = {ng: Input(shape=(self.p['max_query_len'], self.p['max_doc_len']),
                            name='%s_ngram_%d' % (prefix, ng))}
        if self.p['use_masking']:
            inputs["%s_mask" % ng] = Input(shape=(self.p['max_query_len'], self.p['max_doc_len']),
                                               name='%s_ngram_mask_%d' % (prefix, ng))
        return inputs

    def build_train(self):
        # Build train model
        r_query_idf = Input(shape=(self.p['max_query_len'], 1), name='query_idf')
        if self.p['permute']:
            permute_input = Input(shape=(self.p['max_query_len'], 2), name='permute', dtype='int32')
        else:
            permute_input = None

        # Get forward scorer
        doc_scorer = self.forward(r_query_idf, permute_input)

        # {3: Input(shape=(16,800))}
        pos_inputs = {}
        if self.p['use_static_matrices']:
            pos_inputs = self._create_inputs('pos')
        else:
            pos_inputs['query'] = Input(shape=(self.p['max_doc_len'],), name='pos_query')
            pos_inputs['doc'] = Input(shape=(self.p['max_doc_len'],), name='pos_doc')

        if self.p['use_context']:
            pos_inputs['context'] = Input(shape=(self.p['max_query_len'], self.p['max_doc_len']), name='pos_context')

        neg_inputs = {}
        # {0: {3: Input(shape=(16,800))}, ...} 0-> indice of non relevant doc
        # nunmeg -> non relevant docs in softmax
        for neg_ind in range(self.p['num_negative']):
            neg_inputs[neg_ind] = {}
            if self.p['use_static_matrices']:
                neg_inputs[neg_ind] = self._create_inputs('neg%d' % neg_ind)
            else:
                neg_inputs[neg_ind]['query'] = Input(shape=(self.p['max_doc_len'],), name='neg%d_query' % neg_ind)
                neg_inputs[neg_ind]['doc'] = Input(shape=(self.p['max_doc_len'],), name='neg%d_doc' % neg_ind)
            if self.p['use_context']:
                neg_inputs[neg_ind]['context'] = Input(shape=(self.p['max_query_len'], self.p['max_doc_len']),
                                                       name='neg%d_context' % neg_ind)

        # Get score of positive doc and negative doc(s)
        pos_score = doc_scorer(pos_inputs)
        neg_scores = [doc_scorer(neg_inputs[neg_ind]) for neg_ind in range(self.p['num_negative'])]

        # Join pos/neg(s) scores, do softmax and take prob of pos doc
        pos_neg_scores = [pos_score] + neg_scores
        if self.p['custom_loss']:
            loss = 'categorical_crossentropy'
            pos_prob = Lambda(softmax, name='softmax_loss')(pos_neg_scores)
        else:
            loss = 'binary_crossentropy'
            pos_prob = Lambda(pos_softmax, name='pos_softmax_loss')(pos_neg_scores)

        # Construct training inputs
        pos_input_list = [pos_inputs[name] for name in pos_inputs]
        neg_input_list = [neg_inputs[neg_ind][ng] for neg_ind in neg_inputs for ng in neg_inputs[neg_ind]]
        inputs = pos_input_list + neg_input_list

        # Add query IDF vector
        if self.p['query_idf_config']['use_query_idf']:
            inputs.append(r_query_idf)

        # Add permute layer
        if self.p['permute']:
            inputs.append(permute_input)

        # Compile model
        if self.num_gpus > 1:
            with tf.device("/cpu:0"):
                self.train_model = Model(inputs=inputs, outputs=[pos_prob])
                self.train_model = multi_gpu_model(self.train_model, gpus=self.num_gpus)
        else:
            self.train_model = Model(inputs=inputs, outputs=[pos_prob])

        self.train_model.compile(optimizer=Adam(clipvalue=100), loss=loss, metrics=['accuracy'])

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

        doc_inputs = {}
        if self.p['use_static_matrices']:
            doc_inputs = self._create_inputs('doc')
        else:    
            doc_inputs['query'] = Input(shape=(self.p['max_doc_len'],), name='query')
            doc_inputs['doc'] = Input(shape=(self.p['max_doc_len'],), name='doc')
        if self.p['use_context']:
            doc_inputs['context'] = Input(shape=(self.p['max_query_len'], self.p['max_doc_len']), name='doc_context')

        # Get forward scorer
        doc_scorer = self.forward(r_query_idf, permute_idxs=None)

        # Get doc score
        doc_score = doc_scorer(doc_inputs)
        doc_input_list = [doc_inputs[name] for name in doc_inputs]

        # Add query IDF vector
        if self.p['query_idf_config']['use_query_idf']:
            doc_input_list.append(r_query_idf)

        self.predict_model = Model(inputs=doc_input_list, outputs=[doc_score])

    def get_features(self, input, output):
        if self.p['permute']:
            # Add permute
            input['permute'] = np.array(
                [[(bi, qi)
                  for qi in np.random.permutation(input['query_idf'].shape[1])]
                 for bi in range(input['query_idf'].shape[0])],
                dtype=np.int
            )
        return {'input': input, 'output': output}

    def update(self, input, output, class_weight=None):
        if not self.train_model:
            # Build train model
            self.build_train()
            print(self.train_model.summary())

        # Set mode to train
        self.train_mode = True

        # Remove meta-data
        input = {key: input[key] for key in input if key != 'meta-data'}

        # Forward + Backprop
        logs = self.train_model.train_on_batch(input, output['tags'], class_weight=class_weight)

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
        if self.train_mode and self.train_model:
            # Save train weights to temp file and load pred weights from it
            random_fn = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))

            # Get weights from train model
            self.train_model.save_weights('%s.h5' % random_fn)
            self.predict_model.load_weights('%s.h5' % random_fn, by_name=True)
            os.remove("%s.h5" % random_fn)

            # Set mode to predict
            self.train_mode = False

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

    def save(self, sub_name=None):
        path = '%s/%s' % (self.output_dir, type(self).__name__.lower())
        if sub_name:
            assert isinstance(sub_name, str)
            path += sub_name
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            os.system('mkdir -p {}'.format(parent))
        # Save
        self.train_model.save_weights("%s.h5" % path)
