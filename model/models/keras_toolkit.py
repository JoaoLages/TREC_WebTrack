#!/usr/bin/python

from keras.layers import Activation, K
from keras.engine.topology import Layer
import tensorflow as tf


class AttentionMechanism(Layer):
    """
        Attention layer for 2 2D inputs
        Implements the operation:
            Input3 * Softmax(V * tanh(Input1 * A, Input2 * B))
            A and B are learnable matrices, V is a learnable vector
    """
    def __init__(self, matrix_shape, input_dimA, input_dimB, activation_function=tf.tanh, **kwargs):

        assert input_dimB == 1
        assert isinstance(matrix_shape, tuple)

        self.matrix_shape = matrix_shape
        self.input_dimA = input_dimA
        self.input_dimB = input_dimB

        self.activation_function = activation_function

        super(AttentionMechanism, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight matrix for this layer.
        self.A = self.add_weight(name='weight_matrixA',
                                 shape=self.matrix_shape,
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.B = self.add_weight(name='weight_matrixB',
                                 shape=self.matrix_shape,
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.V = self.add_weight(name='weight_vectorV',
                                 shape=(self.matrix_shape[1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionMechanism, self).build(input_shape)

    def call(self, inputs):
        assert type(inputs) is list and len(inputs) > 1

        t1, t2, t3 = inputs

        # Project t1 and t2 into embedded space
        t1_p = tf.reshape(tf.matmul(tf.reshape(t1, [-1, t1.shape[2]]), self.A), [-1, t1.shape[1], self.A.shape[1]])  # (?, 800, X)
        t2_p = tf.reshape(tf.matmul(tf.reshape(t2, [-1, t2.shape[2]]), self.B), [-1, t2.shape[1], self.B.shape[1]])  # (?, 1, X)
        
        # Add both and pass through activation function
        attended_matrix = self.activation_function(tf.add(t1_p, t2_p))  # (?, X, 800)
        
        # Multiply with V and switch axis 1,2
        attended_vector = tf.transpose(
            tf.reshape(
                tf.matmul(tf.reshape(attended_matrix, [-1, attended_matrix.shape[2]]), self.V),
                [-1, attended_matrix.shape[1], self.V.shape[1]]
            ),
            [0, 2, 1]
        )
       
        # Return softmax of attended_vector
        softmax_vector = tf.nn.softmax(attended_vector)  # (?, 1, 800)
                
        return tf.reduce_sum(tf.multiply(t3, softmax_vector), 2, keep_dims=True)  # (?, 1, 1) 

    def compute_output_shape(self, input_shape):
        return (None, 1, 1)


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# Capsule Network Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def set_LearningRateDecay(model, nr_samples, epochs, batch_size, lr_init, lr_final):
    steps = int(nr_samples / batch_size) * epochs

    exp_decay = lambda init, final, steps: (init / final) ** (1 / (steps - 1)) - 1
    lr_decay = exp_decay(lr_init, lr_final, steps)
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)
