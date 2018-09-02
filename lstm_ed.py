import tensorflow as tf
import tensorflow.contrib.rnn as tfrnn


def fc_layer(n_units, activation, input):


    W = tf.get_variable("W", shape=[input.get_shape().as_list()[-1], n_units],
                        dtype=tf.float32,
                        initializer= tf.contrib.layers.xavier_initializer())

    b = tf.get_variable("b", shape=[n_units], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    current = tf.matmul(input, W) + b

    if not activation is None:
        current = activation(current)

    return current

class LSTMED:

    def __init__(self, num_units, batch_size, input, input_size):

        self.num_units = num_units
        self.input_size = input_size
        self.batch_size = batch_size

        self.input = input

        self.__define_network()

    def __define_network(self):

        # encoder
        with tf.variable_scope('encoder'):
            lstm_encoder = tfrnn.LayerNormBasicLSTMCell(self.num_units,layer_norm= False)
            initial_state = lstm_encoder.zero_state(batch_size= self.batch_size, dtype= tf.float32)

            _ , encoder_state = tf.nn.dynamic_rnn(cell= lstm_encoder,
                                                  inputs= self.input,
                                                  initial_state = initial_state)

        # decoder
        with tf.variable_scope('decoder'):
            reversed_input = tf.reverse(self.input, axis= [1])
            reversed_padded_input = tf.pad(reversed_input, paddings=[[0,0], [1,0], [0,0]])

            lstm_decoder = tfrnn.LayerNormBasicLSTMCell(self.num_units,layer_norm= False)

            decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell= lstm_decoder,
                                                               inputs= reversed_padded_input[:,:-1,:],
                                                               initial_state = encoder_state)

        # output layer
        with tf.variable_scope('output'):
            decoder_outputs = tf.reshape(decoder_outputs, [-1, self.num_units])
            output = fc_layer(self.input_size,activation= None, input= decoder_outputs)


        # loss
        self.point_reconstruction_error = tf.squared_difference(tf.reshape(reversed_input, [-1, self.input_size]), output)
        self.point_reconstruction_error = tf.reshape(self.point_reconstruction_error, [self.batch_size, -1 , self.input_size])
        self.point_reconstruction_error = tf.reduce_sum(self.point_reconstruction_error, axis= -1)
        self.point_reconstruction_error = tf.reverse(self.point_reconstruction_error, axis= [1])

        self.loss = tf.reduce_mean(self.point_reconstruction_error)


