import tensorflow as tf

class Encoder(tf.keras.Model):
    """
        The encoder encodes the sentence into a vector of real valued numbers.
    """
    def __init__(self, vocab_len, embed_dim, encode_units, batch_size):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.encode_units = encode_units # input dimension to the RNN layers

        # takes our input, a vector of positive integers representing indicies and turns them into a dense vector
        self.embedding = tf.keras.layers.Embedding(vocab_len, self.embed_dim, mask_zero=True)

        # the GRU layers
        self.gru_forward = tf.keras.layers.GRU(self.encode_units,return_sequences=True, recurrent_initializer='glorot_uniform')
        self.gru_backward = tf.keras.layers.GRU(self.encode_units,return_sequences=True, return_state=True, go_backwards=True, recurrent_initializer='glorot_uniform')

    def call(self, seq_input, hidden_state):
        embed_output = self.embedding(seq_input)

        forward_output = self.gru_forward(embed_output,initial_state=hidden_state)
        backward_output, hidden_state = self.gru_backward(embed_output,initial_state=hidden_state)

        output = tf.concat([forward_output, backward_output], axis = 2)

        return output, hidden_state

    def initalize_hidden(self):
        return tf.zeros((self.batch_size, self.encode_units))
