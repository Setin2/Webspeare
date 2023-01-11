import tensorflow as tf
from attention import Attention

class Decoder(tf.keras.Model):
    """
        The decoder uses the vector from the encoder to predict the target sentence.
    """
    def __init__(self, vocab_len, embed_dim, decode_units, batch_size):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.decode_units = decode_units

        self.embedding = tf.keras.layers.Embedding(vocab_len, embed_dim, mask_zero=True)
        self.GRU = tf.keras.layers.GRU(self.decode_units,return_sequences=True,return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_len)

        self.attention = Attention(self.decode_units)

    def call(self, seq_input, hidden_state, encoder_output):
        # create the context vector and weights for the attention mechanism
        context_vector, attention_weights = self.attention(hidden_state, encoder_output)

        # for the first layer we embedd the input into a dense vector
        embed_output = self.embedding(seq_input)

        concat_output = tf.concat([tf.expand_dims(context_vector, 1), embed_output], axis=-1)

        # 2nd layer, we pass through the GRU
        seq_output, hidden_state = self.GRU(concat_output,initial_state=hidden_state)

        # output passed through the fully connected layer
        output = tf.reshape(seq_output, (-1, seq_output.shape[2]))
        output = self.fc(output)

        return output, hidden_state, attention_weights

    def initalize_hidden(self):
        return tf.zeros((self.batch_size, self.encode_units))