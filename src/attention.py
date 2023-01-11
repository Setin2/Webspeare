import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    """
        The attention layer is used to assist with sentence alignment. 
        The attention weights are simply calculated with a dot product
    """
    def __init__(self, units):
        super(Attention, self).__init__()

        # used to calculate score
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # brodacast addition over the time axis
        query = tf.expand_dims(query, 1)

        # calculate the attention score with a MLP
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))

        # softmax calculates the attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # calculate the context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights