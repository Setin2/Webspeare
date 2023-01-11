from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
import tokenize_corpus
import time
import os

NUM_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.001
EMBEDDING_DIM = 256
ENCODER_UNITS = 1024

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")

def loss_function(actual, pred):
    loss = loss_obj(actual, pred)

    # masking on end of sentence marker
    mask = tf.math.logical_not(tf.math.equal(0,actual))
    mask = tf.cast(mask,dtype=loss.dtype)

    loss = loss * mask
    loss = tf.reduce_mean(loss)

    return loss

def train(num_epochs, steps_per_epoch, lr, batch_size, embed_dim, encode_units):

  @tf.function
  def train_step(input_seq, target, encoder_hidden):
    loss = 0

    with tf.GradientTape() as tape:
      # run the encoder on the input
      encoder_output, encoder_hidden = encoder(input_seq, encoder_hidden)
      # initalize first decoder hidden states to final encoder hidden states
      decoder_hidden = encoder_hidden
      decoder_input = tf.expand_dims([target_lang.word_index['<start>']] * batch_size, 1)

      for t in range(1, target.shape[1]):
        # pass encoder output into decoder
        prediction, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
                
        loss += loss_function(target[:, t], prediction)
        decoder_input = tf.expand_dims(target[:, t], 1)

      batch_loss = (loss / int(target.shape[1]))

      # apply gradients
      variables = encoder.trainable_variables + decoder.trainable_variables
      gradients = tape.gradient(loss, variables)
      optim.apply_gradients(zip(gradients, variables))

      return batch_loss

  for epoch in range(1, num_epochs + 1): # training loop
    total_loss = 0
    start = time.time()
    encoder_hidden = encoder.initalize_hidden()

    for (batch, (input_batch, target_batch)) in enumerate(dataset.take(steps_per_epoch)):

      batch_loss = train_step(input_batch, target_batch, encoder_hidden)
      total_loss += batch_loss

      #if batch % 50 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch, batch_loss.numpy()))

    # save checkpoint
    checkpoint.save(file_prefix = checkpoint_prefix)

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

if __name__ == "__main__":
    # create Tensorflow dataset
    tensors, tokenizers = tokenize_corpus.create_dataset()
    input_tensor, target_tensor = tensors
    input_lang, target_lang = tokenizers

    # create batches from a tensorflow dataset, shuffle the data for training
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(len(input_tensor))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    vocab_input_size = len(input_lang.word_index) + 1
    vocab_target_size = len(target_lang.word_index) + 1

    steps_per_epoch = len(input_tensor) // BATCH_SIZE

    # define neural networks & optimizer
    encoder = Encoder(vocab_input_size, EMBEDDING_DIM, ENCODER_UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_target_size, EMBEDDING_DIM, ENCODER_UNITS, BATCH_SIZE)
    optim = tf.keras.optimizers.Adam(learning_rate = LR)

    # checkpoints for saving
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer = optim, encoder = encoder, decoder = decoder)

    train(NUM_EPOCHS, steps_per_epoch, LR, BATCH_SIZE, EMBEDDING_DIM, ENCODER_UNITS)