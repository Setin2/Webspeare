import pandas as pd
import tensorflow as tf
import numpy as np

MAX_LENGTH = 16

def clean(text):
  """
    Clean the text for processing
  """
  text = str(text).lower()

  # remove text between [],(),{} as it is often does not appear in both original/translation
  for start_marker, end_marker in zip(["(","[","{"], [")","]","}"]):
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1 and end != -1:
      text = text[:start] + text[end+1:]

  # Replace quotations and the - sign
  text = text.replace("-", " ")
  text = text.replace("—", " ")
  text = text.replace("\"", "")
  text = text.replace("“","")

  # Change multi spaces to single spaces
  while "  " in text:
    text = text.replace("  ", " ")

  # Remove any leading or trailing whitespace
  text = text.rstrip().lstrip()

  # Check if string is ascii, returning nothing if not
  if not all(ord(c) < 128 for c in text):
    return ""
    
  # Add start and end tokens
  text = "<start> " + text + " <end>"

  return text

def filter_text(corpus):
  """
    Remove empty or invalid lines from the corpus
  """

  new_corpus = []

  for row in corpus:
      valid = True

      # line is not none type
      if row[2] is None or row[3] is None:
        valid = False
        break

      # line is not empty
      if row[2] == "" or row[3] == "":
          valid = False

      # @ sign signifies that the line isnt valid
      if "@" in row[2] or "@" in row[3]:
          valid = False

      # include line if valid
      if valid:
          new_corpus.append([row[2], row[3]])

  return np.array(new_corpus)

def create_dataset():
    corpus = pd.read_csv("../data/corpus.csv").to_numpy()

    row_num = 0

    for row in corpus:
        corpus[row_num][2] = clean(row[2])
        corpus[row_num][3] = clean(row[3])
        row_num += 1

    corpus = filter_text(corpus)

    # seperate out input and target languages
    target_lang = corpus[:,0]
    input_lang = corpus[:,1]

    # create tokenizers and fit on text
    target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '')
    target_tokenizer.fit_on_texts(target_lang)
    
    input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '')
    input_tokenizer.fit_on_texts(input_lang)

    # tokenize corpus
    target_tensor = target_tokenizer.texts_to_sequences(target_lang)
    input_tensor = input_tokenizer.texts_to_sequences(input_lang)

    filtered_input = []
    filtered_target = []

    # remove lines larger than max length
    for i in range(len(input_tensor)):
      if len(input_tensor[i]) <= MAX_LENGTH and len(target_tensor[i]) <= MAX_LENGTH:
        filtered_input.append(input_tensor[i])
        filtered_target.append(target_tensor[i])

    # pad the tensors
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(filtered_target, padding='post')
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(filtered_input, padding='post')

    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int32)
    target_tensor = tf.convert_to_tensor(target_tensor, dtype=tf.int32)

    return [input_tensor, target_tensor], [input_tokenizer, target_tokenizer]