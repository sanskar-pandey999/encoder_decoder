import re
import unicodedata
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def unicode_normalization(s):
  """Normalizes unicode characters (e.g., handling Hungarian accent marks)."""
  return "".join(
      c
      for c in unicodedata.normalize("NFD", s)
      if unicodedata.category(c) != "Mn"
  )


def preprocess_sentence(w):
  """Normalizes unicode, converts to lowercase, and adds spaces around punctuation."""
  w = unicode_normalization(w.lower().strip())
  # Insert spaces between words and punctuation marks (keeping important ones like ?, !, ., ,)
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  return w.strip()


def add_start_end_tokens(target_sentences):
  """Adds start of sentence (<sos>) and end of sentence (<eos>) tokens to target sentences."""
  return [f"<sos> {sent} <eos>" for sent in target_sentences]


from tensorflow.keras.preprocessing.text import Tokenizer

def prepare_tokenizers(source_sentences, target_sentences, max_vocab_size=5000):
    """
    Fits Keras tokenizers for both source and target languages.
    Preserves `<` and `>` so special tokens (<start>, <end>, <unk>) are retained.
    Limits vocabulary size to prevent overfitting on rare words.
    """
    # Custom filter list: Removed '<' and '>' so <start>, <end>, <unk> don't get stripped
    custom_filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'

    # Source Tokenizer (Hungarian)
    source_tokenizer = Tokenizer(
        num_words=max_vocab_size,
        filters=custom_filters,
        oov_token="<unk>"
    )
    source_tokenizer.fit_on_texts(source_sentences)

    # Target Tokenizer (English)
    target_tokenizer = Tokenizer(
        num_words=max_vocab_size,
        filters=custom_filters,
        oov_token="<unk>"
    )
    target_tokenizer.fit_on_texts(target_sentences)

    return source_tokenizer, target_tokenizer


def process_dataset(raw_pairs):
  """Processes raw sentence pairs through normalization, tokenization, and padding."""
  source_raw = [pair[0] for pair in raw_pairs]
  target_raw = [pair[1] for pair in raw_pairs]

  # 1. Clean and normalize sentences
  source_cleaned = [preprocess_sentence(s) for s in source_raw]
  target_cleaned = [
      preprocess_sentence(s) for s in add_start_end_tokens(target_raw)
  ]

  # 2. Build tokenizers
  source_tokenizer, target_tokenizer = prepare_tokenizers(
      source_cleaned, target_cleaned
  )

  # 3. Vectorize text sequences to integers
  source_tensor = source_tokenizer.texts_to_sequences(source_cleaned)
  target_tensor = target_tokenizer.texts_to_sequences(target_cleaned)

  # 4. Pad sequences to fixed lengths (matching longest sequence or custom limits)
  max_length_source = max(len(t) for t in source_tensor)
  max_length_target = max(len(t) for t in target_tensor)

  source_tensor = pad_sequences(
      source_tensor, maxlen=max_length_source, padding="post"
  )
  target_tensor = pad_sequences(
      target_tensor, maxlen=max_length_target, padding="post"
  )


  return (
      source_tensor,
      target_tensor,
      source_tokenizer,
      target_tokenizer,
      max_length_source,
      max_length_target,

  )