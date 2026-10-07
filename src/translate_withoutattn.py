import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import local preprocessing helper
from src.data_preprocessing import preprocess_sentence

# Resolve paths relative to project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 1. LOAD TOKENIZERS
with open(os.path.join(MODELS_DIR, 'source_tokenizer.pkl'), 'rb') as f:
    source_tokenizer = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'target_tokenizer.pkl'), 'rb') as f:
    target_tokenizer = pickle.load(f)

# 2. LOAD STANDALONE INFERENCE MODELS
encoder_model_no_attention = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, 'encoder_no_attn.keras')
)
decoder_model_no_attention = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, 'decoder_no_attn.keras')
)


# 3. TRANSLATION FUNCTION WITH DEFAULT ARGUMENTS & UNK FIX
def translate_without_attention(
    sentence: str,
    source_tokenizer=source_tokenizer,
    encoder=encoder_model_no_attention,
    target_tokenizer=target_tokenizer,
    decoder=decoder_model_no_attention,
    max_translated_len=30,
    max_encoder_len=63,
):
    # Preprocess input sentence
    processed_sentence = preprocess_sentence(sentence)

    # Vectorize source sentence
    input_seq = source_tokenizer.texts_to_sequences([processed_sentence])

    # Extract unpadded tokenized representation for UI display
    tokenized_sentence = source_tokenizer.sequences_to_texts(input_seq)

    # Pad sequence to match expected model shape (None, 63)
    padded_input_seq = pad_sequences(
        input_seq, maxlen=max_encoder_len, padding='post'
    )

    # Pass through encoder to extract context states
    states = encoder.predict(padded_input_seq, verbose=0)

    current_word = '<sos>'
    decoded_sentence = []

    # Auto-regressive decoding loop
    while len(decoded_sentence) < max_translated_len:
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target_tokenizer.word_index.get(current_word, 1)

        # Predict next token and updated states
        target_y_proba, h, c = decoder.predict([target_seq] + states, verbose=0)
        target_token_index = np.argmax(target_y_proba[0, -1, :])
        current_word = target_tokenizer.index_word.get(target_token_index, '')

        if current_word == '<eos>' or not current_word:
            break

        # Filter out unknown/out-of-vocabulary tokens from final string
        if current_word != '<unk>':
            decoded_sentence.append(current_word)

        states = [h, c]

    return tokenized_sentence, ' '.join(decoded_sentence)


# 4. LOCAL EXECUTION TEST
if __name__ == '__main__':
    my_custom_sentence = 'A kék ég alatt minden lehetséges.'

    tokenized_input, translated_output = translate_without_attention(
        my_custom_sentence
    )

    print('Original Sentence:', my_custom_sentence)
    print('Tokenized Input:', tokenized_input)
    print('Translated Output:', translated_output)