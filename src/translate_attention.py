import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def translate_with_attention(sentence: str,
                             source_tokenizer, encoder,
                             target_tokenizer, decoder,
                             max_encoding_len,
                             max_translated_len = 30):
    input_seq = source_tokenizer.texts_to_sequences([sentence])
    tokenized = source_tokenizer.sequences_to_texts(input_seq)

    input_seq = pad_sequences(input_seq, maxlen=max_encoding_len, padding='post')
    encoder_output, state_h, state_c = encoder(input_seq)

    current_word = '<sos>'
    decoded_sentence = []

    while len(decoded_sentence) < max_translated_len:
        target_seq = np.zeros((1, 1))
        if current_word in target_tokenizer.word_index:
            target_seq[0, 0] = target_tokenizer.word_index[current_word]
        else:
            break

        logits, state_h, state_c, _ = decoder([target_seq, encoder_output, (state_h, state_c)])
        current_token_index = np.argmax(logits[0].numpy())

        if current_token_index in target_tokenizer.index_word:
            current_word = target_tokenizer.index_word[current_token_index]
        else:
            break

        if current_word == '<eos>':
            break

        decoded_sentence.append(current_word)

    return tokenized[0], ' '.join(decoded_sentence)