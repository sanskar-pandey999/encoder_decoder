import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data_loading import load_data
from src.data_preprocessing import process_dataset
from src.model_attention import Encoder, Decoder, TranslatorTrainer

def main():
    print("Loading dataset...")
    pairs = load_data("data/hungarian_english_pairs.txt")

    print("Preprocessing data and tokenizing...")
    (
        source_tensor,
        target_tensor,
        source_tokenizer,
        target_tokenizer,
        max_len_src,
        max_len_tgt,
    ) = process_dataset(pairs)

    # Teacher forcing format for sequence-to-sequence
    decoder_input_data = target_tensor[:, :-1]
    decoder_target_data = target_tensor[:, 1:]

    # Create 20% validation split to monitor generalization and avoid overfitting
    val_size = int(len(source_tensor) * 0.2)
    
    train_src, val_src = source_tensor[:-val_size], source_tensor[-val_size:]
    train_dec_in, val_dec_in = decoder_input_data[:-val_size], decoder_input_data[-val_size:]
    train_dec_tgt, val_dec_tgt = decoder_target_data[:-val_size], decoder_target_data[-val_size:]

    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_src, train_dec_in, train_dec_tgt
    )).batch(batch_size, drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        val_src, val_dec_in, val_dec_tgt
    )).batch(batch_size, drop_remainder=True)

    num_encoder_tokens = len(source_tokenizer.word_index) + 1
    num_decoder_tokens = len(target_tokenizer.word_index) + 1
    
    embedding_dim = 256
    hidden_dim = 512
    epochs = 30

    print("Initializing Attention Encoder and Decoder...")
    encoder = Encoder(num_encoder_tokens, embedding_dim, hidden_dim)
    decoder = Decoder(num_decoder_tokens, embedding_dim, hidden_dim)
    
    optimizer = tf.keras.optimizers.Adam()
    translator_trainer = TranslatorTrainer(encoder, decoder)
    translator_trainer.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"]
    )

    os.makedirs("models", exist_ok=True)

    print("Starting training with Attention and validation tracking...")
    translator_trainer.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    # Save weights, tokenizers, and metadata for inference & Streamlit
    print("Saving attention model weights and metadata...")
    encoder.save_weights("models/encoder_attn_weights.h5")
    decoder.save_weights("models/decoder_attn_weights.h5")
    
    with open("models/source_tokenizer.pkl", "wb") as f:
        pickle.dump(source_tokenizer, f)
    with open("models/target_tokenizer.pkl", "wb") as f:
        pickle.dump(target_tokenizer, f)
        
    metadata = {
        "max_len_src": max_len_src,
        "max_len_tgt": max_len_tgt,
    }
    with open("models/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Attention model successfully trained, validated, and saved!")

if __name__ == "__main__":
    main()