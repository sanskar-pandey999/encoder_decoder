import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# here we import features from other files in our visual studio code (new thing learned)
from src.data_loading import load_data
from src.data_preprocessing import process_dataset
from src.model_no_attn import build_train_model, extract_inference_models


def main():
  print("Loading saved local dataset...")
  # Directly load your already saved local text file
  pairs = load_data("data/hungarian_english_pairs.txt")

  print("Preprocessing text and building tokenizers...")
  (
      source_tensor,
      target_tensor,
      source_tokenizer,
      target_tokenizer,
      max_len_src,
      max_len_tgt,
  ) = process_dataset(pairs)

  num_encoder_tokens = len(source_tokenizer.word_index) + 1
  num_decoder_tokens = len(target_tokenizer.word_index) + 1

  # Teacher Forcing split:
  # Decoder inputs: all tokens except the last one
  # Decoder targets: all tokens shifted over by one (excluding the first sos token)
  decoder_input_data = target_tensor[:, :-1]
  decoder_target_data = target_tensor[:, 1:]
  max_len_tgt_decoder = decoder_input_data.shape[1]

  print(f"Encoder Input Shape: {source_tensor.shape}")
  print(f"Decoder Input Shape: {decoder_input_data.shape}")
  print(f"Decoder Target Shape: {decoder_target_data.shape}")

  print("Building Seq2Seq model...")
  model = build_train_model(
      num_encoder_tokens,
      num_decoder_tokens,
      max_len_src,
      max_len_tgt_decoder,
  )

  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["sparse_categorical_accuracy"],
  )
  model.summary()

  # Create directories for saving outputs if they don't exist
  os.makedirs("models", exist_ok=True)

  # Callbacks
  checkpoint_path = "models/seq2seq_checkpoint.keras"
  checkpoint = ModelCheckpoint(
      checkpoint_path, monitor="val_loss", save_best_only=True, mode="min"
  )
  early_stopping = EarlyStopping(
      monitor="val_loss", patience=3, restore_best_weights=True
  )

  print("Starting training...")
  model.fit(
      [source_tensor, decoder_input_data],
      decoder_target_data,
      batch_size=64,
      validation_split=0.2,
      epochs=30,
      callbacks=[checkpoint, early_stopping],
  )

  print("Extracting and saving standalone inference models...")
  encoder_model, decoder_model = extract_inference_models(
      model, num_encoder_tokens, num_decoder_tokens
  )

  encoder_model.save("models/encoder_no_attn.keras")
  decoder_model.save("models/decoder_no_attn.keras")
  print("Training complete! Models saved successfully in the 'models/' folder.")
  print("Saving tokenizers and metadata for inference...")
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
  print("Training complete! Models, tokenizers, and metadata saved successfully in 'models/'.")
if __name__ == "__main__":
  main()


# doubt why do we save tokenizers using pickle and models using keras ?
# high accuracy and low loss shows overfitting we will change early stopping to monitor from 'loss' to 'val_loss' and introduce validation_split=0.2 in model.fit
# based on loss for early stopping values are loss=0.5435 while sparse_categorical_accuracy =0.8980
# loss measures memorisation val_loss measures generalisation
# using validation loss we found val_loss=5.2848 and val_sparse_categorical_accuracy=0.3077
# we added dropouts to improve accuracy of 0.2 and  we made vocab size smaller 
# aafter this sparse_categoricalaccuracy on training is 45 percentage and on validation is 32 percentage