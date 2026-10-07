import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.models import Model


def build_train_model(
    num_encoder_tokens,
    num_decoder_tokens,
    max_len_src,
    max_len_tgt_decoder,
    latent_dim=256,
    embedding_dim=256,
):
  # Encoder
  encoder_inputs = Input(shape=(max_len_src,), name='encoder_inputs')
  encoder_embedding = Embedding(
      num_encoder_tokens, embedding_dim, mask_zero=True, name='encoder_embed'
  )(encoder_inputs)
  encoder_lstm = LSTM(
      latent_dim, return_state=True, name='encoder_lstm', dropout=0.2, recurrent_dropout=0.2
  )
  _, state_h, state_c = encoder_lstm(encoder_embedding)
  encoder_states = [state_h, state_c]

  # Decoder
  decoder_inputs = Input(shape=(max_len_tgt_decoder,), name='decoder_inputs')
  decoder_embedding = Embedding(
      num_decoder_tokens, embedding_dim, mask_zero=True, name='decoder_embed'
  )(decoder_inputs)
  decoder_lstm = LSTM(
      latent_dim, return_sequences=True, return_state=True, name='decoder_lstm', dropout=0.2, recurrent_dropout=0.2
  )
  decoder_outputs, _, _ = decoder_lstm(
      decoder_embedding, initial_state=encoder_states
  )

  decoder_dense = Dense(
      num_decoder_tokens, activation='softmax', name='decoder_dense'
  )
  decoder_outputs = decoder_dense(decoder_outputs)

  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
  return model


def extract_inference_models(
    model, num_encoder_tokens, num_decoder_tokens, latent_dim=256
):
  # Encoder for inference
  encoder_inputs = model.get_layer('encoder_inputs').input
  encoder_lstm = model.get_layer('encoder_lstm')
  _, state_h, state_c = encoder_lstm(
      model.get_layer('encoder_embed')(encoder_inputs)
  )
  encoder_states = [state_h, state_c]
  encoder_model = Model(encoder_inputs, encoder_states)

  # Decoder for inference
  decoder_inputs = Input(shape=(1,), name='dec_inf_input')
  decoder_state_input_h = Input(shape=(latent_dim,), name='dec_state_h')
  decoder_state_input_c = Input(shape=(latent_dim,), name='dec_state_c')
  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

  decoder_embed = model.get_layer('decoder_embed')(decoder_inputs)
  decoder_lstm = model.get_layer('decoder_lstm')
  decoder_outputs, state_h, state_c = decoder_lstm(
      decoder_embed, initial_state=decoder_states_inputs
  )
  decoder_states = [state_h, state_c]

  decoder_dense = model.get_layer('decoder_dense')
  decoder_outputs = decoder_dense(decoder_outputs)

  decoder_model = Model(
      [decoder_inputs] + decoder_states_inputs,
      [decoder_outputs] + decoder_states,
  )

  return encoder_model, decoder_model