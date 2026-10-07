import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(LuongAttention, self).__init__(**kwargs)
        self.wa = tf.keras.layers.Dense(hidden_dim)

    def call(self, decoder_outputs, encoder_outputs):
        # decoder_outputs: (batch_size, target_seq_len, hidden_dim)
        # encoder_outputs: (batch_size, source_seq_len, hidden_dim)

        # Query projection
        query = self.wa(decoder_outputs)  # (batch_size, target_seq_len, hidden_dim)

        # Attention scores: (batch_size, target_seq_len, source_seq_len)
        score = tf.matmul(query, encoder_outputs, transpose_b=True)

        # Alignment probabilities
        attention_weights = tf.nn.softmax(score, axis=-1)

        # Context vector: (batch_size, target_seq_len, hidden_dim)
        context_vector = tf.matmul(attention_weights, encoder_outputs)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.attention = LuongAttention(hidden_dim)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.wc = tf.keras.layers.Dense(hidden_dim, activation='tanh')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        target_seq, encoder_outputs, states = inputs

        x = self.embedding(target_seq)
        decoder_outputs, state_h, state_c = self.lstm(x, initial_state=states)

        # Luong attention across all sequence positions
        context_vector, attention_weights = self.attention(decoder_outputs, encoder_outputs)

        # Concatenate context vector and decoder LSTM outputs
        concat_input = self.concat([context_vector, decoder_outputs])
        attention_vector = self.wc(concat_input)

        logits = self.fc(attention_vector)
        return logits, state_h, state_c, attention_weights


class TranslatorTrainer(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(TranslatorTrainer, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        if isinstance(inputs, (tuple, list)):
            source_seq, decoder_input = inputs[0], inputs[1]
        else:
            source_seq, decoder_input = inputs, inputs

        encoder_outputs, state_h, state_c = self.encoder(source_seq, training=training)
        logits, _, _, attention_weights = self.decoder(
            [decoder_input, encoder_outputs, [state_h, state_c]],
            training=training
        )
        return logits

    def train_step(self, data):
        # Unpack data dynamically depending on tuple format
        if isinstance(data, (list, tuple)) and len(data) == 2:
            inputs, decoder_target = data
            if isinstance(inputs, (list, tuple)):
                source_seq, decoder_input = inputs[0], inputs[1]
            else:
                source_seq, decoder_input = inputs, inputs
        elif isinstance(data, (list, tuple)) and len(data) == 3:
            source_seq, decoder_input, decoder_target = data
        else:
            source_seq, decoder_input, decoder_target = data[0], data[1], data[2]

        with tf.GradientTape() as tape:
            encoder_outputs, state_h, state_c = self.encoder(source_seq, training=True)
            logits, _, _, _ = self.decoder(
                [decoder_input, encoder_outputs, [state_h, state_c]],
                training=True
            )
            loss = self.compiled_loss(decoder_target, logits)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(decoder_target, logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, (list, tuple)) and len(data) == 2:
            inputs, decoder_target = data
            if isinstance(inputs, (list, tuple)):
                source_seq, decoder_input = inputs[0], inputs[1]
            else:
                source_seq, decoder_input = inputs, inputs
        elif isinstance(data, (list, tuple)) and len(data) == 3:
            source_seq, decoder_input, decoder_target = data
        else:
            source_seq, decoder_input, decoder_target = data[0], data[1], data[2]

        encoder_outputs, state_h, state_c = self.encoder(source_seq, training=False)
        logits, _, _, _ = self.decoder(
            [decoder_input, encoder_outputs, [state_h, state_c]],
            training=False
        )
        self.compiled_loss(decoder_target, logits)
        self.compiled_metrics.update_state(decoder_target, logits)

        return {m.name: m.result() for m in self.metrics}