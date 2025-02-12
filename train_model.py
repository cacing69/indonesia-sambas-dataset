# train_model.py
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.optimizers import Adam

# Hyperparameters
embedding_dim = 256
units = 512
batch_size = 64
epochs = 20

# Load preprocessed data
padded_inputs = np.load('data/padded_inputs.npy')
padded_targets = np.load('data/padded_targets.npy')

# Load tokenizer
import pickle
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# Prepare decoder input and target data
decoder_input_data = padded_targets[:, :-1]  # Semua kecuali token terakhir
decoder_target_data = padded_targets[:, 1:]  # Semua kecuali token pertama
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(units, return_state=True)  # Pastikan return_state=True
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    [padded_inputs, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

# Save the trained model in Keras Native format
model.save('models/indonesia_sambas.keras')