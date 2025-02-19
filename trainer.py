# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Hyperparameters
embedding_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 4
vocab_size = 5000  # Sesuaikan dengan vocabulary size dari tokenizer
max_input_len = 20  # Sesuaikan dengan panjang maksimum input pada preprocessing
batch_size = 64
epochs = 100  # Epoch tinggi untuk meningkatkan akurasi

# Load preprocessed data
padded_inputs = np.load('dataset/padded_inputs.npy')
padded_targets = np.load('dataset/padded_targets.npy')

# Load tokenizer
import pickle
with open('dataset/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# Prepare decoder input and target data
decoder_input_data = padded_targets[:, :-1]  # Semua kecuali token terakhir
decoder_target_data = padded_targets[:, 1:]  # Semua kecuali token pertama

# Transformer block
def transformer_block(inputs, num_heads, ff_dim):
    # Multi-head self-attention
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    # Feed-forward network
    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = tf.keras.layers.Dense(embedding_dim)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
    ffn_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

    return ffn_output

# Encoder
def build_encoder(vocab_size, embedding_dim, num_heads, ff_dim, num_layers, max_input_len):
    inputs = Input(shape=(None,))
    embedding = Embedding(vocab_size, embedding_dim)(inputs)
    x = embedding
    for _ in range(num_layers):
        x = transformer_block(x, num_heads, ff_dim)
    return Model(inputs, x)

# Decoder
def build_decoder(vocab_size, embedding_dim, num_heads, ff_dim, num_layers):
    inputs = Input(shape=(None,))
    encoder_outputs = Input(shape=(None, embedding_dim))
    embedding = Embedding(vocab_size, embedding_dim)(inputs)
    x = embedding
    for _ in range(num_layers):
        x = transformer_block(x, num_heads, ff_dim)
    x = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)
    return Model([inputs, encoder_outputs], x)

# Build the full model
encoder_inputs = Input(shape=(None,))
decoder_inputs = Input(shape=(None,))
encoder = build_encoder(vocab_size, embedding_dim, num_heads, ff_dim, num_layers, max_input_len)
decoder = build_decoder(vocab_size, embedding_dim, num_heads, ff_dim, num_layers)

encoder_outputs = encoder(encoder_inputs)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",  # Loss function
    metrics=["accuracy"]
)

# Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    [padded_inputs, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Save the trained model in Keras Native format
model.save('models/indonesia_sambas_transformer.keras')