# translate.py
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load trained model
model = load_model('models/multilingual_nmt.h5')  # Pastikan file model ada di folder models/

# Hyperparameters
embedding_dim = 256
units = 512
max_input_len = 20  # Sesuaikan dengan panjang maksimum input pada pelatihan

# Encoder model for inference
encoder_inputs = model.input[0]  # Input layer untuk encoder
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # LSTM layer output (encoder)
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

# Debugging: Cetak ringkasan encoder model
print("Encoder Model Summary:")
print(encoder_model.summary())

# Decoder model for inference
decoder_inputs = Input(shape=(None,))  # Input layer untuk decoder
decoder_embedding_layer = model.layers[1]  # Embedding layer untuk decoder
decoder_lstm = model.layers[3]  # LSTM layer untuk decoder
decoder_dense = model.layers[4]  # Dense layer untuk decoder

# Tambahkan layer Embedding untuk decoder
decoder_embedding = decoder_embedding_layer(decoder_inputs)

# Input states untuk decoder
decoder_state_input_h = Input(shape=(units,))
decoder_state_input_c = Input(shape=(units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# LSTM layer untuk decoder dengan initial state
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_outputs = decoder_dense(decoder_outputs)

# Model decoder untuk inferensi
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Translate function
def translate(input_text, source_lang, target_lang):
    # Add language tokens
    input_text = f"<{source_lang}> {input_text}"
    target_prefix = f"<{target_lang}>"

    # Tokenize input text
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

    # Debugging: Cetak shape input_seq
    print("Input sequence shape:", input_seq.shape)

    # Encode input sequence
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get(target_prefix, 0)  # Pastikan token ada di tokenizer

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '<UNK>')
        decoded_sentence += ' ' + sampled_word

        # Exit condition
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_input_len:
            stop_condition = True

        # Update target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence.strip().replace('<end>', '')

# Test translation
input_text = "aku makan malam"
translated_text = translate(input_text, source_lang="id", target_lang="sambas")
print(f"Input: {input_text}")
print(f"Translated: {translated_text}")

# Reverse translation
reverse_text = translate(translated_text, source_lang="sambas", target_lang="id")
print(f"Reverse Translated: {reverse_text}")