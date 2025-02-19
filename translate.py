# translate.py
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('dataset/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load trained model in Keras Native format
model = load_model('models/indonesia_sambas_transformer.keras')

# Hyperparameters
embedding_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 4
vocab_size = len(tokenizer.word_index) + 1
max_input_len = 20  # Sesuaikan dengan panjang maksimum input pada pelatihan

# Translate function
def translate(input_text, source_lang, target_lang):
    # Add language tokens
    input_text = f"<{source_lang}> {input_text}"
    target_prefix = f"<{target_lang}>"

    # Tokenize input text
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get(target_prefix, 0)

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens = model.predict([input_seq, target_seq])

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

    return decoded_sentence.strip().replace('<end>', '')

# Test translation
input_text = "apa yang kamu lakukan?"
translated_text = translate(input_text, source_lang="id", target_lang="sambas")
print(f"Input: {input_text}")
print(f"Translated: {translated_text}")

# Reverse translation
reverse_text = translate(translated_text, source_lang="sambas", target_lang="id")
print(f"Reverse Translated: {reverse_text}")