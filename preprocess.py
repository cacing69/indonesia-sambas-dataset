# preprocess.py
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv('dataset/translasi_indonesia_sambas.csv')

# Normalisasi teks
df['input_text'] = df['input_text'].str.lower()
df['target_text'] = df['target_text'].str.lower()

# Tambahkan token <start> dan <end> pada target text
df['target_text'] = '<start> ' + df['target_text'] + ' <end>'

# Gabungkan input dan target untuk tokenizer
all_texts = pd.concat([df['input_text'], df['target_text']])

# Tokenizer untuk input dan target
tokenizer = Tokenizer(filters='', oov_token='<OOV>')
tokenizer.fit_on_texts(all_texts)

# Tokenisasi input dan target
input_sequences = tokenizer.texts_to_sequences(df['input_text'])
target_sequences = tokenizer.texts_to_sequences(df['target_text'])

# Padding
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

padded_inputs = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
padded_targets = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary size:", vocab_size)
print("Max input length:", max_input_len)
print("Max target length:", max_target_len)

# Simpan tokenizer untuk digunakan saat inferensi
import pickle
with open('dataset/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Simpan padded data
import numpy as np
np.save('dataset/padded_inputs.npy', padded_inputs)
np.save('dataset/padded_targets.npy', padded_targets)