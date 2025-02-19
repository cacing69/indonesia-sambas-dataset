import pandas as pd

def count_words(text):
    """Menghitung jumlah kata dalam sebuah string"""
    return len(str(text).split())

def clean_csv(input_file, output_file):
    # Membaca file CSV
    df = pd.read_csv(input_file)

    # Pastikan kolom yang dibutuhkan ada
    if 'indonesia' not in df.columns or 'sambas' not in df.columns:
        raise ValueError("CSV harus memiliki kolom 'indonesia' dan 'sambas'")

    # Filter hanya baris dengan lebih dari satu kata di kedua kolom
    df_cleaned = df[(df['indonesia'].apply(count_words) > 1) & (df['sambas'].apply(count_words) > 1)]

    # Simpan hasil ke file baru
    df_cleaned.to_csv(output_file, index=False)
    print(f"Dataset telah dibersihkan dan disimpan di {output_file}")

# Contoh penggunaan
input_file = "data/translasi_indonesia_sambas.csv"  # Ganti dengan path file CSV kamu
output_file = "data/translasi_indonesia_sambas_cleaned.csv"
clean_csv(input_file, output_file)
