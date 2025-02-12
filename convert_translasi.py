# convert_dataset.py
import pandas as pd

def convert_to_multilingual(input_file, output_file):
    # Load dataset awal
    df = pd.read_csv(input_file)

    # Inisialisasi list untuk menyimpan data baru
    multilingual_data = []

    # Iterasi setiap baris dalam dataset
    for _, row in df.iterrows():
        indonesia = row['indonesia'].strip()
        sambas = row['sambas'].strip()

        # Tambahkan pasangan terjemahan Indonesia → Sambas
        multilingual_data.append({
            'input_text': f"<id> {indonesia}",
            'target_text': f"<sambas> {sambas}"
        })

        # Tambahkan pasangan terjemahan Sambas → Indonesia
        multilingual_data.append({
            'input_text': f"<sambas> {sambas}",
            'target_text': f"<id> {indonesia}"
        })

    # Buat DataFrame baru untuk dataset multilingual
    multilingual_df = pd.DataFrame(multilingual_data)

    # Simpan dataset multilingual ke file CSV
    multilingual_df.to_csv(output_file, index=False)
    print(f"Dataset multilingual berhasil disimpan di {output_file}")

# Jalankan konverter
if __name__ == "__main__":
    input_file = 'data/translasi_indonesia_sambas.csv'  # Ganti dengan nama file dataset awalmu
    output_file = 'data/dataset_multilingual.csv'  # Nama file output
    convert_to_multilingual(input_file, output_file)