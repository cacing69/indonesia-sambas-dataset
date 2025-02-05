import pandas as pd
import os
import shutil
from datetime import datetime

# Fungsi untuk membuat backup file
def create_backup(input_file):
    try:
        # Pastikan folder 'backup' ada, jika tidak, buat folder tersebut
        if not os.path.exists("backup"):
            os.makedirs("backup")

        # Dapatkan timestamp saat ini
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Buat nama file backup dengan format: timestamp_namafile.csv
        file_name = os.path.basename(input_file)
        backup_file_name = f"{timestamp}_{file_name}"
        backup_path = os.path.join("backup", backup_file_name)

        # Salin file input ke folder backup
        shutil.copy(input_file, backup_path)
        print(f"Backup berhasil dibuat di: {backup_path}")

    except Exception as e:
        print(f"Terjadi kesalahan saat membuat backup: {e}")

# Fungsi untuk membersihkan DataFrame dari baris kosong dan karakter baru
def clean_dataframe(df):
    try:
        # Hapus baris kosong (semua nilai NaN)
        df = df.dropna(how='all')

        # Bersihkan setiap sel dari spasi/tab/karakter tersembunyi
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Ganti string kosong dengan NaN dan hapus baris yang hanya berisi NaN
        df = df.replace('', pd.NA).dropna(how='all')

        return df

    except Exception as e:
        print(f"Terjadi kesalahan saat membersihkan DataFrame: {e}")
        raise

# Fungsi untuk menghapus baris kosong di akhir file
def remove_trailing_newlines(file_path):
    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            # Hapus baris kosong di akhir file
            while lines and lines[-1].strip() == '':
                lines.pop()
            # Tulis ulang file tanpa baris kosong di akhir
            f.seek(0)
            f.writelines(lines)
            f.truncate()
        print(f"Baris kosong di akhir file {file_path} berhasil dihapus.")

    except Exception as e:
        print(f"Terjadi kesalahan saat membersihkan baris kosong di akhir file: {e}")
        raise

# Fungsi utama untuk menghapus duplikat dan baris kosong
def process_csv(input_file, output_file, group_by_columns):
    try:
        # Langkah 1: Membuat backup file input
        create_backup(input_file)

        # Langkah 2: Baca file CSV
        df = pd.read_csv(input_file)

        # Langkah 3: Bersihkan DataFrame dari baris kosong dan karakter baru
        df = clean_dataframe(df)

        # Langkah 4: Periksa apakah kolom yang dimaksud ada dalam dataset
        for column in group_by_columns:
            if column not in df.columns:
                raise ValueError(f"Kolom '{column}' tidak ditemukan dalam dataset.")

        # Langkah 5: Hapus duplikat berdasarkan kolom tertentu
        df_cleaned = df.drop_duplicates(subset=group_by_columns, keep='first')

        # Langkah 6: Simpan hasil ke file CSV baru
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8', lineterminator='\n')

        # Langkah 7: Hapus baris kosong di akhir file
        remove_trailing_newlines(output_file)

        print(f"File berhasil diproses. Hasil disimpan di: {output_file}")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        raise

# Contoh penggunaan
if __name__ == "__main__":
    # Input file CSV
    input_csv = "data/indonesia_sambas.csv"  # Ganti dengan nama file inputmu

    # Output file CSV
    output_csv = "data/indonesia_sambas.csv"  # Nama file output (berbeda dari input)

    # Kolom-kolom yang digunakan untuk grouping (menghapus duplikat)
    columns_to_group_by = ["indonesia", "sambas"]  # Ganti dengan nama kolom yang ingin di-group

    # Proses file CSV
    process_csv(input_csv, output_csv, columns_to_group_by)