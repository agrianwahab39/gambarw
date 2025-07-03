# Sistem Deteksi Forensik Keaslian Gambar

Repositori ini menyediakan serangkaian modul Python untuk menganalisis keaslian gambar secara forensik. Modul-modul utama mencakup klasifikasi manipulasi, validasi silang berbagai teknik, serta utilitas pendukung seperti penyimpanan riwayat hasil analisis.

## Persyaratan

- Python 3.8 atau lebih baru
- Paket pihak ketiga: `numpy`, `opencv-python`, `Pillow`, `pytest`, `coverage`, `radon`

Instalasi cepat:

```bash
pip install numpy opencv-python Pillow pytest coverage radon
```

Tambahkan paket lain sesuai kebutuhan (misalnya `scikit-learn`, `scipy`) apabila tersedia.

## Menjalankan Analisis Gambar

1. Siapkan gambar yang akan diuji.
2. Jalankan pipeline utama:

```bash
python main.py path/to/image.jpg
```

Opsi tambahan dapat dilihat melalui `python main.py -h`.

## Memulai Pengujian Basis Path Testing

Basis path testing memberikan gambaran kerumitan kode serta cakupan jalur yang telah dieksekusi.

1. Pastikan `pytest`, `coverage` dan `radon` telah terpasang seperti pada bagian persyaratan.
2. Jalankan skrip berikut dari direktori repositori:

```bash
python -m py_compile classification.py app2.py utils.py main.py basis_path_report.py
pip install radon coverage pytest
python basis_path_report.py
python basis_path_report.py
```

3. Setelah selesai, buka `BASIS_PATH_REPORT.md` untuk melihat metrik siklomatik, jumlah jalur dasar, cakupan, serta jumlah bug (jika ada tes yang gagal).

Laporan detail mengenai perbaikan isu ditemukan pada `VALIDATION_FIX_REPORT.md`.
