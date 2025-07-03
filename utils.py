"""
Utility functions for Forensic Image Analysis System
"""

import numpy as np
import warnings
import json
from datetime import datetime
import os
import shutil
from validator import ForensicValidator
warnings.filterwarnings('ignore')

def detect_outliers_iqr(data, factor=1.5):
    """Detect outliers using IQR method"""
    if len(data) == 0:
        return np.array([], dtype=int)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]

def calculate_skewness(data):
    """Calculate skewness with empty-array handling"""
    if len(data) == 0:
        return 0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis with empty-array handling"""
    if len(data) == 0:
        return 0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def normalize_array(arr):
    """Normalize array to 0-1 range"""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator

# ======================= FUNGSI RIWAYAT ANALISIS (DIPERBARUI) =======================

HISTORY_FILE = 'analysis_history.json'
THUMBNAIL_DIR = 'history_thumbnails'

def load_analysis_history():
    """
    Memuat riwayat analisis dari file JSON.
    Mengembalikan list kosong jika file tidak ada atau rusak.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            # Handle empty file case
            content = f.read()
            if not content:
                return []
            history = json.loads(content)
        # Pastikan data yang dimuat adalah list
        if not isinstance(history, list):
            print(f"Peringatan: Isi dari '{HISTORY_FILE}' bukan sebuah list. Mengembalikan list kosong.")
            return []
        return history
    except json.JSONDecodeError:
        print(f"Peringatan: Gagal membaca '{HISTORY_FILE}' karena format JSON tidak valid. Mengembalikan list kosong.")
        return []
    except Exception as e:
        print(f"Peringatan: Terjadi error saat memuat riwayat: {e}. Mengembalikan list kosong.")
        return []

def save_analysis_to_history(image_name, analysis_summary, processing_time, thumbnail_path):
    """
    Menyimpan ringkasan analisis baru ke dalam file riwayat JSON.
    """
    history = load_analysis_history()

    new_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': image_name,
        'thumbnail_path': thumbnail_path,
        'analysis_summary': analysis_summary,
        'processing_time': processing_time
    }
    
    history.append(new_entry)

    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error: Gagal menyimpan riwayat ke '{HISTORY_FILE}': {e}")

def delete_all_history():
    """
    Menghapus semua riwayat analisis.
    Mengembalikan True jika berhasil, False jika gagal.
    """
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        
        # Hapus juga folder thumbnails jika ada
        if os.path.exists(THUMBNAIL_DIR):
            shutil.rmtree(THUMBNAIL_DIR)
        
        print("✅ Semua riwayat analisis berhasil dihapus.")
        return True
    except Exception as e:
        print(f"❌ Error menghapus riwayat: {e}")
        return False

def delete_selected_history(selected_indices):
    """
    Menghapus riwayat analisis yang dipilih berdasarkan indeks.
    
    Args:
        selected_indices (list): List indeks yang akan dihapus
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    history = load_analysis_history()
    
    if not history:
        print("⚠️ Tidak ada riwayat untuk dihapus.")
        return False
        
    try:
        # Kumpulkan path thumbnail yang akan dihapus
        thumbnails_to_delete = []
        valid_indices = [i for i in selected_indices if 0 <= i < len(history)]
        
        if not valid_indices:
            print("⚠️ Tidak ada indeks valid yang dipilih.")
            return False

        # Buat daftar item yang akan disimpan (bukan dihapus)
        history_to_keep = [entry for i, entry in enumerate(history) if i not in valid_indices]

        # Identifikasi thumbnail yang akan dihapus
        for idx in valid_indices:
            thumbnail_path = history[idx].get('thumbnail_path')
            if thumbnail_path and os.path.exists(thumbnail_path):
                thumbnails_to_delete.append(thumbnail_path)
        
        # Simpan history yang sudah diperbarui
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_keep, f, indent=4)
        
        # Hapus file thumbnail yang sudah tidak terpakai
        for thumbnail_path in thumbnails_to_delete:
            try:
                os.remove(thumbnail_path)
            except OSError as e:
                print(f"⚠️ Gagal menghapus thumbnail {thumbnail_path}: {e}")
        
        print(f"✅ Berhasil menghapus {len(valid_indices)} entri riwayat.")
        return True
        
    except Exception as e:
        print(f"❌ Error menghapus riwayat terpilih: {e}")
        # Kembalikan file history ke state semula jika terjadi error
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        return False


def get_history_count():
    """
    Mengembalikan jumlah entri dalam riwayat analisis.
    """
    history = load_analysis_history()
    return len(history)

def clear_empty_thumbnail_folder():
    """
    Menghapus folder thumbnail jika kosong.
    """
    try:
        if os.path.exists(THUMBNAIL_DIR) and not os.listdir(THUMBNAIL_DIR):
            os.rmdir(THUMBNAIL_DIR)
            print("📁 Folder thumbnail kosong telah dihapus.")
    except Exception as e:
        print(f"⚠️ Error membersihkan folder thumbnail: {e}")

# ======================= AKHIR DARI FUNGSI RIWAYAT =======================

def test_forensic_validator_cross_algorithm():
    """
    Tests the 'validate_cross_algorithm' method in ForensicValidator.
    This specifically targets the function from app2.py.
    """
    # --- 1. SETUP: Create a mock analysis_results dictionary ---
    # Data ini mensimulasikan hasil analisis yang "baik" dan konsisten
    mock_results_good = {
        'localization_analysis': {
            'kmeans_localization': {
                'cluster_ela_means': [5.1, 18.3, 4.9],
                'tampering_cluster_id': 1,
                'tampering_percentage': 15.5,
            },
            'combined_tampering_mask': True # Menandakan mask ada
        },
        'ela_image': "dummy_image_object", # Cukup string palsu untuk pengecekan
        'ela_mean': 12.5,
        'ela_std': 20.1,
        'ela_regional_stats': {'regional_inconsistency': 0.4, 'outlier_regions': 3},
        'noise_analysis': {'overall_inconsistency': 0.35},
        'ransac_inliers': 25,
        'sift_matches': 150,
        'geometric_transform': ('affine', 'dummy_matrix'),
        'block_matches': [1] * 15 # 15 block matches
    }

    # Data ini mensimulasikan hasil yang "buruk" dan tidak konsisten
    mock_results_bad = {
        'localization_analysis': {
            'kmeans_localization': {
                'cluster_ela_means': [5.1, 5.3], # Tidak ada perbedaan signifikan
                'tampering_cluster_id': -1,
                'tampering_percentage': 0.1, # Terlalu kecil
            },
        },
        'ela_image': "dummy_image_object",
        'ela_mean': 2.1,
        'ela_std': 5.5,
        'ela_regional_stats': {'regional_inconsistency': 0.05, 'outlier_regions': 0},
        'noise_analysis': {'overall_inconsistency': 0.02},
        'ransac_inliers': 1,
        'sift_matches': 5,
        'geometric_transform': None,
        'block_matches': []
    }

    # --- 2. ACTION: Instantiate the validator and run the method ---
    validator = ForensicValidator()
    
    # Uji dengan data yang baik
    _, score_good, _, failed_good = validator.validate_cross_algorithm(mock_results_good)
    
    # Uji dengan data yang buruk
    _, score_bad, _, failed_bad = validator.validate_cross_algorithm(mock_results_bad)

    # --- 3. VERIFICATION: Check the results ---
    assert score_good > 80, f"Good results should yield a high validation score, but got {score_good}"
    assert len(failed_good) == 0, "Good results should not have failed validations"

    assert score_bad < 50, f"Bad results should yield a low validation score, but got {score_bad}"
    assert len(failed_bad) > 0, "Bad results should have failed validations"
    
    # Uji kasus edge: data kosong
    _, score_empty, _, _ = validator.validate_cross_algorithm({})
    assert score_empty == 0.0, "Empty analysis should result in a score of 0.0"
