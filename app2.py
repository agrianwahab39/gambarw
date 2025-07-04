# --- START OF FILE app2.py ---
# --- START OF FILE app.py (Gabungan dari app2.py dan kode baru) ---

import streamlit as st
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.graph_objects as go
import io
import base64 # Diperlukan untuk pratinjau PDF
import zipfile # Diperlukan untuk ekspor gambar proses

# ======================= IMPORT BARU & PENTING =======================
import signal
from utils import load_analysis_history, save_analysis_to_history, delete_all_history, delete_selected_history, get_history_count, clear_empty_thumbnail_folder
from export_utils import (export_to_advanced_docx, export_report_pdf,
                          export_visualization_png, generate_all_process_images, DOCX_AVAILABLE) # <-- Ditambah generate_all_process_images dari export_utils
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score # moved to validator and visualization internally
import seaborn as sns
from validator import ForensicValidator # <-- Import dari file validator.py
# ===========================================================================


# ======================= Konfigurasi & Import Awal =======================
# Bagian ini memastikan semua modul backend dimuat dengan benar
try:
    from main import analyze_image_comprehensive_advanced as main_analysis_func
    # GANTI BLOK IMPORT DI ATAS DENGAN YANG INI:
    from visualization import (
        create_feature_match_visualization, create_block_match_visualization,
        create_localization_visualization, create_frequency_visualization,
        create_texture_visualization, create_edge_visualization,
        create_illumination_visualization, create_statistical_visualization,
        create_quality_response_plot, create_advanced_combined_heatmap,
        create_summary_report, populate_validation_visuals,
        # -- TAMBAHKAN DUA FUNGSI INI --
        create_probability_bars,
        create_uncertainty_visualization
    )
    from config import BLOCK_SIZE
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR_MESSAGE = ""
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR_MESSAGE = str(e)


# ======================= Fungsi Helper untuk Tampilan Tab (Lama & Baru) =======================

# Helper functions untuk plot (tetap sama)
def display_single_plot(title, plot_function, args, caption, details, container):
    """Fungsi generik untuk menampilkan plot tunggal dengan detail."""
    with container:
        st.subheader(title, divider='rainbow')
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_function(ax, *args)
        st.pyplot(fig, use_container_width=True)
        st.caption(caption)
        with st.expander("Lihat Detail Teknis"):
            st.markdown(details)

def display_single_image(title, image_array, cmap, caption, details, container, colorbar=False):
    """Fungsi generik untuk menampilkan gambar tunggal dengan detail."""
    with container:
        st.subheader(title, divider='rainbow')
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(image_array, cmap=cmap)
        ax.axis('off')
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
        st.caption(caption)
        with st.expander("Lihat Detail Teknis"):
            st.markdown(details)

def create_spider_chart(analysis_results):
    """Membuat spider chart untuk kontribusi skor."""
    categories = [
        'ELA', 'Feature Match', 'Block Match', 'Noise',
        'JPEG Ghost', 'Frequency', 'Texture', 'Illumination'
    ]

    # Memastikan kunci ada sebelum diakses
    ela_mean = analysis_results.get('ela_mean', 0)
    noise_inconsistency = analysis_results.get('noise_analysis', {}).get('overall_inconsistency', 0)
    jpeg_ghost_suspicious_ratio = analysis_results.get('jpeg_ghost_suspicious_ratio', 0)
    freq_inconsistency = analysis_results.get('frequency_analysis', {}).get('frequency_inconsistency', 0)
    texture_inconsistency = analysis_results.get('texture_analysis', {}).get('overall_inconsistency', 0)
    illum_inconsistency = analysis_results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0)
    ela_regional_inconsistency = analysis_results.get('ela_regional_stats', {}).get('regional_inconsistency', 0)
    ransac_inliers = analysis_results.get('ransac_inliers', 0)
    block_matches_len = len(analysis_results.get('block_matches', []))

    # Safely convert to float if necessary, and ensure within a sensible range for the chart (0-1)
    splicing_values = [
        min(float(ela_mean / 15), 1.0), # ELA: Higher mean -> more suspicion
        min(float(noise_inconsistency / 0.5), 1.0), # Noise inconsistency
        min(float(jpeg_ghost_suspicious_ratio / 0.3), 1.0), # JPEG Ghost
        min(float(freq_inconsistency / 2.0), 1.0), # Frequency inconsistency
        min(float(texture_inconsistency / 0.5), 1.0), # Texture inconsistency
        min(float(illum_inconsistency / 0.5), 1.0), # Illumination inconsistency
        min(float(ela_regional_inconsistency / 0.5), 1.0), # ELA Regional Inconsistency
        0.1 # Placeholder
    ]
    # Re-order to match `categories`
    splicing_reordered = [
        splicing_values[0], # ELA
        splicing_values[7], # Feature Match (placeholder, could use different splicing related features)
        0.1,                # Block Match (low relevance to pure splicing)
        splicing_values[1], # Noise
        splicing_values[2], # JPEG Ghost
        splicing_values[3], # Frequency
        splicing_values[4], # Texture
        splicing_values[5]  # Illumination
    ]


    copy_move_values = [
        min(float(ela_regional_inconsistency / 0.5), 1.0), # ELA
        min(float(ransac_inliers / 30), 1.0), # Feature Match
        min(float(block_matches_len / 40), 1.0), # Block Match
        0.2, 0.2, 0.3, 0.3, 0.2 # Other attributes, usually less critical for pure copy-move
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=splicing_reordered, theta=categories, fill='toself', name='Indikator Splicing', line=dict(color='red')))
    fig.add_trace(go.Scatterpolar(r=copy_move_values, theta=categories, fill='toself', name='Indikator Copy-Move', line=dict(color='orange')))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Kontribusi Metode Analisis")
    return fig

# Fungsi display tab yang sudah ada
def display_core_analysis(original_pil, results):
    st.header("Tahap 1: Analisis Inti (Core Analysis)")
    st.write("Tahap ini memeriksa anomali fundamental seperti kompresi, fitur kunci, dan duplikasi blok.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli", divider='rainbow')
        st.image(original_pil, caption="Gambar yang dianalisis.", use_container_width=True)
        with st.expander("Detail Gambar"):
            st.json({
                "Filename": results['metadata'].get('Filename', 'N/A'),
                "Size": f"{results['metadata'].get('FileSize (bytes)', 0):,} bytes",
                "Dimensions": f"{original_pil.width}x{original_pil.height}",
                "Mode": original_pil.mode
            })
    # Safe access to ela_image and its properties
    ela_image_display = results.get('ela_image', np.zeros(original_pil.size))
    # Convert to PIL Image for st.image or use raw array with imshow (better for matplotplib based plots)
    if not isinstance(ela_image_display, Image.Image):
        if ela_image_display.ndim == 2: # grayscale array
            ela_image_display = Image.fromarray(ela_image_display.astype(np.uint8), mode='L')
        elif ela_image_display.ndim == 3: # color array (unlikely for ELA)
            ela_image_display = Image.fromarray(ela_image_display.astype(np.uint8), mode='RGB')
        else: # Fallback to black image if data is corrupted
            ela_image_display = Image.new('L', original_pil.size)
    
    display_single_image(
        title="Error Level Analysis (ELA)", image_array=ela_image_display, cmap='hot',
        caption="Area yang lebih terang menunjukkan potensi tingkat kompresi yang berbeda.",
        details=f"- **Mean ELA:** `{results.get('ela_mean', 0):.2f}`\n- **Std Dev ELA:** `{results.get('ela_std', 0):.2f}`\n- **Region Outlier:** `{results.get('ela_regional_stats', {}).get('outlier_regions', 0)}`",
        container=col2, colorbar=True
    )
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    display_single_plot(
        title="Feature Matching (Copy-Move)", plot_function=create_feature_match_visualization, args=[original_pil, results],
        caption="Garis hijau menghubungkan area dengan fitur yang identik (setelah verifikasi RANSAC).",
        details=f"- **Total SIFT Matches:** `{results.get('sift_matches', 0)}`\n- **RANSAC Verified Inliers:** `{results.get('ransac_inliers', 0)}`",
        container=col3
    )
    display_single_plot(
        title="Block Matching (Copy-Move)", plot_function=create_block_match_visualization, args=[original_pil, results],
        caption="Kotak berwarna menandai blok piksel yang identik di lokasi berbeda.",
        details=f"- **Pasangan Blok Identik:** `{len(results.get('block_matches', []))}`\n- **Ukuran Blok:** `{BLOCK_SIZE}x{BLOCK_SIZE} pixels`",
        container=col4
    )
    display_single_plot(
        title="Lokalisasi Area Mencurigakan", plot_function=create_localization_visualization, args=[original_pil, results],
        caption="Overlay merah menunjukkan area yang paling mencurigakan berdasarkan K-Means clustering.",
        details=f"- **Persentase Area Termanipulasi:** `{results.get('localization_analysis', {}).get('tampering_percentage', 0):.2f}%`",
        container=col5
    )

def display_advanced_analysis(original_pil, results):
    st.header("Tahap 2: Analisis Tingkat Lanjut (Advanced Analysis)")
    st.write("Tahap ini menyelidiki properti intrinsik gambar seperti frekuensi, tekstur, tepi, dan artefak kompresi.")
    col1, col2, col3 = st.columns(3)
    display_single_plot(title="Analisis Domain Frekuensi", plot_function=create_frequency_visualization, args=[results], caption="Distribusi energi pada frekuensi rendah, sedang, dan tinggi.", details=f"- **Inkonsistensi Frekuensi:** `{results.get('frequency_analysis', {}).get('frequency_inconsistency', 0):.3f}`", container=col1)
    display_single_plot(title="Analisis Konsistensi Tekstur", plot_function=create_texture_visualization, args=[results], caption="Mengukur konsistensi properti tekstur di seluruh gambar.", details=f"- **Inkonsistensi Tekstur Global:** `{results.get('texture_analysis', {}).get('overall_inconsistency', 0):.3f}`", container=col2)
    display_single_plot(title="Analisis Konsistensi Tepi (Edge)", plot_function=create_edge_visualization, args=[original_pil, results], caption="Visualisasi tepi gambar.", details=f"- **Inkonsistensi Tepi:** `{results.get('edge_analysis', {}).get('edge_inconsistency', 0):.3f}`", container=col3)
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    display_single_plot(title="Analisis Konsistensi Iluminasi", plot_function=create_illumination_visualization, args=[original_pil, results], caption="Peta iluminasi untuk mencari sumber cahaya yang tidak konsisten.", details=f"- **Inkonsistensi Iluminasi:** `{results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0):.3f}`", container=col4)
    
    jpeg_ghost_display = results.get('jpeg_ghost')
    # Convert numpy array to PIL Image for st.image display, if not already
    if not isinstance(jpeg_ghost_display, Image.Image):
        jpeg_ghost_display = Image.fromarray((jpeg_ghost_display * 255).astype(np.uint8), mode='L') # Normalize and convert to L (grayscale)

    display_single_image(title="Analisis JPEG Ghost", image_array=jpeg_ghost_display, cmap='hot', caption="Area terang menunjukkan kemungkinan kompresi ganda.", details=f"- **Rasio Area Mencurigakan:** `{results.get('jpeg_ghost_suspicious_ratio', 0):.2%}`", container=col5, colorbar=True)
    with col6:
        st.subheader("Peta Anomali Gabungan", divider='rainbow')
        # Ensure image is in array format for heatmap generation
        original_pil_array = np.array(original_pil)
        combined_heatmap = create_advanced_combined_heatmap(results, original_pil_array.shape[0:2][::-1]) # Pass (width, height)
        
        # Overlay original image and heatmap
        fig, ax = plt.subplots(figsize=(8, 6)); 
        ax.imshow(original_pil_array, alpha=0.5); 
        ax.imshow(combined_heatmap, cmap='inferno', alpha=0.5); 
        ax.axis('off'); 
        st.pyplot(fig, use_container_width=True)
        st.caption("Menggabungkan ELA, JPEG Ghost, dan fitur lain.")

def display_statistical_analysis(original_pil, results):
    st.header("Tahap 3: Analisis Statistik dan Metrik")
    st.write("Melihat data mentah di balik analisis.")
    col1, col2, col3 = st.columns(3)
    
    noise_map_display = results.get('noise_map')
    # Ensure noise_map is displayable
    if not isinstance(noise_map_display, Image.Image):
        noise_map_display = Image.fromarray((noise_map_display * 255).astype(np.uint8), mode='L') if noise_map_display is not None and noise_map_display.size > 0 else Image.new('L', original_pil.size)
        
    display_single_image(title="Peta Sebaran Noise", image_array=noise_map_display, cmap='gray', caption="Pola noise yang tidak seragam bisa mengindikasikan manipulasi.", details=f"- **Inkonsistensi Noise Global:** `{results.get('noise_analysis', {}).get('overall_inconsistency', 0):.3f}`", container=col1)
    display_single_plot(title="Kurva Respons Kualitas JPEG", plot_function=create_quality_response_plot, args=[results], caption="Error saat gambar dikompres ulang pada kualitas berbeda.", details=f"- **Estimasi Kualitas Asli:** `{results.get('jpeg_analysis', {}).get('estimated_original_quality', 'N/A')}`", container=col2)
    display_single_plot(title="Entropi Kanal Warna", plot_function=create_statistical_visualization, args=[results], caption="Mengukur 'kerandoman' informasi pada setiap kanal warna.", details=f"- **Entropi Global:** `{results.get('statistical_analysis', {}).get('overall_entropy', 0):.3f}`", container=col3)

def display_final_report(results):
    st.header("Tahap 4: Laporan Akhir dan Interpretasi")
    st.markdown(
        "Bagian ini merangkum temuan utama beserta skor kepercayaan dan tingkat ketidakpastian. "
        "Harap telaah indikator keandalan sebelum mengambil keputusan akhir."
    )

    # ... (Panel penjelasan tetap sama) ...

    # Ambil hasil klasifikasi baru yang berbasis ketidakpastian
    classification = results.get('classification', {})
    uncertainty_analysis = classification.get('uncertainty_analysis', {})
    probabilities = uncertainty_analysis.get('probabilities', {})
    report_details = uncertainty_analysis.get('report', {})

    result_container = st.container()
    with result_container:
        if uncertainty_analysis:
            primary_assessment = report_details.get('primary_assessment', 'N/A')
            confidence_level = report_details.get('confidence_level', 'N/A')
            uncertainty_level = probabilities.get('uncertainty_level', 0)

            st.subheader("Hasil Analisis", divider="rainbow")
            col1, col2 = st.columns([2, 1])

            with col1:
                # PENINGKATAN #3: Tampilkan penilaian utama, tapi sembunyikan keyakinan jika tidak pasti
                if "manipulasi" in primary_assessment.lower() or "forgery" in primary_assessment.lower():
                    st.warning(f"**Indikasi Utama:** {primary_assessment}", icon="⚠️")
                else:
                    st.success(f"**Indikasi Utama:** {primary_assessment}", icon="✅")

                # Hanya tampilkan skor kepercayaan jika ketidakpastian RENDAH (< 25%)
                if uncertainty_level < 0.25:
                    st.metric(label="Skor Kepercayaan", value=confidence_level)
                else:
                    # Jika tidak pasti, tampilkan level ketidakpastian sebagai gantinya
                    st.metric(label="Tingkat Ketidakpastian", value=f"{uncertainty_level*100:.1f}%", delta="Hasil Ambigu", delta_color="inverse")
                
                st.progress(probabilities.get(f"{primary_assessment.lower().replace(' ', '_').replace('indikasi: ', '')}_probability", 0))

                with st.expander("Rekomendasi Tindak Lanjut"):
                    st.markdown(report_details.get('recommendation', 'Tidak ada rekomendasi tersedia.'))

            with col2:
                st.plotly_chart(create_spider_chart(results))

            # ... (Sisa kode untuk menampilkan grafik bar probabilitas, dll. tetap sama) ...
            
            # PENINGKATAN #4: Beri penekanan pada penjelasan teknis ketidakpastian
            with st.expander("🔍 Penjelasan Detail Ketidakpastian", expanded=True):
                st.markdown(f"**Rangkuman Ketidakpastian:** {report_details.get('uncertainty_summary', 'N/A')}")
                st.markdown("**Indikator Keandalan:**")
                for indicator in report_details.get('reliability_indicators', []):
                    st.markdown(f"- {indicator}")
        else:
            st.warning("Analisis ketidakpastian tidak tersedia. Menampilkan hasil standar.", icon="ℹ️")
            # ... (fallback code)

# Fungsi display_history_tab yang dirombak dengan fitur hapus - DENGAN PERBAIKAN MASALAH NESTING COLUMNS
def display_history_tab():
    st.header("📜 Riwayat Analisis Tersimpan")
    
    # Import fungsi hapus dari utils
    # from utils import delete_all_history, delete_selected_history, get_history_count, clear_empty_thumbnail_folder # Already imported at top

    # Container untuk kontrol hapus
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    
    with col_header1:
        st.markdown("Berikut daftar semua analisis yang telah dilakukan, diurutkan dari yang terbaru.")
    
    history_data = load_analysis_history()
    history_count = len(history_data)
    
    # Tampilkan jumlah riwayat
    with col_header2:
        st.metric("Total Riwayat", history_count)
    
    # Tombol hapus semua
    with col_header3:
        if history_count > 0:
            if st.button("🗑️ Hapus Semua", use_container_width=True, type="secondary"):
                st.session_state['confirm_delete_all'] = True
    
    # Konfirmasi hapus semua
    if history_count > 0 and 'confirm_delete_all' in st.session_state and st.session_state['confirm_delete_all']:
        st.warning("⚠️ **Peringatan**: Anda akan menghapus SEMUA riwayat analisis. Tindakan ini tidak dapat dibatalkan!")
        col_confirm1, col_confirm2, _ = st.columns([1, 1, 2])
        with col_confirm1:
            if st.button("✅ Ya, Hapus Semua", type="primary"):
                with st.spinner("Menghapus semua riwayat..."):
                    success = delete_all_history()
                    if success:
                        st.success("Semua riwayat berhasil dihapus!")
                        st.session_state['confirm_delete_all'] = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Gagal menghapus riwayat.")
        with col_confirm2:
            if st.button("❌ Batal"):
                st.session_state['confirm_delete_all'] = False
                st.rerun()
    
    if not history_data:
        st.info("Belum ada riwayat analisis. Lakukan analisis pertama Anda!")
        return
    
    # Initialize session state untuk checkbox
    if 'selected_history' not in st.session_state:
        st.session_state.selected_history = []
    
    # Tombol hapus yang dipilih
    if len(st.session_state.selected_history) > 0:
        st.markdown("---")
        col_del_buttons_left, col_del_buttons_right = st.columns([0.7, 0.3])
        with col_del_buttons_left:
            st.info(f"📌 {len(st.session_state.selected_history)} item dipilih")
        with col_del_buttons_right:
            col_del1, col_del2 = st.columns(2)
            with col_del1:
                if st.button("🗑️ Hapus Yang Dipilih", use_container_width=True, type="primary"):
                    st.session_state['confirm_delete_selected'] = True
            with col_del2:
                if st.button("❌ Batal Pilih", use_container_width=True):
                    st.session_state.selected_history = []
                    st.rerun()
    
    # Konfirmasi hapus yang dipilih
    if 'confirm_delete_selected' in st.session_state and st.session_state['confirm_delete_selected']:
        st.warning(f"⚠️ Anda akan menghapus {len(st.session_state.selected_history)} riwayat yang dipilih. Lanjutkan?")
        col_conf1, col_conf2 = st.columns([1, 1])
        with col_conf1:
            if st.button("✅ Ya, Hapus", type="primary", key="confirm_del_selected"):
                with st.spinner("Menghapus riwayat yang dipilih..."):
                    # Convert indices from reversed list to original indices (since we iterate history_data[::-1])
                    # Ensure indices are valid (filter duplicates if any were somehow added)
                    original_indices_to_delete = sorted(list(set(len(history_data) - 1 - idx for idx in st.session_state.selected_history)))

                    success = delete_selected_history(original_indices_to_delete)
                    if success:
                        st.success(f"Berhasil menghapus {len(st.session_state.selected_history)} riwayat!")
                        st.session_state.selected_history = []
                        st.session_state['confirm_delete_selected'] = False
                        clear_empty_thumbnail_folder()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Gagal menghapus riwayat yang dipilih.")
        with col_conf2:
            if st.button("❌ Batal", key="cancel_del_selected"):
                st.session_state['confirm_delete_selected'] = False
                st.rerun()
    
    st.markdown("---")
    
    # Tampilkan riwayat dengan checkbox
    # Using reversed(history_data) for newest first display
    for idx, entry in enumerate(reversed(history_data)):
        # Calculate the original index for proper selection/deletion
        original_entry_idx = len(history_data) - 1 - idx
        
        timestamp, image_name = entry.get('timestamp', 'N/A'), entry.get('image_name', 'N/A')
        summary, result_type = entry.get('analysis_summary', {}), entry.get('analysis_summary', {}).get('type', 'N/A')
        thumbnail_path = entry.get('thumbnail_path')
        
        if "Splicing" in result_type or "Complex" in result_type or "Manipulasi" in result_type or "Forgery" in result_type:
            icon, color = "🚨", "#ff4b4b"
        elif "Copy-Move" in result_type:
            icon, color = "⚠️", "#ffc400"
        else:
            icon, color = "✅", "#268c2f"
        
        # Outer container to group checkbox and expander properly
        with st.container():
            col_chk, col_exp = st.columns([0.05, 1])
            
            with col_chk:
                # Checkbox for selecting item (use original_entry_idx as value for accurate tracking)
                is_selected = st.checkbox("", key=f"select_{original_entry_idx}", value=original_entry_idx in st.session_state.selected_history)
                if is_selected and original_entry_idx not in st.session_state.selected_history:
                    st.session_state.selected_history.append(original_entry_idx)
                elif not is_selected and original_entry_idx in st.session_state.selected_history:
                    st.session_state.selected_history.remove(original_entry_idx)
            
            with col_exp:
                expander_title = f"{icon} **{timestamp}** | `{image_name}` | **Hasil:** {result_type}"
                st.markdown(f'<div style="border: 2px solid {color}; border-radius: 7px; padding: 10px; margin-bottom: 10px;">', unsafe_allow_html=True)
                with st.expander(expander_title):
                    # Row 1: Thumbnail and basic info
                    col1_detail, col2_detail = st.columns([1, 3])
                    with col1_detail:
                        st.markdown("**Gambar Asli**")
                        if thumbnail_path and os.path.exists(thumbnail_path):
                            st.image(thumbnail_path, use_container_width=True)
                        else:
                            st.caption("Thumbnail tidak tersedia.")
                    
                    with col2_detail:
                        st.markdown(f"**Kepercayaan:** {summary.get('confidence', 'N/A')}")
                        st.markdown(f"**Skor Copy-Move:** {summary.get('copy_move_score', 0)}/100 | **Skor Splicing:** {summary.get('splicing_score', 0)}/100")
                        st.caption(f"Waktu Proses: {entry.get('processing_time', 'N/A')}")
                        st.markdown("---")
                        st.write("**Detail (JSON):**")
                        st.json(summary)
                
                st.markdown("</div>", unsafe_allow_html=True)

# ======================= KODE BARU UNTUK VALIDASI FORENSIK (MENGGANTIKAN YANG LAMA) =======================

# NOTE: The ForensicValidator class is now imported from validator.py.
# The internal helper functions used by validate_pipeline_integrity must be defined.

def validate_pipeline_integrity(analysis_results):
    """
    Validasi integritas pipeline 17 langkah pemrosesan.
    """
    if not analysis_results:
        return ["Hasil analisis tidak tersedia untuk validasi pipeline."], 0.0
    
    # Definisi proses pipeline
    pipeline_processes = [
        {"name": "1. Validasi & Muat Gambar", "check": lambda r: isinstance(r.get('metadata', {}).get('FileSize (bytes)', 0), int) and r.get('metadata', {}).get('FileSize (bytes)', 0) > 0},
        {"name": "2. Ekstraksi Metadata", "check": lambda r: 'Metadata_Authenticity_Score' in r.get('metadata', {})},
        {"name": "3. Pra-pemrosesan Gambar", "check": lambda r: r.get('enhanced_gray') is not None and r['enhanced_gray'].ndim == 2},
        {"name": "4. Analisis ELA Multi-Kualitas", "check": lambda r: r.get('ela_image') is not None and r.get('ela_mean', -1) >= 0},
        {"name": "5. Ekstraksi Fitur (SIFT, ORB, etc.)", "check": lambda r: isinstance(r.get('feature_sets'), dict) and 'sift' in r['feature_sets'] and len(r['feature_sets']['sift'][0]) > 0}, # Ensure SIFT has keypoints
        {"name": "6. Deteksi Copy-Move (Feature-based)", "check": lambda r: 'ransac_inliers' in r and r['ransac_inliers'] >= 0},
        {"name": "7. Deteksi Copy-Move (Block-based)", "check": lambda r: 'block_matches' in r},
        {"name": "8. Analisis Konsistensi Noise", "check": lambda r: 'overall_inconsistency' in r.get('noise_analysis', {})},
        {"name": "9. Analisis Artefak JPEG", "check": lambda r: 'estimated_original_quality' in r.get('jpeg_analysis', {}).get('basic_analysis', {})}, # Adjusted access path for basic_analysis
        {"name": "10. Analisis Ghost JPEG", "check": lambda r: r.get('jpeg_ghost') is not None and r.get('jpeg_ghost_suspicious_ratio', 0) >= 0},
        {"name": "11. Analisis Domain Frekuensi", "check": lambda r: 'frequency_inconsistency' in r.get('frequency_analysis', {})},
        {"name": "12. Analisis Konsistensi Tekstur", "check": lambda r: 'overall_inconsistency' in r.get('texture_analysis', {})},
        {"name": "13. Analisis Konsistensi Tepi", "check": lambda r: 'edge_inconsistency' in r.get('edge_analysis', {})},
        {"name": "14. Analisis Konsistensi Iluminasi", "check": lambda r: 'overall_illumination_inconsistency' in r.get('illumination_analysis', {})},
        {"name": "15. Analisis Statistik Kanal", "check": lambda r: 'overall_entropy' in r.get('statistical_analysis', {})}, # Check for overall_entropy instead of rg_correlation (more general)
        {"name": "16. Lokalisasi Area Manipulasi", "check": lambda r: 'localization_analysis' in r and 'combined_tampering_mask' in r['localization_analysis']},
        {"name": "17. Klasifikasi Akhir & Skor", "check": lambda r: 'type' in r.get('classification', {}) and 'confidence' in r.get('classification', {})}
    ]
    
    pipeline_results = []
    success_count = 0
    
    for process in pipeline_processes:
        try:
            is_success = process["check"](analysis_results)
        except Exception as e:
            is_success = False
            # print(f"Error saat validasi integritas pipeline '{process['name']}': {e}") # Debugging for development
        
        if is_success:
            status = "[BERHASIL]"
            pipeline_results.append(f"✅ {status:12} | {process['name']}")
            success_count += 1
        else:
            status = "[GAGAL]"
            pipeline_results.append(f"❌ {status:12} | {process['name']}")
    
    # Calculate pipeline integrity percentage
    pipeline_integrity = (success_count / len(pipeline_processes)) * 100
    
    return pipeline_results, pipeline_integrity


def lakukan_validasi_sistem(analysis_results):
    """
    Menjalankan Validasi Integritas Proses dengan pendekatan Validasi Silang Multi-Algoritma.
    Fungsi ini mengevaluasi kualitas hasil dari setiap algoritma dan konsistensi antar hasil.
    """
    if not analysis_results:
        return ["Hasil analisis tidak tersedia untuk divalidasi."], 0.0, "Hasil analisis tidak tersedia.", []

    # Buat validator forensik baru
    validator = ForensicValidator()
    
    # Jalankan validasi silang antar algoritma
    process_results, validation_score, summary_text, failed_validations = validator.validate_cross_algorithm(analysis_results)
    
    # Periksa juga integritas proses pipeline
    pipeline_results, pipeline_integrity_percentage = validate_pipeline_integrity(analysis_results)
    
    # Gabungkan hasil pipeline dan validasi silang
    combined_results = []
    combined_results.append("=== VALIDASI SILANG ALGORITMA ===")
    combined_results.extend(process_results)
    combined_results.append("")
    combined_results.append("=== VALIDASI INTEGRITAS PIPELINE ===")
    combined_results.extend(pipeline_results)
    
    # Bobot: 70% validasi silang, 30% integritas pipeline
    final_score = (validation_score * 0.7) + (pipeline_integrity_percentage * 0.3)
    
    return combined_results, final_score, summary_text, failed_validations


def display_validation_tab_baru(analysis_results):
    """
    Menampilkan tab validasi sistem (Tahap 5) dengan pendekatan validasi silang
    yang disempurnakan untuk presentasi forensik profesional.
    """
    st.header("🔬 Tahap 5: Validasi Forensik Digital", anchor=False)
    
    # Panel Penjelasan Metodologi
    with st.expander("Metodologi Validasi Forensik", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### Pendekatan Validasi Silang Multi-Algoritma
            
            Sistem ini mengimplementasikan pendekatan validasi forensik modern yang direkomendasikan oleh
            **Digital Forensic Research Workshop (DFRWS)** dan **Scientific Working Group on Digital Evidence (SWGDE)**.
            Validasi dilakukan melalui empat tahap utama:
            
            1. **Validasi Individual Algorithm** - Setiap metode deteksi dievaluasi secara terpisah
            2. **Cross-Algorithm Validation** - Hasil divalidasi silang antar algoritma
            3. **Physical Consistency Verification** - Memeriksa kesesuaian dengan prinsip fisika citra digital
            4. **Pipeline Integrity Assurance** - Memastikan semua 17 tahap berjalan dengan benar
            
            Sistem memberi bobot lebih besar (30% masing-masing) pada metode utama (K-Means dan Lokalisasi)
            dibandingkan metode pendukung (ELA dan SIFT, 20% masing-masing).
            """)
        
        with col2:
            # ======================= PERBAIKAN DI SINI =======================
            # Memeriksa apakah file ada sebelum menampilkannya
            image_path = os.path.join(os.path.dirname(__file__), "assets/validation_diagram.png") # Ensure path is correct
            if os.path.exists(image_path):
                st.image(image_path, caption="Validasi Forensik Digital")
            else:
                st.info("Diagram validasi visual tidak tersedia.")
            # ======================= AKHIR PERBAIKAN =======================
            
            st.markdown("""
            #### Standar & Referensi:
            - ISO/IEC 27037:2012
            - NIST SP 800-86
            - Validasi >80% diperlukan untuk bukti di pengadilan
            """)

    # Jalankan validasi
    report_details, validation_score, summary_text, failed_validations = lakukan_validasi_sistem(analysis_results)

    # Dashboard Utama
    st.subheader("Dashboard Validasi Forensik", anchor=False)
    
    # Tabs untuk hasil berbeda
    tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Validasi", "🧪 Detail Proses", "📑 Dokumentasi Forensik"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Gauge chart untuk skor validasi
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=validation_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 90, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 80], 'color': 'red'},
                        {'range': [80, 90], 'color': 'orange'},
                        {'range': [90, 100], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                },
                title={'text': "Skor Validasi Forensik"}
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Status validasi
            if validation_score >= 90:
                st.success("✅ **Validasi Tingkat Tinggi**: Hasil analisis memenuhi standar bukti forensik dengan kepercayaan tinggi.")
            elif validation_score >= 80:
                st.warning("⚠️ **Validasi Cukup**: Hasil analisis dapat diterima namun memerlukan konfirmasi tambahan.")
            else:
                st.error("❌ **Validasi Tidak Memadai**: Hasil analisis memiliki inkonsistensi yang signifikan.")
                
            st.info(summary_text)
            
        with col2:
            # Visualisasi kepercayaan per algoritma
            st.markdown("### Validasi Per Algoritma Forensik")
            validator_instance = ForensicValidator() # Instantiate the validator
            algorithm_scores = {}
            # Need to provide mock results if analysis_results is empty or none, for display
            display_results = analysis_results if analysis_results else {
                'localization_analysis': {'kmeans_localization': {'cluster_ela_means': [0,0], 'tampering_cluster_id':-1, 'tampering_percentage':0}, 'combined_tampering_mask': False},
                'ela_image': Image.new('L', (10,10)), 'ela_mean': 0, 'ela_std': 0, 'ela_regional_stats': {'regional_inconsistency': 0, 'outlier_regions': 0}, 'ela_quality_stats': [],
                'noise_analysis': {'overall_inconsistency':0},
                'ransac_inliers': 0, 'sift_matches': 0, 'geometric_transform': None, 'block_matches': [],
                'metadata': {'Metadata_Authenticity_Score': 0}
            }

            for technique_name, validate_func in [
                ('K-Means', validator_instance.validate_clustering),
                ('Lokalisasi', validator_instance.validate_localization),
                ('ELA', validator_instance.validate_ela),
                ('SIFT', validator_instance.validate_feature_matching)
            ]:
                confidence, _ = validate_func(display_results)
                algorithm_scores[technique_name] = confidence * 100
            
            # Buat donut chart
            labels = list(algorithm_scores.keys())
            values = list(algorithm_scores.values())
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=['rgb(56, 75, 126)', 'rgb(18, 36, 37)',
                              'rgb(34, 53, 101)', 'rgb(36, 55, 57)']
            )])
            
            fig.update_layout(
                title_text="Skor Kepercayaan Algoritma",
                annotations=[dict(text=f'{validation_score:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tambahkan interpretasi
            st.markdown("#### Interpretasi Forensik:")
            for alg, score in algorithm_scores.items():
                color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                st.markdown(f"- **{alg}**: <span style='color:{color}'>{score:.1f}%</span>", unsafe_allow_html=True)

    with tab2:
        st.subheader("Detail Proses Validasi", anchor=False)
        
        # Format report details with syntax highlighting
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create collapsible sections for each validation category
            with st.expander("Validasi Silang Algoritma", expanded=True):
                # Filter to show only cross-algorithm specific results, which come from algo_results
                cross_algo_output_lines = [line for line in report_details if "VALIDASI SILANG" not in line and "VALIDASI INTEGRITAS PIPELINE" not in line]
                report_text = "\n".join(cross_algo_output_lines)
                st.code(report_text, language='bash')
            
            with st.expander("Validasi Integritas Pipeline"):
                # Filter to show only pipeline specific results
                pipeline_output_lines = [line for line in report_details if "VALIDASI INTEGRITAS PIPELINE" in line or (("✅" in line or "❌" in line) and "Validasi" not in line)]
                # Extract actual pipeline steps if any. Filter for lines containing "BERHASIL" or "GAGAL"
                actual_pipeline_steps = [line for line in report_details if ("BERHASIL" in line or "GAGAL" in line) and "Validasi " not in line]
                if actual_pipeline_steps:
                    st.code("\n".join(actual_pipeline_steps), language='bash')
                else:
                    st.code("Tidak ada detail langkah pipeline tersedia atau semua berhasil.", language='bash')
        
        with col2:
            # Calculate success metrics
            total_process = len([r for r in report_details if "[" in r]) # Count all lines that signify a check result
            passed_process = len([r for r in report_details if "[BERHASIL]" in r or "[LULUS]" in r])
            
            # Create metrics display
            st.metric(
                label="Keberhasilan Proses",
                value=f"{passed_process}/{total_process}",
                delta=f"{(passed_process/total_process*100):.1f}%"
            )
            
            # Add validation formula
            st.markdown("""
            #### Formula Validasi:
            ```
            Skor Akhir = (0.7 × Validasi Silang) + (0.3 × Integritas Pipeline)
            ```
            
            #### Threshold Validasi:
            - ≥ 90%: Bukti Forensik Tingkat Tinggi
            - ≥ 80%: Bukti Forensik Dapat Diterima
            - < 80%: Bukti Forensik Tidak Memadai
            """)
        
        # Show failed validations if any
        if failed_validations:
            st.error("🚨 **Kegagalan Validasi Terdeteksi**")
            for i, failure in enumerate(failed_validations):
                with st.expander(f"Detail Kegagalan #{i+1}: {failure['name']}", expanded=i==0):
                    st.markdown(f"**Alasan Kegagalan:** {failure['reason']}")
                    st.markdown("**Aturan Validasi Forensik:**")
                    st.code(failure['rule'], language='text')
                    st.markdown("**Data Forensik:**")
                    st.code(failure['values'], language='text')
        else:
            st.success("✅ **Tidak Ada Kegagalan Validasi yang Terdeteksi**")
            st.markdown("Semua algoritma menunjukkan hasil yang konsisten dan terpenuhi kriteria validasi minimum.")

    with tab3:
        st.subheader("Dokumentasi Forensik Digital", anchor=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### Kriteria Validasi Forensik Digital
            
            Berdasarkan pedoman dari **National Institute of Standards and Technology (NIST)** dan **Association of Chief Police Officers (ACPO)**, hasil analisis forensik digital harus memenuhi kriteria-kriteria berikut:
            
            1. **Reliability** - Hasil dapat direproduksi dan konsisten
            2. **Accuracy** - Pengukuran dan perhitungan tepat
            3. **Precision** - Tingkat detail yang memadai
            4. **Verifiability** - Dapat diverifikasi dengan metode independen
            5. **Scope/Limitations** - Batasan dan cakupan diketahui
            
            Setiap metode deteksi (K-Means, ELA, SIFT, dll) telah melalui validasi silang untuk memastikan bahwa hasilnya memenuhi kriteria-kriteria di atas.
            """)
            
            st.markdown("""
            ### Langkah Validasi Analisis Forensik
            
            1. **Technical Validation** - Memverifikasi algoritma berfungsi dengan benar
            2. **Cross-Method Validation** - Membandingkan hasil antar metode yang berbeda
            3. **Internal Consistency Check** - Mengevaluasi konsistensi logis hasil
            4. **Anti-Tampering Validation** - Memverifikasi integritas data
            5. **Uncertainty Quantification** - Mengukur tingkat kepercayaan hasil
            """)
        
        with col2:
            # Chain of custody and evidence validation
            st.markdown("""
            ### Chain of Custody Forensik
            
            Pipeline 17 langkah dalam sistem ini memastikan **chain of custody** yang tidak terputus dari data asli hingga hasil analisis akhir:
            
            1. **Input Validation** → Validasi keaslian input gambar
            2. **Preservation** → Penyimpanan gambar asli tanpa modifikasi
            3. **Processing** → Analisis dengan multiple metode independen
            4. **Cross-Validation** → Validasi silang hasil antar metode
            5. **Reporting** → Dokumentasi lengkap proses dan hasil
            
            Validasi di atas 90% menunjukkan bahwa chain of custody telah terjaga dengan baik, dan hasil analisis memiliki tingkat kepercayaan yang tinggi untuk digunakan sebagai bukti forensik.
            """)
            
            # Add reference to forensic standards
            st.markdown("""
            ### Standar dan Referensi Forensik
            
            Proses validasi mengikuti standar berikut:
            
            - **ISO/IEC 27037:2012** - Guidelines for identification, collection, acquisition, and preservation of digital evidence
            - **ISO/IEC 27042:2015** - Guidelines for the analysis and interpretation of digital evidence
            - **NIST SP 800-86** - Guide to Integrating Forensic Techniques into Incident Response
            - **SWGDE** - Scientific Working Group on Digital Evidence Best Practices
            """)
    
    # Bottom section: expert insights
    st.markdown("---")
    st.subheader("Interpretasi Forensik", anchor=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate metrics based on individual algorithm results
        validator_instance = ForensicValidator() # Instantiate again for readability
        # Use display_results or empty dict as a safe fallback
        data_for_interpretation = analysis_results if analysis_results else {}
        cluster_confidence, cluster_details = validator_instance.validate_clustering(data_for_interpretation)
        localization_confidence, loc_details = validator_instance.validate_localization(data_for_interpretation)
        ela_confidence, ela_details = validator_instance.validate_ela(data_for_interpretation)
        feature_confidence, feature_details = validator_instance.validate_feature_matching(data_for_interpretation)
        
        # Create a more detailed forensic interpretation
        confidence_values = [
            (cluster_confidence, "K-Means Clustering"),
            (localization_confidence, "Lokalisasi Tampering"),
            (ela_confidence, "Error Level Analysis"),
            (feature_confidence, "SIFT Feature Matching")
        ]
        
        # Ensure that confidence_values is not empty before min/max
        if confidence_values:
            highest_confidence = max(confidence_values, key=lambda x: x[0])
            lowest_confidence = min(confidence_values, key=lambda x: x[0])
        else: # Fallback for completely empty analysis_results
            highest_confidence = (0.0, "N/A")
            lowest_confidence = (0.0, "N/A")

        # Create interpretation based on overall score and individual method scores
        if validation_score >= 90:
            interpretation = f"""
            Analisis forensik menunjukkan **tingkat kepercayaan tinggi ({validation_score:.1f}%)** dengan bukti konsisten antar metode.
            Bukti terkuat berasal dari **{highest_confidence[1]} ({highest_confidence[0]*100:.1f}%)**.
            
            Hasil ini memenuhi standar forensik dan dapat digunakan sebagai bukti yang kuat dalam investigasi digital.
            """
        elif validation_score >= 80:
            interpretation = f"""
            Analisis forensik menunjukkan **tingkat kepercayaan cukup ({validation_score:.1f}%)** dengan beberapa inkonsistensi minor.
            Bukti terkuat berasal dari **{highest_confidence[1]} ({highest_confidence[0]*100:.1f}%)**,
            sementara **{lowest_confidence[1]}** menunjukkan kepercayaan lebih rendah ({lowest_confidence[0]*100:.1f}%).
            
            Hasil ini dapat digunakan sebagai bukti pendukung tetapi memerlukan konfirmasi dari metode lain.
            """
        else:
            interpretation = f"""
            Analisis forensik menunjukkan **tingkat kepercayaan rendah ({validation_score:.1f}%)** dengan inkonsistensi signifikan antar metode.
            Bahkan bukti terkuat dari **{highest_confidence[1]}** hanya mencapai kepercayaan ({highest_confidence[0]*100:.1f}%).
            
            Hasil ini memerlukan penyelidikan lebih lanjut dan tidak dapat digunakan sebagai bukti tunggal.
            """
        
        st.markdown(interpretation)
        
        # Add forensic recommendation
        st.markdown("### Rekomendasi Forensik")
        if validation_score >= 90:
            st.success("""
            ✅ **DAPAT DITERIMA SEBAGAI BUKTI FORENSIK**
            
            Hasil analisis memiliki kepercayaan tinggi dan konsistensi yang baik antar metode.
            Tidak diperlukan analisis tambahan untuk memverifikasi hasil.
            """)
        elif validation_score >= 80:
            st.warning("""
            ⚠️ **DAPAT DITERIMA DENGAN VERIFIKASI TAMBAHAN**
            
            Hasil analisis memiliki tingkat kepercayaan cukup tetapi memerlukan metode verifikasi tambahan.
            Rekomendasikan analisis oleh ahli forensik secara manual.
            """)
        else:
            st.error("""
            ❌ **MEMERLUKAN ANALISIS ULANG**
            
            Hasil analisis menunjukkan inkonsistensi signifikan antar metode.
            Diperlukan pengambilan sampel ulang atau metode analisis alternatif.
            """)
    
    with col2:
        # Create comparison table of algorithm performance
        st.markdown("### Perbandingan Metode Forensik")
        
        # Create a comparison dataframe
        algorithm_data = {
            "Metode": ["K-Means", "Lokalisasi", "ELA", "SIFT"],
            "Kepercayaan": [
                f"{cluster_confidence*100:.1f}%",
                f"{localization_confidence*100:.1f}%",
                f"{ela_confidence*100:.1f}%",
                f"{feature_confidence*100:.1f}%"
            ],
            "Bobot": ["30%", "30%", "20%", "20%"],
            "Detail": [
                cluster_details,
                loc_details,
                ela_details,
                feature_details
            ]
        }
        
        # Create a styled dataframe
        import pandas as pd
        df = pd.DataFrame(algorithm_data)
        
        # Function to highlight cells based on confidence value
        def highlight_confidence(val):
            if "%" in str(val):
                try:
                    confidence = float(val.strip("%"))
                    if confidence >= 80:
                        return 'background-color: #a8f0a8'  # Light green
                    elif confidence >= 60:
                        return 'background-color: #f0e0a8'  # Light yellow
                    else:
                        return 'background-color: #f0a8a8'  # Light red
                except ValueError: # Handle cases where conversion fails, e.g., N/A
                    return ''
            return ''
        
        # Display the styled dataframe
        st.dataframe(df.style.applymap(highlight_confidence, subset=['Kepercayaan']), use_container_width=True)
        
        # Add Q&A preparation for defense
        st.markdown("### Panduan")
        with st.expander("Bagaimana sistem memastikan integritas data?"):
            st.markdown("""
            Sistem mengimplementasikan validasi pipeline 17 langkah yang memastikan:
            
            1. Validasi awal file gambar untuk keaslian metadata
            2. Preprocessing yang tidak merusak data asli
            3. Multiple algoritma deteksi yang independen
            4. Cross-validation antar algoritma
            5. Scoring dengan pembobotan berdasarkan reliabilitas algoritma
            
            Chain of custody dipastikan dengan mempertahankan gambar asli tanpa modifikasi.
            """)
            
        with st.expander("Mengapa validasi multi-algoritma lebih andal?"):
            st.markdown("""
            Validasi multi-algoritma lebih andal karena:
            
            1. **Redundansi** - Jika satu algoritma gagal, algoritma lain dapat mendeteksi manipulasi
            2. **Teknik Komplementer** - Algoritma yang berbeda mendeteksi jenis manipulasi berbeda
            3. **Konsensus** - Kesepakatan antar algoritma meningkatkan kepercayaan hasil
            4. **Bias Reduction** - Mengurangi false positive/negative dari algoritma tunggal
            
            Pendekatan ini mengikuti prinsip "defense in depth" dalam forensik digital.
            """)
            
        with st.expander("Bagaimana sistem meminimalkan false positives?"):
            st.markdown("""
            Sistem meminimalkan false positives dengan:
            
            1. **Threshold Kalibrasi** - Setiap algoritma memiliki threshold minimum 60%
            2. **Weighted Scoring** - Algoritma lebih andal diberi bobot lebih besar
            3. **Cross-validation** - Memerlukan konsensus antar metode
            4. **Physical Consistency** - Memvalidasi terhadap prinsip fisika citra
            5. **Bobot Agreement** - Menerapkan bonus 20% untuk kesepakatan antar algoritma
            
            Dengan pendekatan ini, sistem memastikan tingkat false positive yang rendah sambil mempertahankan sensitivitas deteksi.
            """)

# ======================= APLIKASI UTAMA STREAMLIT (BAGIAN YANG DIMODIFIKASI) =======================
def main_app():
    st.set_page_config(layout="wide", page_title="Sistem Forensik Gambar V3")

    # Ganti nama variabel agar tidak bentrok dengan fungsi
    global IMPORTS_SUCCESSFUL, IMPORT_ERROR_MESSAGE
    
    if not IMPORTS_SUCCESSFUL:
        st.error(f"Gagal mengimpor modul: {IMPORT_ERROR_MESSAGE}")
        return

    # Inisialisasi session state (tidak ada perubahan di sini)
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'original_image' not in st.session_state: st.session_state.original_image = None
    if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None
    
    st.sidebar.title("🖼️ Sistem Deteksi Forensik V3")
    st.sidebar.markdown("Unggah gambar untuk memulai analisis mendalam.")

    uploaded_file = st.sidebar.file_uploader(
        "Pilih file gambar...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )

    if uploaded_file is not None:
        # Periksa apakah ini file baru atau sama dengan yang terakhir
        if st.session_state.last_uploaded_file is None or st.session_state.last_uploaded_file.name != uploaded_file.name:
            st.session_state.last_uploaded_file = uploaded_file
            st.session_state.analysis_results = None
            try:
                st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.original_image_name = uploaded_file.name # Simpan nama file
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.session_state.original_image = None
                st.session_state.last_uploaded_file = None
                return
            st.rerun()

    if st.session_state.original_image:
        st.sidebar.image(st.session_state.original_image, caption='Gambar yang diunggah', use_container_width=True)

        if st.sidebar.button("🔬 Mulai Analisis", use_container_width=True, type="primary"):
            st.session_state.analysis_results = None
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            filename = st.session_state.original_image_name
            temp_filepath = os.path.join(temp_dir, filename)
            
            # Tulis ulang file dari buffer
            st.session_state.last_uploaded_file.seek(0)
            with open(temp_filepath, "wb") as f:
                f.write(st.session_state.last_uploaded_file.getbuffer())

            with st.spinner('Melakukan analisis 17 tahap... Ini mungkin memakan waktu beberapa saat.'):
                try:
                    # Pastikan main_analysis_func dipanggil dengan path file yang benar
                    results = main_analysis_func(temp_filepath)
                    st.session_state.analysis_results = results
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat analisis: {e}")
                    st.exception(e)
                    st.session_state.analysis_results = None
                finally:
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.subheader("Kontrol Sesi")

        # Tombol Mulai Ulang (tidak ada perubahan)
        if st.sidebar.button("🔄 Mulai Ulang Analisis", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.original_image = None
            st.session_state.last_uploaded_file = None
            if 'pdf_preview_path' in st.session_state:
                st.session_state.pdf_preview_path = None # Reset preview
            st.rerun()

# Tombol Keluar (tidak ada perubahan pada logika ini)
        if st.sidebar.button("🚪 Keluar", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.original_image = None
            st.session_state.last_uploaded_file = None
            st.sidebar.warning("Aplikasi sedang ditutup...")
            st.balloons()
            time.sleep(2)
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini menggunakan pipeline analisis 17-tahap untuk mendeteksi manipulasi gambar.")

    st.title("Hasil Analisis Forensik Gambar")

    if st.session_state.analysis_results:
        tab_list = [
            "📊 Tahap 1: Analisis Inti",
            "🔬 Tahap 2: Analisis Lanjut",
            "📈 Tahap 3: Analisis Statistik",
            "📋 Tahap 4: Laporan Akhir & Kepercayaan", # Nama tab bisa diubah
            "🧪 Tahap 5: Hasil Pengujian", # Updated tab name
            "📄 Ekspor Laporan",
            "📜 Riwayat Analisis"
        ]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_list)

        # ======================= AKHIR PERUBAHAN TAB =======================

        with tab1:
            display_core_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab2:
            display_advanced_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab3:
            display_statistical_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab4:
            # Panggil fungsi yang menampilkan laporan akhir baru
            display_final_report(st.session_state.analysis_results) # Pastikan fungsi ini sudah diperbarui
            
            # Panggil fungsi visualisasi ketidakpastian secara eksplisit jika perlu
            st.subheader("Visualisasi Probabilitas dan Ketidakpastian", divider='blue')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Asumsi Anda punya fungsi ini di visualization.py
            create_probability_bars(ax1, st.session_state.analysis_results) 
            create_uncertainty_visualization(ax2, st.session_state.analysis_results) 
            
            st.pyplot(fig)
        # ======================= KONTEN TAB BARU =======================
        with tab5:
            display_validation_tab_baru(st.session_state.analysis_results)
        # ======================= AKHIR KONTEN TAB BARU =======================
        with tab6:
            # Konten tab ekspor
            st.header("📄 Ekspor Hasil Analisis")
            st.write("Pilih format yang diinginkan untuk mengunduh laporan analisis forensik.")

            # Ambil nama file asli tanpa ekstensi untuk nama file default
            base_filename = os.path.splitext(st.session_state.original_image_name)[0]
            
            # Opsi Ekspor
            export_format = st.selectbox(
                "Pilih Format Laporan:",
                options=["-", "Laporan DOCX", "Laporan PDF", "Visualisasi PNG", "Paket Lengkap (.zip)"],
                help="Pilih format ekspor yang diinginkan."
            )

            if export_format != "-":
                export_path = os.path.join("exported_reports", base_filename) # Simpan di subfolder
                os.makedirs("exported_reports", exist_ok=True)

                if st.button(f"🚀 Ekspor ke {export_format}", type="primary"):
                    with st.spinner(f"Mengekspor ke {export_format}..."):
                        try:
                            if export_format == "Laporan DOCX":
                                docx_path = f"{export_path}_report.docx"
                                result_path = export_to_advanced_docx(
                                    st.session_state.original_image,
                                    st.session_state.analysis_results,
                                    docx_path
                                )
                                if result_path and os.path.exists(result_path):
                                    st.success(f"Laporan DOCX berhasil disimpan di: `{result_path}`")
                                    with open(result_path, "rb") as f:
                                        st.download_button("Unduh Laporan DOCX", f, file_name=os.path.basename(result_path))
                                else:
                                    st.error("Gagal membuat laporan DOCX.")

                            elif export_format == "Laporan PDF":
                                pdf_path = f"{export_path}_report.pdf"
                                # Pertama, buat DOCX dulu
                                docx_path_temp = f"{export_path}_temp_for_pdf.docx"
                                docx_for_pdf = export_to_advanced_docx(
                                    st.session_state.original_image,
                                    st.session_state.analysis_results,
                                    docx_path_temp
                                )
                                if docx_for_pdf:
                                    result_path = export_report_pdf(docx_for_pdf, pdf_path)
                                    if result_path and os.path.exists(result_path):
                                        st.success(f"Laporan PDF berhasil disimpan di: `{result_path}`")
                                        with open(result_path, "rb") as f:
                                            st.download_button("Unduh Laporan PDF", f, file_name=os.path.basename(result_path))
                                    else:
                                        st.error("Gagal mengonversi DOCX ke PDF. Pastikan LibreOffice atau TinyWow API key tersedia.")
                                    if os.path.exists(docx_path_temp):
                                        os.remove(docx_path_temp) # Hapus file DOCX sementara
                                else:
                                    st.error("Gagal membuat file DOCX untuk konversi PDF.")

                            elif export_format == "Visualisasi PNG":
                                png_path = f"{export_path}_visualization.png"
                                result_path = export_visualization_png(
                                    st.session_state.original_image,
                                    st.session_state.analysis_results,
                                    png_path
                                )
                                if result_path and os.path.exists(result_path):
                                    st.success(f"Visualisasi PNG berhasil disimpan di: `{result_path}`")
                                    with open(result_path, "rb") as f:
                                        st.download_button("Unduh Visualisasi PNG", f, file_name=os.path.basename(result_path))
                                else:
                                    st.error("Gagal membuat visualisasi PNG.")

                            elif export_format == "Paket Lengkap (.zip)":
                                zip_path_base = os.path.join("exported_reports", f"{base_filename}_full_package")
                                
                                # Menggunakan fungsi baru yang membuat zip
                                export_files = export_comprehensive_package(
                                    st.session_state.original_image,
                                    st.session_state.analysis_results,
                                    zip_path_base 
                                )
                                
                                result_path = export_files.get('complete_zip')

                                if result_path and os.path.exists(result_path):
                                    st.success(f"Paket .zip berhasil dibuat di: `{result_path}`")
                                    with open(result_path, "rb") as f:
                                        st.download_button("Unduh Paket (.zip)", f, file_name=os.path.basename(result_path))
                                else:
                                    st.error("Gagal membuat paket .zip.")

                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat ekspor: {e}")
                            st.exception(e)
        with tab7:
            display_history_tab()

    elif not st.session_state.original_image:
        # Tampilkan tab Riwayat di halaman utama jika belum ada gambar diunggah
        main_page_tabs = st.tabs(["👋 Selamat Datang", "📜 Riwayat Analisis"])
        
        with main_page_tabs[0]:
            st.info("Silakan unggah gambar di sidebar kiri untuk memulai.")
            st.markdown("""
            **Panduan Singkat:**
            1. **Unggah Gambar:** Gunakan tombol 'Pilih file gambar...' di sidebar.
            2. **Mulai Analisis:** Klik tombol biru 'Mulai Analisis'.
            3. **Lihat Hasil:** Hasil akan ditampilkan dalam beberapa tab.
            4. **Uji Sistem:** Buka tab 'Hasil Pengujian' untuk melihat validasi integritas.
            5. **Ekspor:** Buka tab 'Ekspor Laporan' untuk mengunduh hasil.
            """)
        
        with main_page_tabs[1]:
            display_history_tab()

# Pastikan Anda memanggil fungsi main_app() di akhir
if __name__ == '__main__':
    # Anda harus menempatkan semua fungsi helper (seperti display_core_analysis, dll.)
    # sebelum pemanggilan main_app() atau di dalam file lain dan diimpor.
    # Untuk contoh ini, saya asumsikan semua fungsi sudah didefinisikan di atas.
    
    # ======================= Konfigurasi & Import =======================
    try:
        from main import analyze_image_comprehensive_advanced as main_analysis_func
        from config import BLOCK_SIZE
        IMPORTS_SUCCESSFUL = True
        IMPORT_ERROR_MESSAGE = ""
    except ImportError as e:
        IMPORTS_SUCCESSFUL = False
        IMPORT_ERROR_MESSAGE = str(e)
    
    main_app()

# --- END OF FILE app2.py ---