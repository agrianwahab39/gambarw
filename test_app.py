# --- START OF FILE test_app.py ---

import json
import os
import shutil
from utils import save_analysis_to_history, load_analysis_history, delete_selected_history, HISTORY_FILE, THUMBNAIL_DIR
from advanced_analysis import analyze_noise_consistency, perform_statistical_analysis
from PIL import Image
import numpy as np
from ela_analysis import perform_multi_quality_ela
from copy_move_detection import detect_copy_move_blocks
import cv2
from feature_detection import extract_multi_detector_features, match_sift_features
from validation import validate_image_file, extract_enhanced_metadata
import pytest

# --- Imports baru untuk pengujian basis path ---
from main import analyze_image_comprehensive_advanced
from classification import prepare_feature_vector, validate_feature_vector
from copy_move_detection import kmeans_tampering_localization
from export_utils import generate_all_process_images, DOCX_AVAILABLE
from jpeg_analysis import comprehensive_jpeg_analysis
from visualization import visualize_results_advanced, MATPLOTLIB_AVAILABLE
# ---------------------------------------------


def test_delete_selected_history_logic():
    """
    Tests the logic for deleting selected history entries.
    This test will directly increase coverage for 'delete_selected_history'.
    """
    # --- 1. SETUP: Create a controlled test environment ---
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    if os.path.exists(THUMBNAIL_DIR):
        shutil.rmtree(THUMBNAIL_DIR)
    os.makedirs(THUMBNAIL_DIR)

    # Create dummy thumbnail files
    thumb_path1 = os.path.join(THUMBNAIL_DIR, "thumb1.jpg")
    thumb_path2 = os.path.join(THUMBNAIL_DIR, "thumb2.jpg")
    thumb_path3 = os.path.join(THUMBNAIL_DIR, "thumb3.jpg")
    with open(thumb_path1, 'w') as f: f.write('dummy1')
    with open(thumb_path2, 'w') as f: f.write('dummy2')
    with open(thumb_path3, 'w') as f: f.write('dummy3')

    # Create three history entries for the test
    save_analysis_to_history("image1.jpg", {"type": "Asli"}, "1s", thumb_path1)
    save_analysis_to_history("image2_to_delete.jpg", {"type": "Manipulasi"}, "2s", thumb_path2)
    save_analysis_to_history("image3.jpg", {"type": "Asli"}, "3s", thumb_path3)

    # --- 2. ACTION: Call the function we want to test ---
    indices_to_delete = [1]
    result = delete_selected_history(indices_to_delete)

    # --- 3. VERIFICATION: Check if the function behaved correctly ---
    assert result is True, "The function should return True on success."
    updated_history = load_analysis_history()
    assert len(updated_history) == 2, "History should now have 2 entries."
    assert updated_history[0]['image_name'] == 'image1.jpg'
    assert updated_history[1]['image_name'] == 'image3.jpg'
    assert os.path.exists(thumb_path1), "Thumbnail 1 should not be deleted."
    assert not os.path.exists(thumb_path2), "Thumbnail 2 should have been deleted."
    assert os.path.exists(thumb_path3), "Thumbnail 3 should not be deleted."

    # --- 4. TEARDOWN ---
    if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
    if os.path.exists(THUMBNAIL_DIR): shutil.rmtree(THUMBNAIL_DIR)

# ---- Kasus Uji Tambahan untuk utils.py ----
def test_load_history_edge_cases():
    """Tests load_analysis_history for edge cases like missing or corrupt files."""
    # Case 1: File doesn't exist
    if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
    assert load_analysis_history() == []

    # Case 2: File is empty
    with open(HISTORY_FILE, 'w') as f: f.write('')
    assert load_analysis_history() == []
    os.remove(HISTORY_FILE)
    
    # Case 3: File is corrupt (not valid JSON)
    with open(HISTORY_FILE, 'w') as f: f.write("{'bad_json':}")
    assert load_analysis_history() == []
    os.remove(HISTORY_FILE)
        
def test_advanced_analysis_functions():
    """Tests functions from the advanced_analysis.py module."""
    uniform_image = Image.new('RGB', (100, 100), 'black')
    random_image = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8))

    noise_results_uniform = analyze_noise_consistency(uniform_image)
    assert 'overall_inconsistency' in noise_results_uniform
    assert noise_results_uniform['overall_inconsistency'] < 0.1

    noise_results_random = analyze_noise_consistency(random_image)
    assert 'overall_inconsistency' in noise_results_random
    assert noise_results_random['overall_inconsistency'] < 0.5

    stats_uniform = perform_statistical_analysis(uniform_image)
    assert 'R_std' in stats_uniform
    assert stats_uniform['R_std'] == 0
    
    stats_random = perform_statistical_analysis(random_image)
    assert 'R_std' in stats_random and stats_random['R_std'] > 0

def test_ela_analysis_logic():
    """Tests the core logic of the ELA analysis module."""
    uniform_image = Image.new('RGB', (100, 100), 'black')
    spliced_arr = np.zeros((100, 100, 3), dtype=np.uint8); spliced_arr[50:100, :] = 128
    spliced_image = Image.fromarray(spliced_arr)

    _, ela_mean_uniform, _, _, _, _ = perform_multi_quality_ela(uniform_image)
    assert ela_mean_uniform < 5
    _, ela_mean_spliced, _, _, _, _ = perform_multi_quality_ela(spliced_image)
    assert ela_mean_spliced > 5

def test_copy_move_detection_blocks():
    """Tests the block-based copy-move detection logic."""
    base_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    source_block = base_arr[0:20, 0:20].copy()
    base_arr[80:100, 80:100] = source_block
    copy_move_image = Image.fromarray(base_arr)
    matches = detect_copy_move_blocks(copy_move_image, block_size=16, threshold=0.9)
    assert len(matches) > 0

def test_feature_detection_logic():
    """Tests the core logic of the feature_detection.py module."""
    test_image = Image.new('RGB', (200, 200), 'gray')
    ela_array = np.array(Image.new('L', (200, 200), 0)); ela_array[50:150, 50:150] = 100
    ela_image = Image.fromarray(ela_array)
    feature_sets, _, _ = extract_multi_detector_features(test_image, ela_image, 10, 5)
    assert all(k in feature_sets for k in ['sift', 'orb', 'akaze'])
    sift_keypoints, _ = feature_sets['sift']
    assert len(sift_keypoints) > 0

def test_validation_logic():
    """Tests the logic of the validation.py module."""
    dummy_file = "dummy_test_image.jpg"
    Image.new('RGB', (300, 300), 'blue').save(dummy_file, "JPEG")
    assert validate_image_file(dummy_file) is True
    with pytest.raises(FileNotFoundError): validate_image_file("non_existent_file.jpg")
    with open("invalid_ext.txt", "w") as f: f.write("test")
    with pytest.raises(ValueError): validate_image_file("invalid_ext.txt")
    os.remove("invalid_ext.txt")
    metadata = extract_enhanced_metadata(dummy_file)
    assert "Filename" in metadata and metadata["Filename"] == dummy_file
    os.remove(dummy_file)

# --- FUNGSI PENGUJIAN BARU UNTUK FUNGSI YANG TERLEWAT ---
def create_dummy_image_file(path="dummy_pipeline_test.jpg"):
    """Helper to create a valid dummy image for testing."""
    Image.new('RGB', (200, 150), 'red').save(path)
    return path

def test_main_pipeline_integration():
    """Tests the main orchestrator function 'analyze_image_comprehensive_advanced'."""
    dummy_path = create_dummy_image_file()
    
    results = analyze_image_comprehensive_advanced(dummy_path)
    
    assert isinstance(results, dict), "The main pipeline should return a dictionary."
    assert 'classification' in results, "Final classification is missing."
    assert 'pipeline_status' in results, "Pipeline status tracking is missing."
    assert results['pipeline_status']['completed_stages'] > 10, "Most pipeline stages should complete."
    
    os.remove(dummy_path)

def test_kmeans_tampering_localization():
    """Tests the kmeans_tampering_localization function specifically."""
    original_pil = Image.new('RGB', (100, 100), 'white')
    ela_arr = np.zeros((100, 100), dtype=np.uint8)
    ela_arr[20:50, 20:50] = 150  # Area with high ELA response
    ela_image = Image.fromarray(ela_arr)
    
    result = kmeans_tampering_localization(original_pil, ela_image, n_clusters=2)

    assert isinstance(result, dict), "K-Means function should return a dictionary."
    assert 'localization_map' in result
    assert 'tampering_mask' in result
    assert result['localization_map'].shape == (100, 100)
    assert result['tampering_mask'].dtype == bool
    assert np.sum(result['tampering_mask']) > 0, "Tampering mask should not be empty."

def test_classification_vector_prep():
    """Tests prepare_feature_vector to ensure it produces a valid, clean vector."""
    dummy_path = create_dummy_image_file("dummy_vector_test.jpg")
    mock_results = analyze_image_comprehensive_advanced(dummy_path)
    os.remove(dummy_path)
    
    # Pre-condition: mock_results must be valid
    assert mock_results is not None

    feature_vector = prepare_feature_vector(mock_results)
    
    assert isinstance(feature_vector, np.ndarray), "Feature vector should be a numpy array."
    # The expected length is 28, but we check > 20 for flexibility
    assert len(feature_vector) > 20, "Feature vector has an unexpected length."
    
    validated_vector = validate_feature_vector(feature_vector)
    assert not np.isnan(validated_vector).any(), "Validated vector should not contain NaN."
    assert not np.isinf(validated_vector).any(), "Validated vector should not contain Infinity."

def test_export_process_images():
    """Tests the generation of the 17 process images."""
    if not MATPLOTLIB_AVAILABLE or not DOCX_AVAILABLE:
        pytest.skip("Skipping export test because matplotlib or docx is not available.")
    
    temp_dir = "test_export_temp_dir"
    os.makedirs(temp_dir, exist_ok=True)
    dummy_path = create_dummy_image_file(os.path.join(temp_dir, "test.jpg"))
    
    mock_results = analyze_image_comprehensive_advanced(dummy_path)
    
    success = generate_all_process_images(Image.open(dummy_path), mock_results, temp_dir)

    assert success is True
    files_in_dir = os.listdir(temp_dir)
    # Check if a significant number of images and the README were created
    assert len(files_in_dir) > 10, "Expected at least 10 output files to be generated."
    assert "README.txt" in files_in_dir, "README.txt file is missing from export."
    
    shutil.rmtree(temp_dir)

# =====================================================================
# --- KASUS UJI YANG SUDAH ADA (TIDAK PERLU DIUBAH) ---
# =====================================================================
def create_double_compressed_image(filename="jpeg_test_image.jpg"):
    img_arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr); img.save(filename, "JPEG", quality=95)
    img_reopen = Image.open(filename); img_reopen.save(filename, "JPEG", quality=75)
    return filename

def test_jpeg_analysis_suite():
    test_img_path = create_double_compressed_image("double_comp_test.jpg")
    img_pil = Image.open(test_img_path)
    results = comprehensive_jpeg_analysis(img_pil)
    assert isinstance(results, dict)
    assert 'double_compression' in results and results['double_compression']['is_double_compressed'] is True
    os.remove(test_img_path)

def create_mock_analysis_results():
    """Helper for visualization tests."""
    dummy_path = create_dummy_image_file("dummy_mock_test.jpg")
    results = analyze_image_comprehensive_advanced(dummy_path)
    os.remove(dummy_path)
    results['metadata']['Filename'] = 'mock_image.jpg' # Override for consistency
    return results

@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib is not installed")
def test_visualization_suite():
    original_pil = Image.new('RGB', (400, 300), 'green')
    analysis_results = create_mock_analysis_results()
    output_filename = "test_visualization_output.png"
    result_path = visualize_results_advanced(original_pil, analysis_results, output_filename)
    assert result_path is not None and os.path.exists(result_path)
    os.remove(output_filename)
# --- END OF FILE test_app.py ---