"""
Advanced Forensic Image Analysis System v2.0
Main execution file

Usage:
    python main.py <image_path> [options]

Example:
    python main.py test_image.jpg
    python main.py test_image.jpg --export-all
    python main.py test_image.jpg --output-dir ./results
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

# PERBAIKAN: Import fungsi save_analysis_to_history dari utils
from utils import save_analysis_to_history, HISTORY_FILE, THUMBNAIL_DIR, load_analysis_history, delete_all_history, delete_selected_history

# Import semua modul
from validation import validate_image_file, extract_enhanced_metadata, advanced_preprocess_image
from ela_analysis import perform_multi_quality_ela
from feature_detection import extract_multi_detector_features
from copy_move_detection import detect_copy_move_advanced, detect_copy_move_blocks, kmeans_tampering_localization
from advanced_analysis import (analyze_noise_consistency, analyze_frequency_domain,
                              analyze_texture_consistency, analyze_edge_consistency,
                              analyze_illumination_consistency, perform_statistical_analysis)
from jpeg_analysis import comprehensive_jpeg_analysis # Changed from advanced_jpeg_analysis, assuming this includes everything
from classification import classify_manipulation_advanced, prepare_feature_vector
# No longer directly visualize results in main, use export_utils functions
from export_utils import export_complete_package, export_visualization_png # Keeping export_visualization_png for --export-vis option


# ======================= FUNGSI BARU UNTUK MEMPERBAIKI LOKALISASI =======================
def advanced_tampering_localization(image_pil, analysis_results):
    """
    Advanced tampering localization menggunakan multiple methods.
    Fungsi ini menggabungkan beberapa hasil untuk membuat masker deteksi yang lebih andal.
    """
    print("  -> Combining multiple localization methods...")
    
    # Ambil data yang diperlukan dari hasil analisis
    ela_image_obj = analysis_results.get('ela_image') # It could be a PIL Image or numpy array
    if ela_image_obj is None:
        print("  Warning: ELA image data is missing, cannot perform advanced localization.")
        # Ensure fallback to correct size if original_pil.size is tuple (w,h) but mask is (h,w)
        fallback_mask_shape = (image_pil.height, image_pil.width) # Mask expects (H,W)
        return {
            'kmeans_localization': {'localization_map': np.zeros(fallback_mask_shape), 'tampering_mask': np.zeros(fallback_mask_shape, dtype=bool)},
            'threshold_mask': np.zeros(fallback_mask_shape, dtype=bool),
            'combined_tampering_mask': np.zeros(fallback_mask_shape, dtype=bool),
            'tampering_percentage': 0
        }
    
    # Ensure ela_image_obj is a numpy array for consistency in processing
    ela_array = np.array(ela_image_obj.convert('L')) if isinstance(ela_image_obj, Image.Image) else np.array(ela_image_obj)
    
    # 1. K-means based localization (hasil dari copy_move_detection)
    # The `kmeans_tampering_localization` returns its own mask.
    kmeans_result = kmeans_tampering_localization(image_pil, ela_array) # Pass ela_array, not PIL image

    kmeans_mask_initial = kmeans_result.get('tampering_mask', np.zeros_like(ela_array, dtype=bool))
    
    # Check if kmeans_mask is valid, if it's smaller or different size (shouldn't be, but robust)
    if kmeans_mask_initial.shape != ela_array.shape:
        print(f"  Warning: K-means mask shape {kmeans_mask_initial.shape} != ELA shape {ela_array.shape}. Resizing K-means mask.")
        kmeans_mask = cv2.resize(kmeans_mask_initial.astype(np.uint8), (ela_array.shape[1], ela_array.shape[0])).astype(bool)
    else:
        kmeans_mask = kmeans_mask_initial

    # 2. Threshold-based localization from ELA
    ela_mean = analysis_results.get('ela_mean', 0)
    ela_std = analysis_results.get('ela_std', 1) # Use 1 if std is zero to avoid division by zero
    
    # Dynamic threshold: e.g., 2 standard deviations above the mean for very high ELA values
    # Clip threshold to avoid extreme values
    ela_threshold = np.clip(ela_mean + 2 * ela_std, 10, 200) # Minimum 10, Maximum 200 for plausible ELA range
    threshold_mask = ela_array > ela_threshold

    # 3. Combined localization mask
    # Menggabungkan masker dari K-Means dan ELA thresholding
    combined_mask = np.logical_or(kmeans_mask, threshold_mask)

    # 4. Morphological operations untuk membersihkan masker (closing to fill small gaps, opening to remove small specks)
    kernel_size = min(max(5, int(min(image_pil.width, image_pil.height) / 100)), 15) # Adapt kernel size based on image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Convert bool mask to uint8 for OpenCV ops, then back to bool
    cleaned_mask_uint8 = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned_mask_uint8 = cv2.morphologyEx(cleaned_mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

    cleaned_mask = cleaned_mask_uint8.astype(bool) # Final mask as boolean

    # Calculate tampering percentage
    # Use sum of elements on cleaned_mask (boolean mask)
    tampering_percentage = (np.sum(cleaned_mask) / cleaned_mask.size) * 100 if cleaned_mask.size > 0 else 0

    return {
        'kmeans_localization': kmeans_result, # Keep full KMeans result for diagnostics
        'threshold_mask': threshold_mask.astype(bool),
        'combined_tampering_mask': cleaned_mask, # This is the crucial combined output
        'tampering_percentage': float(tampering_percentage) # Ensure float for JSON serialization
    }
# ======================= AKHIR FUNGSI BARU =======================


# Bagian yang perlu dimodifikasi di main.py untuk tracking status pipeline

def analyze_image_comprehensive_advanced(image_path, output_dir="./results"):
    """Advanced comprehensive image analysis pipeline with status tracking"""
    print(f"\n{'='*80}")
    print(f"ADVANCED FORENSIC IMAGE ANALYSIS SYSTEM v2.0")
    print(f"Enhanced Detection: Copy-Move, Splicing, Authentic Images")
    print(f"{'='*80}\n")

    start_time = time.time()
    
    # Initialize pipeline status tracking
    pipeline_status = {
        'total_stages': 17,
        'completed_stages': 0,
        'failed_stages': [],
        'stage_details': {} # Detailed success/failure for each stage
    }

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize analysis_results dictionary to populate throughout the pipeline
    analysis_results = {
        'metadata': {},
        'ela_image': None, 'ela_mean': 0.0, 'ela_std': 0.0, 'ela_regional_stats': {}, 'ela_quality_stats': [], 'ela_variance': np.array([]),
        'feature_sets': {}, 'sift_keypoints': [], 'sift_descriptors': None, 'sift_matches': 0, 'ransac_matches': [], 'ransac_inliers': 0, 'geometric_transform': None,
        'block_matches': [],
        'noise_analysis': {}, 'noise_map': np.array([]),
        'jpeg_analysis': {}, 'jpeg_ghost': np.array([]), 'jpeg_ghost_suspicious_ratio': 0.0,
        'frequency_analysis': {},
        'texture_analysis': {},
        'edge_analysis': {},
        'illumination_analysis': {},
        'statistical_analysis': {},
        'color_analysis': {}, # Re-added color_analysis placeholder, although illumination_analysis might cover this
        'roi_mask': np.array([]), # Feature detection ROI mask
        'enhanced_gray': np.array([]), # Preprocessed grayscale image
        'localization_analysis': {}, # This key will be updated by advanced_tampering_localization
        'classification': {}, # This key will be updated by classification_advanced
        'processing_time': "0s",
        'pipeline_status': pipeline_status # Initially set, will be updated.
    }


    # 1. File Validation
    print("🚀 [1/17] Validating image file...")
    try:
        validate_image_file(image_path)
        print("✅ [1/17] File validation passed")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['file_validation'] = True
    except Exception as e:
        print(f"❌ [1/17] File validation failed: {e}")
        pipeline_status['failed_stages'].append('file_validation')
        pipeline_status['stage_details']['file_validation'] = False
        analysis_results['classification'] = {'type': 'Failed to Load', 'confidence': 'Very Low', 'copy_move_score': 0, 'splicing_score': 0, 'details': [f"File validation failed: {e}"]}
        return analysis_results # Exit early if file invalid

    # 2. Load image
    print("🖼️ [2/17] Loading image...")
    try:
        original_image = Image.open(image_path)
        print(f"✅ [2/17] Image loaded: {os.path.basename(image_path)}")
        print(f"  Size: {original_image.size}, Mode: {original_image.mode}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['image_loading'] = True
    except Exception as e:
        print(f"❌ [2/17] Error loading image: {e}")
        pipeline_status['failed_stages'].append('image_loading')
        pipeline_status['stage_details']['image_loading'] = False
        analysis_results['classification'] = {'type': 'Failed to Load', 'confidence': 'Very Low', 'copy_move_score': 0, 'splicing_score': 0, 'details': [f"Image loading failed: {e}"]}
        return analysis_results

    # 3. Enhanced metadata extraction
    print("🔍 [3/17] Extracting enhanced metadata...")
    try:
        metadata = extract_enhanced_metadata(image_path)
        analysis_results['metadata'] = metadata # Populate result dict
        print(f"  Authenticity Score: {metadata['Metadata_Authenticity_Score']}/100")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['metadata_extraction'] = True
    except Exception as e:
        print(f"❌ [3/17] Metadata extraction failed: {e}")
        metadata_default = {'Metadata_Authenticity_Score': 0, 'Filename': os.path.basename(image_path), 'FileSize (bytes)': os.path.getsize(image_path), 'Metadata_Inconsistency': ['Error extracting metadata']}
        analysis_results['metadata'] = metadata_default
        pipeline_status['failed_stages'].append('metadata_extraction')
        pipeline_status['stage_details']['metadata_extraction'] = False

    # 4. Advanced preprocessing
    print("🔧 [4/17] Advanced preprocessing...")
    try:
        # Pass a copy of the image to preprocessing to ensure it's not modified in place
        preprocessed_image_pil, original_preprocessed_pil_copy = advanced_preprocess_image(original_image.copy())
        analysis_results['enhanced_gray'] = np.array(preprocessed_image_pil.convert('L')) # Save enhanced grayscale for later steps
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['preprocessing'] = True
    except Exception as e:
        print(f"❌ [4/17] Preprocessing failed: {e}")
        preprocessed_image_pil = original_image.copy().convert('RGB')
        original_preprocessed_pil_copy = original_image.copy().convert('RGB')
        analysis_results['enhanced_gray'] = np.array(original_image.convert('L')) # Fallback to raw grayscale
        pipeline_status['failed_stages'].append('preprocessing')
        pipeline_status['stage_details']['preprocessing'] = False

    # 5. Multi-quality ELA
    print("📊 [5/17] Multi-quality Error Level Analysis...")
    try:
        # ELA operates on PIL Image and returns a PIL Image for ela_image_data, so handle that correctly.
        ela_image_data, ela_mean, ela_std, ela_regional, ela_quality_stats, ela_variance = perform_multi_quality_ela(preprocessed_image_pil.copy())
        analysis_results['ela_image'] = ela_image_data # This is a PIL Image object
        analysis_results['ela_mean'] = ela_mean
        analysis_results['ela_std'] = ela_std
        analysis_results['ela_regional_stats'] = ela_regional
        analysis_results['ela_quality_stats'] = ela_quality_stats
        analysis_results['ela_variance'] = ela_variance # Store numpy array
        print(f"  ELA Stats: μ={ela_mean:.2f}, σ={ela_std:.2f}, Regions={ela_regional['outlier_regions']}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['ela_analysis'] = True
    except Exception as e:
        print(f"❌ [5/17] ELA analysis failed: {e}")
        # Default fallback values for ELA. `ela_image` is PIL `L` mode.
        analysis_results['ela_image'] = Image.new('L', preprocessed_image_pil.size)
        analysis_results['ela_mean'] = 0.0
        analysis_results['ela_std'] = 0.0
        analysis_results['ela_regional_stats'] = {'outlier_regions': 0, 'regional_inconsistency': 0.0, 'suspicious_regions': [], 'mean_variance':0.0}
        analysis_results['ela_quality_stats'] = []
        analysis_results['ela_variance'] = np.zeros(preprocessed_image_pil.size[::-1]) # shape is H,W, convert from (W,H)
        pipeline_status['failed_stages'].append('ela_analysis')
        pipeline_status['stage_details']['ela_analysis'] = False

    # 6. Multi-detector feature extraction
    print("🎯 [6/17] Multi-detector feature extraction...")
    try:
        # Pass copies to ensure `extract_multi_detector_features` has independent objects
        # And ensure `ela_image` is valid for use here, pass the PIL Image object
        feature_sets, roi_mask, gray_enhanced_output = extract_multi_detector_features(
            preprocessed_image_pil.copy(), analysis_results['ela_image'], analysis_results['ela_mean'], analysis_results['ela_std'])
        
        analysis_results['feature_sets'] = feature_sets
        analysis_results['sift_keypoints'] = feature_sets.get('sift', ([], None))[0] # Store keypoints separately
        analysis_results['sift_descriptors'] = feature_sets.get('sift', ([], None))[1] # Store descriptors separately

        analysis_results['roi_mask'] = roi_mask # This is a numpy array (uint8)
        analysis_results['enhanced_gray'] = gray_enhanced_output # This is a numpy array

        total_features = sum(len(kp) for kp, _ in feature_sets.values() if kp is not None) # Handle None keypoints safely
        print(f"  Total keypoints: {total_features}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['feature_extraction'] = True
    except Exception as e:
        print(f"❌ [6/17] Feature extraction failed: {e}")
        analysis_results['feature_sets'] = {'sift': ([], None), 'orb': ([], None), 'akaze': ([], None)}
        analysis_results['sift_keypoints'] = []
        analysis_results['sift_descriptors'] = None
        analysis_results['roi_mask'] = np.ones(preprocessed_image_pil.size[::-1], dtype=np.uint8) * 255 # HxW, All 255 for mask
        analysis_results['enhanced_gray'] = np.array(preprocessed_image_pil.convert('L'))
        pipeline_status['failed_stages'].append('feature_extraction')
        pipeline_status['stage_details']['feature_extraction'] = False

    # 7. Advanced copy-move detection (Feature-based)
    print("🔄 [7/17] Advanced copy-move detection (Feature-based)...")
    try:
        # `detect_copy_move_advanced` needs `feature_sets` dict, and image_shape is (W, H)
        ransac_matches, ransac_inliers, transform = detect_copy_move_advanced(
            analysis_results['feature_sets'], preprocessed_image_pil.size)
        analysis_results['ransac_matches'] = ransac_matches # Match objects
        analysis_results['sift_matches'] = len(ransac_matches) # Total matches before RANSAC (or after)
        analysis_results['ransac_inliers'] = ransac_inliers # Count of RANSAC-verified matches
        analysis_results['geometric_transform'] = transform # (transform_type, matrix)
        print(f"  RANSAC inliers: {ransac_inliers}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['feature_based_copymove_detection'] = True
    except Exception as e:
        print(f"❌ [7/17] Feature-based copy-move detection failed: {e}")
        analysis_results['ransac_matches'] = []
        analysis_results['sift_matches'] = 0
        analysis_results['ransac_inliers'] = 0
        analysis_results['geometric_transform'] = None
        pipeline_status['failed_stages'].append('feature_based_copymove_detection')
        pipeline_status['stage_details']['feature_based_copymove_detection'] = False

    # 8. Enhanced block matching
    print("🧩 [8/17] Enhanced block-based detection...")
    try:
        block_matches = detect_copy_move_blocks(preprocessed_image_pil.copy())
        analysis_results['block_matches'] = block_matches # List of dicts
        print(f"  Block matches: {len(block_matches)}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['block_based_copymove_detection'] = True
    except Exception as e:
        print(f"❌ [8/17] Block matching failed: {e}")
        analysis_results['block_matches'] = []
        pipeline_status['failed_stages'].append('block_based_copymove_detection')
        pipeline_status['stage_details']['block_based_copymove_detection'] = False

    # 9. Advanced noise analysis
    print("📡 [9/17] Advanced noise consistency analysis...")
    try:
        # Pass a copy, result is a dict with floats/lists
        noise_analysis_res = analyze_noise_consistency(preprocessed_image_pil.copy())
        analysis_results['noise_analysis'] = noise_analysis_res
        # Noise map usually would be generated/stored as a separate visual from this module
        # If analyze_noise_consistency returns a map, add it. Otherwise, generate a simple one.
        # Currently, it doesn't return a direct map. Let's make a dummy one if not explicitly generated later.
        analysis_results['noise_map'] = np.zeros(preprocessed_image_pil.size[::-1]) # HxW np array
        if noise_analysis_res.get('noise_characteristics'):
            # Generate a simple map based on overall inconsistency or laplacian variance per block
            # For simplicity, if actual map is not produced, take preprocessed_image_pil and calculate
            # its own raw laplacian as a "noise map" visualization for `main`
            try:
                gray_for_noise_map = np.array(preprocessed_image_pil.convert('L'))
                if gray_for_noise_map.shape[0] >=3 and gray_for_noise_map.shape[1] >=3:
                    laplacian = np.abs(cv2.Laplacian(gray_for_noise_map, cv2.CV_64F))
                    analysis_results['noise_map'] = (laplacian / np.max(laplacian) * 255).astype(np.uint8)
            except Exception:
                analysis_results['noise_map'] = np.zeros(preprocessed_image_pil.size[::-1], dtype=np.uint8) # Fallback to black if no map possible
            

        print(f"  Noise inconsistency: {noise_analysis_res.get('overall_inconsistency', 0):.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['noise_analysis'] = True
    except Exception as e:
        print(f"❌ [9/17] Noise analysis failed: {e}")
        analysis_results['noise_analysis'] = {'overall_inconsistency': 0.0, 'outlier_count': 0, 'noise_characteristics': []}
        analysis_results['noise_map'] = np.zeros(preprocessed_image_pil.size[::-1]) # HxW black map
        pipeline_status['failed_stages'].append('noise_analysis')
        pipeline_status['stage_details']['noise_analysis'] = False

    # 10. Comprehensive JPEG analysis (includes basic, ghost, block, double compression)
    print("📷 [10/17] Comprehensive JPEG artifact analysis...")
    try:
        # Assuming `comprehensive_jpeg_analysis` returns all nested results correctly
        jpeg_full_results = comprehensive_jpeg_analysis(original_preprocessed_pil_copy.copy()) # Use original preprocessed for full fidelity
        analysis_results['jpeg_analysis'] = jpeg_full_results
        
        # Extract direct maps and ratios for convenience from full result. Safely access deeply nested items.
        analysis_results['jpeg_ghost'] = jpeg_full_results.get('ghost_analysis', {}).get('ghost_map', np.array([]))
        analysis_results['jpeg_ghost_suspicious_ratio'] = jpeg_full_results.get('ghost_analysis', {}).get('ghost_coverage', 0.0)

        # Handle the case where jpeg_ghost map might be empty from comprehensive_jpeg_analysis
        if analysis_results['jpeg_ghost'].size == 0 and preprocessed_image_pil.size[0]*preprocessed_image_pil.size[1] > 0: # Ensure not totally empty or 0 sized if image exists
            analysis_results['jpeg_ghost'] = np.zeros(preprocessed_image_pil.size[::-1]) # Ensure it's a valid zero-filled array of correct shape
            
        print(f"  JPEG anomalies: {analysis_results['jpeg_ghost_suspicious_ratio']:.1%}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['jpeg_analysis'] = True

    except Exception as e:
        print(f"❌ [10/17] JPEG analysis failed: {e}")
        # Default fallback values for all parts of JPEG analysis result
        analysis_results['jpeg_analysis'] = {
            'basic_analysis': {'quality_responses': [], 'response_variance': 0.0, 'double_compression_indicator': 0.0, 'estimated_original_quality': 0, 'compression_inconsistency': False},
            'ghost_analysis': {'ghost_map': np.array([]), 'ghost_coverage': 0.0, 'ghost_intensity': 0.0, 'total_ghost_score': 0.0},
            'block_analysis': {'block_artifacts': [], 'overall_blocking_score': 0.0},
            'double_compression': {'double_compression_score': 0.0, 'is_double_compressed': False, 'indicators': []}
        }
        analysis_results['jpeg_ghost'] = np.zeros(preprocessed_image_pil.size[::-1]) # HxW black map
        analysis_results['jpeg_ghost_suspicious_ratio'] = 0.0
        pipeline_status['failed_stages'].append('jpeg_analysis')
        pipeline_status['stage_details']['jpeg_analysis'] = False

    # 11. Frequency domain analysis
    print("🌊 [11/17] Frequency domain analysis...")
    try:
        # Returns a dict with float/dict
        frequency_analysis_res = analyze_frequency_domain(preprocessed_image_pil.copy())
        analysis_results['frequency_analysis'] = frequency_analysis_res
        print(f"  Frequency inconsistency: {frequency_analysis_res.get('frequency_inconsistency', 0):.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['frequency_analysis'] = True
    except Exception as e:
        print(f"❌ [11/17] Frequency analysis failed: {e}")
        analysis_results['frequency_analysis'] = {'frequency_inconsistency': 0.0, 'dct_stats': {'low_freq_energy': 0.0, 'high_freq_energy': 0.0, 'mid_freq_energy': 0.0, 'freq_ratio': 0.0}}
        pipeline_status['failed_stages'].append('frequency_analysis')
        pipeline_status['stage_details']['frequency_analysis'] = False

    # 12. Texture consistency analysis
    print("🧵 [12/17] Texture consistency analysis...")
    try:
        # Returns a dict with floats/dict
        texture_analysis_res = analyze_texture_consistency(preprocessed_image_pil.copy())
        analysis_results['texture_analysis'] = texture_analysis_res
        print(f"  Texture inconsistency: {texture_analysis_res.get('overall_inconsistency', 0):.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['texture_analysis'] = True
    except Exception as e:
        print(f"❌ [12/17] Texture analysis failed: {e}")
        analysis_results['texture_analysis'] = {'overall_inconsistency': 0.0, 'texture_consistency': {}, 'texture_features': []}
        pipeline_status['failed_stages'].append('texture_analysis')
        pipeline_status['stage_details']['texture_analysis'] = False

    # 13. Edge consistency analysis
    print("📐 [13/17] Edge density analysis...")
    try:
        # Returns a dict with floats/lists
        edge_analysis_res = analyze_edge_consistency(preprocessed_image_pil.copy())
        analysis_results['edge_analysis'] = edge_analysis_res
        print(f"  Edge inconsistency: {edge_analysis_res.get('edge_inconsistency', 0):.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['edge_analysis'] = True
    except Exception as e:
        print(f"❌ [13/17] Edge analysis failed: {e}")
        analysis_results['edge_analysis'] = {'edge_inconsistency': 0.0, 'edge_densities': [], 'edge_variance': 0.0}
        pipeline_status['failed_stages'].append('edge_analysis')
        pipeline_status['stage_details']['edge_analysis'] = False

    # 14. Illumination analysis
    print("💡 [14/17] Illumination consistency analysis...")
    try:
        # Returns a dict with floats
        illumination_analysis_res = analyze_illumination_consistency(preprocessed_image_pil.copy())
        analysis_results['illumination_analysis'] = illumination_analysis_res
        # Assuming no direct "color_analysis" map is needed explicitly beyond this
        analysis_results['color_analysis'] = {'illumination_inconsistency': illumination_analysis_res.get('overall_illumination_inconsistency', 0.0)}
        print(f"  Illumination inconsistency: {illumination_analysis_res.get('overall_illumination_inconsistency', 0):.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['illumination_analysis'] = True
    except Exception as e:
        print(f"❌ [14/17] Illumination analysis failed: {e}")
        analysis_results['illumination_analysis'] = {'illumination_mean_consistency': 0.0, 'illumination_std_consistency': 0.0, 'gradient_consistency': 0.0, 'overall_illumination_inconsistency': 0.0}
        analysis_results['color_analysis'] = {'illumination_inconsistency': 0.0}
        pipeline_status['failed_stages'].append('illumination_analysis')
        pipeline_status['stage_details']['illumination_analysis'] = False

    # 15. Statistical analysis
    print("📈 [15/17] Statistical analysis...")
    try:
        # Returns a dict with floats
        statistical_analysis_res = perform_statistical_analysis(preprocessed_image_pil.copy())
        analysis_results['statistical_analysis'] = statistical_analysis_res
        print(f"  Overall entropy: {statistical_analysis_res.get('overall_entropy', 0):.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['statistical_analysis'] = True
    except Exception as e:
        print(f"❌ [15/17] Statistical analysis failed: {e}")
        statistical_analysis_default = {ch: 0.0 for ch_metric in ['_mean', '_std', '_skewness', '_kurtosis', '_entropy'] for ch in ['R', 'G', 'B']}
        statistical_analysis_default.update({'rg_correlation': 0.0, 'rb_correlation': 0.0, 'gb_correlation': 0.0, 'overall_entropy': 0.0})
        analysis_results['statistical_analysis'] = statistical_analysis_default
        pipeline_status['failed_stages'].append('statistical_analysis')
        pipeline_status['stage_details']['statistical_analysis'] = False

    # 16. Advanced tampering localization (combines K-Means & ELA, etc.)
    print("🎯 [16/17] Advanced tampering localization...")
    try:
        # `advanced_tampering_localization` uses analysis_results['ela_image']
        localization_results = advanced_tampering_localization(preprocessed_image_pil.copy(), analysis_results)
        analysis_results['localization_analysis'] = localization_results
        print(f"  Tampering area: {localization_results.get('tampering_percentage', 0):.1f}% of image")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['localization_analysis'] = True
    except Exception as e:
        print(f"❌ [16/17] Localization analysis failed: {e}")
        # Default to a safe structure for localization_analysis
        default_loc_map_shape = preprocessed_image_pil.size[::-1] # (H,W)
        analysis_results['localization_analysis'] = {
            'kmeans_localization': {'localization_map': np.zeros(default_loc_map_shape), 'tampering_mask': np.zeros(default_loc_map_shape, dtype=bool)},
            'threshold_mask': np.zeros(default_loc_map_shape, dtype=bool),
            'combined_tampering_mask': np.zeros(default_loc_map_shape, dtype=bool), # Crucial key for heatmap and classification
            'tampering_percentage': 0.0
        }
        pipeline_status['failed_stages'].append('localization_analysis')
        pipeline_status['stage_details']['localization_analysis'] = False
    
    # 17. Advanced classification (uses all collected data)
    print("🤖 [17/17] Advanced manipulation classification...")
    try:
        classification = classify_manipulation_advanced(analysis_results)
        analysis_results['classification'] = classification
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['classification'] = True
    except Exception as e:
        print(f"❌ [17/17] Classification failed: {e}")
        classification_error_default = {
            'type': 'Analysis Error', 'confidence': 'Error',
            'copy_move_score': 0, 'splicing_score': 0, 'details': [f"Classification error: {str(e)}"],
            'ml_scores': {}, 'feature_vector': [], 'traditional_scores': {},
            # Provide minimum uncertainty analysis structure if it failed
            'uncertainty_analysis': { 
                'probabilities': {'copy_move_probability': 0.0, 'splicing_probability': 0.0, 'authentic_probability': 1.0, 'uncertainty_level': 1.0, 'confidence_intervals': {k:{'lower':0,'upper':0} for k in ['copy_move','splicing','authentic']}},
                'report': {'primary_assessment': 'Classification Failed', 'confidence_level': 'Sangat Rendah', 'uncertainty_summary': 'Internal error, result unreliable', 'reliability_indicators': [], 'recommendation': 'Rerun analysis.'},
                'formatted_output': "Classification process failed."
            }
        }
        analysis_results['classification'] = classification_error_default
        pipeline_status['failed_stages'].append('classification')
        pipeline_status['stage_details']['classification'] = False

    # Final updates for processing time and overall pipeline status summary
    processing_time = time.time() - start_time
    analysis_results['processing_time'] = f"{processing_time:.2f}s"
    # Populate the main pipeline_status dict in analysis_results
    analysis_results['pipeline_status'] = pipeline_status


    # Print pipeline summary to console
    print(f"\n{'='*80}")
    print(f"PIPELINE STATUS SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Total Stages: {pipeline_status['total_stages']}")
    print(f"📊 Completed Successfully: {pipeline_status['completed_stages']}")
    print(f"📊 Failed Stages: {len(pipeline_status['failed_stages'])}")
    print(f"📊 Success Rate: {(pipeline_status['completed_stages']/pipeline_status['total_stages']*100):.1f}%")
    
    if pipeline_status['failed_stages']:
        print(f"📊 Failed Components: {', '.join(pipeline_status['failed_stages'])}")

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE - Processing Time: {processing_time:.2f}s")
    print(f"{'='*80}")
    print(f"📊 FINAL RESULT: {analysis_results['classification']['type']}")
    print(f"📊 CONFIDENCE: {analysis_results['classification']['confidence']}")
    print(f"📊 Copy-Move Score: {analysis_results['classification']['copy_move_score']}/100")
    print(f"📊 Splicing Score: {analysis_results['classification']['splicing_score']}/100")
    print(f"{'='*80}\n")

    if analysis_results['classification'].get('details'):
        print("📋 Detection Details:")
        for detail in analysis_results['classification']['details']:
            print(f"  {detail}")
        print()

    # Save to history (existing code)
    try:
        image_filename = os.path.basename(image_path)
        analysis_summary_for_history = {
            'type': analysis_results['classification'].get('type', 'N/A'),
            'confidence': analysis_results['classification'].get('confidence', 'N/A'),
            'copy_move_score': analysis_results['classification'].get('copy_move_score', 0),
            'splicing_score': analysis_results['classification'].get('splicing_score', 0)
        }
        
        # Ensure thumbnail directory exists
        os.makedirs(THUMBNAIL_DIR, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        thumbnail_filename = f"thumb_{os.path.splitext(image_filename)[0]}_{timestamp_str}.jpg" # Add filename to thumbnail name for uniqueness
        thumbnail_path = os.path.join(THUMBNAIL_DIR, thumbnail_filename)
        
        # Open the original image again for thumbnail generation
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB") # Ensure it's RGB before saving
            img_rgb.thumbnail((128, 128))
            img_rgb.save(thumbnail_path, "JPEG", quality=85)
            
        save_analysis_to_history(
            image_filename, 
            analysis_summary_for_history, 
            f"{processing_time:.2f}s",
            thumbnail_path
        )
        print(f"💾 Analysis results and thumbnail saved to history ({thumbnail_path}).")
    except Exception as e:
        print(f"⚠️ Failed to save analysis to history: {e}")
        import traceback
        traceback.print_exc()

    return analysis_results

def main():
    parser = argparse.ArgumentParser(description='Advanced Forensic Image Analysis System v2.0')
    parser.add_argument('image_path', nargs='?', help='Path to the image file to analyze (optional if using app mode)') # Made optional
    parser.add_argument('--output-dir', '-o', default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--export-all', '-e', action='store_true',
                       help='Export complete package (PNG, PDF, DOCX, etc.)')
    parser.add_argument('--export-vis', '-v', action='store_true',
                       help='Export only visualization (PNG)') # Changed from export vis PDF to PNG by default
    parser.add_argument('--export-report', '-r', action='store_true',
                       help='Export only DOCX report')
    parser.add_argument('--full-export-package', '-p', action='store_true',
                        help='Export comprehensive package (all 17 images, HTML, reports, ZIP)')

    args = parser.parse_args()

    # If no image path provided and not in Streamlit context, tell user to use CLI
    if not args.image_path:
        print("Please provide an image path or run in Streamlit app mode.")
        print("Usage: python main.py <image_path> [options]")
        print("Or for comprehensive package: python main.py <image_path> -p")
        print("Exiting...")
        sys.exit(1)


    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file '{args.image_path}' not found!")
        sys.exit(1)

    try:
        analysis_results = analyze_image_comprehensive_advanced(args.image_path, args.output_dir)

        if analysis_results is None:
            print("❌ Analysis failed!")
            sys.exit(1)

        # Re-open original image for export functions which expect a PIL object
        original_image = Image.open(args.image_path)
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        base_path = os.path.join(args.output_dir, base_filename)

        if args.full_export_package: # New comprehensive export package
            print("\n📦 Exporting comprehensive forensic package...")
            from export_utils import export_comprehensive_package # Import the comprehensive package function
            export_comprehensive_package(original_image, analysis_results, base_path)
        elif args.export_all:
            print("\n📦 Exporting complete package (standard set)...")
            export_complete_package(original_image, analysis_results, base_path)
        elif args.export_vis:
            print("\n📊 Exporting PNG visualization...")
            # Using the specific PNG visualization export function
            export_visualization_png(original_image, analysis_results, f"{base_path}_analysis_visuals.png")
        elif args.export_report:
            print("\n📄 Exporting DOCX report...")
            from export_utils import export_to_advanced_docx
            export_to_advanced_docx(original_image, analysis_results, f"{base_path}_report.docx")
        else: # Default: just save a PNG summary if no export options given
            print("\n📊 Exporting basic PNG summary visualization...")
            export_visualization_png(original_image, analysis_results, f"{base_path}_analysis_summary.png")

        print("✅ Analysis completed successfully!")

    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# --- END OF FILE main.py ---