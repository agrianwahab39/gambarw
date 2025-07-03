"""
JPEG Analysis Module for Forensic Image Analysis System
Contains functions for JPEG artifact analysis, ghost detection, and compression analysis
"""

import numpy as np
import cv2
import os
from PIL import Image, ImageChops
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    class ndimage:
        @staticmethod
        def gaussian_filter1d(a, sigma):
            # simple fallback using convolution
            kernel_size = int(6*sigma+1)
            kernel = np.exp(-0.5*((np.arange(kernel_size)-kernel_size//2)/sigma)**2)
            kernel /= kernel.sum()
            pad = kernel_size//2
            a_padded = np.pad(a, pad, mode='edge')
            return np.convolve(a_padded, kernel, mode='valid')
from utils import detect_outliers_iqr, safe_divide
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ======================= JPEG Quality Analysis =======================

def advanced_jpeg_analysis(image_pil, qualities=range(60, 96, 10)):
    """Optimized JPEG artifact analysis with multiple quality testing"""
    print(f"  Testing {len(qualities)} JPEG qualities...")
    
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    # Resize if too large for faster processing
    original_size = image_pil.size
    if max(original_size) > 1500:
        ratio = 1500 / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        print(f"  Resized for analysis: {original_size} → {new_size}")
    
    compression_artifacts = {}
    quality_responses = []
    
    for quality in qualities:
        temp_file = f"temp_quality_{quality}.jpg"
        
        try:
            # Compress and decompress
            image_pil.save(temp_file, 'JPEG', quality=quality)
            with Image.open(temp_file) as recompressed:
                # Calculate difference
                diff = ImageChops.difference(image_pil, recompressed)
                diff_array = np.array(diff.convert('L'))
            
            # Response metrics
            response_mean = np.mean(diff_array)
            response_std = np.std(diff_array)
            response_energy = np.sum(diff_array ** 2)
            response_max = np.max(diff_array)
            response_percentile_95 = np.percentile(diff_array, 95)
            
            quality_responses.append({
                'quality': quality,
                'response_mean': response_mean,
                'response_std': response_std,
                'response_energy': response_energy,
                'response_max': response_max,
                'response_percentile_95': response_percentile_95
            })
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"  Warning: Error processing quality {quality}: {e}")
            continue
    
    # Analyze response patterns
    if not quality_responses:
        return {
            'quality_responses': [],
            'response_variance': 0.0,
            'double_compression_indicator': 0.0,
            'estimated_original_quality': 0,
            'compression_inconsistency': False,
            'optimal_quality': 0,
            'quality_curve_analysis': {}
        }
    
    responses = np.array([r['response_mean'] for r in quality_responses])
    response_variance = np.var(responses)
    
    # Detect double compression
    response_diff = np.diff(responses)
    double_compression_indicator = np.std(response_diff) if len(response_diff) > 0 else 0.0
    
    # Find optimal quality (minimum response)
    min_response_idx = np.argmin(responses)
    estimated_quality = quality_responses[min_response_idx]['quality']
    
    # Advanced quality curve analysis
    quality_curve_analysis = analyze_quality_curve(quality_responses)
    
    # Compression inconsistency detection
    compression_inconsistency = detect_compression_inconsistency(quality_responses)
    
    return {
        'quality_responses': quality_responses,
        'response_variance': response_variance,
        'double_compression_indicator': double_compression_indicator,
        'estimated_original_quality': estimated_quality,
        'compression_inconsistency': compression_inconsistency,
        'optimal_quality': estimated_quality,
        'quality_curve_analysis': quality_curve_analysis
    }

def analyze_quality_curve(quality_responses):
    """Analyze the quality response curve for anomalies"""
    if len(quality_responses) < 3:
        return {'curve_smoothness': 0.0, 'anomaly_points': [], 'curve_type': 'insufficient_data'}
    
    qualities = [r['quality'] for r in quality_responses]
    responses = [r['response_mean'] for r in quality_responses]
    
    # Calculate curve smoothness (second derivative)
    if len(responses) >= 3:
        second_derivative = np.diff(responses, 2)
        curve_smoothness = np.mean(np.abs(second_derivative))
    else:
        curve_smoothness = 0.0
    
    # Detect anomaly points (unusual jumps)
    anomaly_points = []
    if len(responses) > 2:
        for i in range(1, len(responses) - 1):
            left_slope = responses[i] - responses[i-1]
            right_slope = responses[i+1] - responses[i]
            
            # Check for unusual slope changes
            if abs(left_slope - right_slope) > np.std(responses) * 2:
                anomaly_points.append({
                    'quality': qualities[i],
                    'response': responses[i],
                    'anomaly_score': abs(left_slope - right_slope)
                })
    
    # Determine curve type
    if np.all(np.diff(responses) <= 0):
        curve_type = 'monotonic_decreasing'
    elif np.all(np.diff(responses) >= 0):
        curve_type = 'monotonic_increasing'
    else:
        curve_type = 'non_monotonic'
    
    return {
        'curve_smoothness': curve_smoothness,
        'anomaly_points': anomaly_points,
        'curve_type': curve_type,
        'total_variation': np.sum(np.abs(np.diff(responses)))
    }

def detect_compression_inconsistency(quality_responses):
    """Detect JPEG compression inconsistencies"""
    if len(quality_responses) < 3:
        return False
    
    responses = [r['response_mean'] for r in quality_responses]
    
    # Check for unusual response patterns
    # Normal behavior: response should generally decrease as quality increases
    response_diff = np.diff(responses)
    
    # Count increasing responses (unusual behavior)
    increasing_count = np.sum(response_diff > 0)
    total_transitions = len(response_diff)
    
    # If more than 30% of transitions are increasing, it's suspicious
    if total_transitions > 0:
        inconsistency_ratio = increasing_count / total_transitions
    else:
        inconsistency_ratio = 0
    
    # Also check for large variance in response
    response_cv = np.std(responses) / (np.mean(responses) + 1e-6)
    
    # Threshold-based detection
    is_inconsistent = (inconsistency_ratio > 0.3) or (response_cv > 1.0)
    
    return bool(is_inconsistent)

# ======================= JPEG Ghost Analysis =======================

def jpeg_ghost_analysis(image_pil, qualities=range(50, 101, 5)):
    """Perform comprehensive JPEG ghost analysis with enhanced contrast and detail."""
    print(f"  Performing advanced JPEG ghost analysis with {len(qualities)} qualities...")
    
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    original_array = np.array(image_pil)
    h, w, c = original_array.shape
    
    ghost_map = np.zeros((h, w))
    suspicious_map = np.zeros((h, w), dtype=bool)
    quality_response_map = np.zeros((h, w, len(qualities)))
    
    temp_filename = "temp_ghost.jpg"
    
    # Test different JPEG qualities
    min_diff_per_pixel = np.full((h, w), float('inf'))
    quality_map = np.zeros((h, w))
    
    for idx, quality in enumerate(qualities):
        try:
            # Compress at this quality
            image_pil.save(temp_filename, 'JPEG', quality=quality)
            compressed = Image.open(temp_filename)
            compressed_array = np.array(compressed)
            
            # Calculate difference per pixel
            diff = np.mean(np.abs(original_array.astype(float) - compressed_array.astype(float)), axis=2)
            
            # Store response for this quality
            quality_response_map[:, :, idx] = diff
            
            # Track minimum difference for each pixel
            mask = diff < min_diff_per_pixel
            min_diff_per_pixel[mask] = diff[mask]
            quality_map[mask] = quality
            
            # Find areas with unexpectedly low difference (already compressed at this quality)
            threshold = np.percentile(diff, 10)
            low_diff_mask = diff < threshold
            
            # Accumulate ghost evidence
            ghost_map += low_diff_mask.astype(float)
            
            # Mark suspicious areas for specific qualities
            if quality in [70, 80, 90]:  # Common compression qualities
                suspicious_threshold = threshold * 0.5
                suspicious_mask = diff < suspicious_threshold
                suspicious_map |= suspicious_mask
                
        except Exception as e:
            print(f"  Warning: Error processing quality {quality}: {e}")
            continue
            
    # Enhanced ghost map processing for better contrast
    if np.any(ghost_map > 0):
        ghost_map = np.log1p(ghost_map)  # Logarithmic scaling
        ghost_map = (ghost_map - np.min(ghost_map)) / (np.max(ghost_map) - np.min(ghost_map) + 1e-9) # Normalize to 0-1
    else:
        # Jika tidak ada ghost yang terdeteksi, kembalikan peta dengan nilai konstan kecil
        # untuk menghindari peta kosong dan memberikan visual yang informatif.
        ghost_map = np.full((h, w), 0.1)
    
    # Enhanced suspicious area detection using response variance
    response_variance = np.var(quality_response_map, axis=2)
    
    # Areas with very low variance in quality response are suspicious
    low_variance_threshold = np.percentile(response_variance, 25)
    low_variance_mask = response_variance < low_variance_threshold
    suspicious_map |= low_variance_mask
    
    # Advanced ghost pattern analysis, now passing quality_map
    ghost_analysis = analyze_ghost_patterns(ghost_map, quality_map, quality_response_map, qualities)
    
    # Clean up
    try:
        os.remove(temp_filename)
    except:
        pass
    
    print(f"  Advanced JPEG ghost analysis completed")
    
    return ghost_map, suspicious_map, ghost_analysis

def analyze_ghost_patterns(ghost_map, quality_map, quality_response_map, qualities):
    """Analyze JPEG ghost patterns for detailed insights, including regional quality estimation."""
    h, w = ghost_map.shape
    
    # Find ghost regions (connected components) using an adaptive threshold
    if np.any(ghost_map > 0):
        threshold = np.percentile(ghost_map[ghost_map > 0], 85) # Use 85th percentile of non-zero values
    else:
        threshold = 1.0 # Should not find any components
        
    ghost_binary = (ghost_map > threshold).astype(np.uint8)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ghost_cleaned = cv2.morphologyEx(ghost_binary, cv2.MORPH_CLOSE, kernel)
    ghost_cleaned = cv2.morphologyEx(ghost_cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(ghost_cleaned)
    
    ghost_regions = []
    for label in range(1, num_labels):
        region_mask = (labels == label)
        region_size = np.sum(region_mask)
        
        if region_size > 100:  # Minimum size threshold
            # Calculate region statistics
            region_ghost_mean = np.mean(ghost_map[region_mask])
            region_responses = quality_response_map[region_mask, :]
            region_response_variance = np.var(region_responses, axis=1)

            # Find estimated quality for the region from the quality_map
            qualities_in_region = quality_map[region_mask]
            estimated_region_quality = np.median(qualities_in_region)

            # Find region bounding box
            coords = np.argwhere(region_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            ghost_regions.append({
                'label': label,
                'size': region_size,
                'bounding_box': (int(x_min), int(y_min), int(x_max), int(y_max)),
                'ghost_strength': region_ghost_mean,
                'response_variance': np.mean(region_response_variance),
                'estimated_quality': estimated_region_quality,
                'centroid': (int(np.mean(coords[:, 1])), int(np.mean(coords[:, 0])))
            })
    
    # Calculate overall ghost statistics
    ghost_coverage = np.sum(ghost_cleaned) / (h * w) # Use cleaned map for coverage
    ghost_intensity = np.mean(ghost_map[ghost_cleaned > 0]) if np.any(ghost_cleaned > 0) else 0
    
    # Quality-specific analysis
    quality_analysis = {}
    for idx, quality in enumerate(qualities):
        response = quality_response_map[:, :, idx]
        quality_analysis[quality] = {
            'mean_response': np.mean(response),
            'response_variance': np.var(response),
            'low_response_area': np.sum(response < np.percentile(response, 10)) / (h * w)
        }
    
    return {
        'ghost_regions': ghost_regions,
        'ghost_coverage': ghost_coverage,
        'ghost_intensity': ghost_intensity,
        'quality_analysis': quality_analysis,
        'total_ghost_score': ghost_coverage * ghost_intensity * 10 # Scale score for significance
    }

# ======================= Block-wise JPEG Analysis =======================

def analyze_jpeg_blocks(image_pil, block_size=8):
    """Analyze JPEG 8x8 blocks for compression artifacts"""
    print("  Analyzing JPEG block artifacts...")
    
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    image_array = np.array(image_pil)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    h, w = gray_image.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    
    block_artifacts = []
    blocking_map = np.zeros((h, w))
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            # Extract 8x8 block
            block = gray_image[i*block_size:(i+1)*block_size,
                              j*block_size:(j+1)*block_size]
            
            # DCT analysis
            block_float = block.astype(np.float32)
            dct_block = cv2.dct(block_float)
            
            # Calculate blocking artifacts
            # 1. High frequency energy in specific patterns
            high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
            
            # 2. Quantization noise estimation
            # Look for patterns typical of JPEG quantization
            quantization_noise = estimate_quantization_noise(dct_block)
            
            # 3. Block boundary artifacts
            boundary_artifacts = 0
            if i > 0:  # Top boundary
                top_diff = np.abs(gray_image[i*block_size, j*block_size:(j+1)*block_size] - 
                                gray_image[i*block_size-1, j*block_size:(j+1)*block_size])
                boundary_artifacts += np.mean(top_diff)
            
            if j > 0:  # Left boundary
                left_diff = np.abs(gray_image[i*block_size:(i+1)*block_size, j*block_size] - 
                                 gray_image[i*block_size:(i+1)*block_size, j*block_size-1])
                boundary_artifacts += np.mean(left_diff)
            
            # Store block analysis
            block_artifacts.append({
                'position': (i, j),
                'high_freq_energy': high_freq_energy,
                'quantization_noise': quantization_noise,
                'boundary_artifacts': boundary_artifacts,
                'block_variance': np.var(block)
            })
            
            # Update blocking map
            blocking_score = (high_freq_energy + quantization_noise + boundary_artifacts) / 3
            blocking_map[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size] = blocking_score
    
    # Calculate overall statistics
    high_freq_energies = [b['high_freq_energy'] for b in block_artifacts]
    quantization_noises = [b['quantization_noise'] for b in block_artifacts]
    boundary_artifacts_list = [b['boundary_artifacts'] for b in block_artifacts]
    
    # Detect outlier blocks
    outlier_blocks = []
    
    # High frequency outliers
    hf_outliers = detect_outliers_iqr(np.array(high_freq_energies))
    qn_outliers = detect_outliers_iqr(np.array(quantization_noises))
    ba_outliers = detect_outliers_iqr(np.array(boundary_artifacts_list))
    
    all_outlier_indices = set(hf_outliers) | set(qn_outliers) | set(ba_outliers)
    
    for idx in all_outlier_indices:
        outlier_blocks.append(block_artifacts[idx])
    
    return {
        'block_artifacts': block_artifacts,
        'blocking_map': blocking_map,
        'outlier_blocks': outlier_blocks,
        'overall_blocking_score': np.mean(blocking_map),
        'blocking_variance': np.var(blocking_map),
        'high_freq_consistency': np.std(high_freq_energies) / (np.mean(high_freq_energies) + 1e-6),
        'quantization_consistency': np.std(quantization_noises) / (np.mean(quantization_noises) + 1e-6)
    }

def estimate_quantization_noise(dct_block):
    """Estimate quantization noise in DCT block"""
    # Standard JPEG quantization patterns
    # Look for energy concentration in specific DCT coefficients
    
    # Low frequency coefficients (0-2, 0-2)
    low_freq = np.sum(np.abs(dct_block[0:3, 0:3]))
    
    # Medium frequency coefficients
    mid_freq = np.sum(np.abs(dct_block[3:5, 3:5]))
    
    # High frequency coefficients
    high_freq = np.sum(np.abs(dct_block[5:, 5:]))
    
    # Quantization noise is estimated based on the distribution
    # of energy across frequency bands
    total_energy = low_freq + mid_freq + high_freq
    
    if total_energy == 0:
        return 0
    
    # Calculate frequency distribution
    low_ratio = low_freq / total_energy
    mid_ratio = mid_freq / total_energy
    high_ratio = high_freq / total_energy
    
    # Quantization noise score based on unnatural frequency distribution
    # Natural images have smooth frequency roll-off
    # Heavy quantization creates sharp cut-offs
    
    expected_low = 0.7  # Natural images have most energy in low frequencies
    expected_mid = 0.2
    expected_high = 0.1
    
    noise_score = (
        abs(low_ratio - expected_low) +
        abs(mid_ratio - expected_mid) +
        abs(high_ratio - expected_high)
    )
    
    return noise_score

# ======================= Double JPEG Detection =======================

def detect_double_jpeg(image_pil, quality_range=(50, 95, 5)):
    """Detect double JPEG compression"""
    print("  Detecting double JPEG compression...")
    
    start_q, end_q, step_q = quality_range
    qualities = list(range(start_q, end_q + 1, step_q))
    
    # Perform JPEG ghost analysis
    ghost_map, suspicious_map, ghost_analysis = jpeg_ghost_analysis(image_pil, qualities)
    
    # Additional double compression indicators
    double_compression_score = 0
    indicators = []
    
    # 1. Ghost pattern strength
    ghost_strength = ghost_analysis['total_ghost_score']
    if ghost_strength > 0.1:
        double_compression_score += 30
        indicators.append(f"Strong ghost patterns detected (score: {ghost_strength:.3f})")
    
    # 2. Multiple ghost regions
    num_ghost_regions = len(ghost_analysis['ghost_regions'])
    if num_ghost_regions > 3:
        double_compression_score += 20
        indicators.append(f"Multiple ghost regions found ({num_ghost_regions} regions)")
    
    # 3. Quality-specific responses
    quality_analysis = ghost_analysis['quality_analysis']
    
    # Look for quality levels with unusually low response
    low_response_qualities = []
    for quality, analysis in quality_analysis.items():
        if analysis['low_response_area'] > 0.2:  # 20% of image has low response
            low_response_qualities.append(quality)
    
    if len(low_response_qualities) >= 2:
        double_compression_score += 25
        indicators.append(f"Low response at qualities: {low_response_qualities}")
    
    # 4. Block-wise analysis
    block_analysis = analyze_jpeg_blocks(image_pil)
    
    if block_analysis['blocking_variance'] > 100:  # High variance in blocking artifacts
        double_compression_score += 15
        indicators.append(f"High blocking variance detected: {block_analysis['blocking_variance']:.1f}")
    
    # 5. Frequency domain analysis
    freq_analysis = analyze_double_compression_frequency(image_pil)
    if freq_analysis['double_compression_indicator'] > 0.5:
        double_compression_score += 20
        indicators.append(f"Frequency domain anomalies: {freq_analysis['double_compression_indicator']:.3f}")
    
    # Determine confidence level
    if double_compression_score >= 70:
        confidence = "Very High"
    elif double_compression_score >= 50:
        confidence = "High"
    elif double_compression_score >= 30:
        confidence = "Medium"
    elif double_compression_score >= 15:
        confidence = "Low"
    else:
        confidence = "Very Low"
    
    return {
        'double_compression_score': min(double_compression_score, 100),
        'confidence': confidence,
        'indicators': indicators,
        'ghost_analysis': ghost_analysis,
        'block_analysis': block_analysis,
        'frequency_analysis': freq_analysis,
        'is_double_compressed': double_compression_score >= 30
    }

def analyze_double_compression_frequency(image_pil):
    """Analyze frequency domain for double compression artifacts"""
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    image_array = np.array(image_pil)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # DCT analysis
    dct_coeffs = cv2.dct(gray_image.astype(np.float32))
    
    # Analyze DCT coefficient distribution
    # Double compression creates specific patterns in DCT domain
    
    # 1. Histogram analysis of DCT coefficients
    dct_flat = dct_coeffs.flatten()
    hist, bins = np.histogram(dct_flat, bins=100, range=(-50, 50))
    
    # Look for periodic patterns in histogram (indication of quantization)
    hist_smoothed = ndimage.gaussian_filter1d(hist.astype(float), sigma=1)
    hist_diff = np.diff(hist_smoothed)
    
    # Count zero crossings (peaks and valleys)
    zero_crossings = np.sum(np.diff(np.signbit(hist_diff)))
    
    # High number of zero crossings indicates complex quantization pattern
    # which can be a sign of double compression
    double_compression_indicator = min(zero_crossings / 20.0, 1.0)  # Normalize to 0-1
    
    # 2. Block-wise DCT consistency
    h, w = gray_image.shape
    block_size = 8
    blocks_h = h // block_size
    blocks_w = w // block_size
    
    dct_variances = []
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            block = gray_image[i*block_size:(i+1)*block_size,
                              j*block_size:(j+1)*block_size]
            block_dct = cv2.dct(block.astype(np.float32))
            
            # Calculate variance of AC coefficients
            ac_coeffs = block_dct.copy()
            ac_coeffs[0, 0] = 0  # Remove DC component
            dct_variance = np.var(ac_coeffs)
            dct_variances.append(dct_variance)
    
    # Inconsistency in DCT variance across blocks
    dct_variance_consistency = np.std(dct_variances) / (np.mean(dct_variances) + 1e-6)
    
    return {
        'double_compression_indicator': double_compression_indicator,
        'dct_variance_consistency': dct_variance_consistency,
        'zero_crossings': zero_crossings,
        'histogram_complexity': zero_crossings
    }

# ======================= Comprehensive JPEG Analysis =======================

def comprehensive_jpeg_analysis(image_pil):
    """Perform comprehensive JPEG analysis combining all methods"""
    print("🔍 Performing comprehensive JPEG analysis...")
    
    results = {}
    
    # 1. Basic JPEG analysis
    print("  - Basic JPEG artifact analysis...")
    results['basic_analysis'] = advanced_jpeg_analysis(image_pil)
    
    # 2. JPEG ghost analysis
    print("  - JPEG ghost detection...")
    ghost_map, suspicious_map, ghost_analysis = jpeg_ghost_analysis(image_pil)
    results['ghost_map'] = ghost_map
    results['suspicious_map'] = suspicious_map
    results['ghost_analysis'] = ghost_analysis
    
    # 3. Block-wise analysis
    print("  - Block-wise artifact analysis...")
    results['block_analysis'] = analyze_jpeg_blocks(image_pil)
    
    # 4. Double compression detection
    print("  - Double compression detection...")
    results['double_compression'] = detect_double_jpeg(image_pil)
    
    # 5. Overall JPEG score calculation
    results['overall_score'] = calculate_overall_jpeg_score(results)
    
    print("  ✅ Comprehensive JPEG analysis completed")
    
    return results

def calculate_overall_jpeg_score(jpeg_results):
    """Calculate overall JPEG manipulation score"""
    score = 0
    max_score = 100
    indicators = []
    
    # Basic analysis contribution (30%)
    basic = jpeg_results['basic_analysis']
    if basic['compression_inconsistency']:
        score += 20
        indicators.append("JPEG compression inconsistency detected")
    
    if basic['response_variance'] > 50:
        score += 10
        indicators.append(f"High response variance: {basic['response_variance']:.1f}")
    
    # Ghost analysis contribution (25%)
    ghost = jpeg_results['ghost_analysis']
    ghost_score = ghost['total_ghost_score'] * 25
    score += min(ghost_score, 25)
    
    if ghost_score > 5:
        indicators.append(f"JPEG ghost patterns detected (score: {ghost_score:.1f})")
    
    # Block analysis contribution (20%)
    blocks = jpeg_results['block_analysis']
    if blocks['blocking_variance'] > 100:  # High variance in blocking artifacts
        score += 15
        indicators.append(f"High blocking variance: {blocks['blocking_variance']:.1f}")
    
    if len(blocks['outlier_blocks']) > 5:
        score += 5
        indicators.append(f"Multiple outlier blocks: {len(blocks['outlier_blocks'])}")
    
    # Double compression contribution (25%)
    double_comp = jpeg_results['double_compression']
    double_score = (double_comp['double_compression_score'] / 100) * 25
    score += double_score
    
    if double_comp['is_double_compressed']:
        indicators.append(f"Double compression detected ({double_comp['confidence']} confidence)")
    
    # Normalize score
    final_score = min(score, max_score)
    
    # Determine overall assessment
    if final_score >= 70:
        assessment = "Highly Suspicious"
    elif final_score >= 50:
        assessment = "Suspicious"
    elif final_score >= 30:
        assessment = "Moderately Suspicious"
    elif final_score >= 15:
        assessment = "Slightly Suspicious"
    else:
        assessment = "Normal"
    
    return {
        'overall_score': final_score,
        'assessment': assessment,
        'indicators': indicators,
        'confidence_level': get_confidence_level(final_score)
    }

def get_confidence_level(score):
    """Get confidence level based on score"""
    if score >= 80:
        return "Very High"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Medium"
    elif score >= 20:
        return "Low"
    else:
        return "Very Low"

# ======================= Utility Functions =======================

def visualize_jpeg_analysis(image_pil, jpeg_results, output_filename="jpeg_analysis.png"):
    """Create visualization of JPEG analysis results"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive JPEG Analysis Results', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image_pil)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ghost map
    if 'ghost_map' in jpeg_results:
        im1 = axes[0, 1].imshow(jpeg_results['ghost_map'], cmap='hot')
        axes[0, 1].set_title('JPEG Ghost Map')
        axes[0, 1].axis('off')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Blocking map
    if 'block_analysis' in jpeg_results:
        im2 = axes[0, 2].imshow(jpeg_results['block_analysis']['blocking_map'], cmap='viridis')
        axes[0, 2].set_title('Blocking Artifacts Map')
        axes[0, 2].axis('off')
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Quality response curve
    if 'basic_analysis' in jpeg_results:
        quality_responses = jpeg_results['basic_analysis'].get('quality_responses', [])
        if quality_responses:
            qualities = [r['quality'] for r in quality_responses]
            responses = [r['response_mean'] for r in quality_responses]
            axes[1, 0].plot(qualities, responses, 'b-o', label='Mean Response')
            axes[1, 0].set_title('Quality Response Curve')
            axes[1, 0].set_xlabel('JPEG Quality')
            axes[1, 0].set_ylabel('Mean Error Response')
            axes[1, 0].grid(True)
            est_q = jpeg_results['basic_analysis'].get('estimated_original_quality', 0)
            if est_q:
                axes[1, 0].axvline(x=est_q, color='r', linestyle='--', label=f'Est. Quality: {est_q}')
            axes[1, 0].legend()

    # Double compression indicators
    if 'double_compression' in jpeg_results:
        double_comp = jpeg_results['double_compression']
        score = double_comp['double_compression_score']
        color = 'darkred' if score > 50 else 'darkorange' if score > 30 else 'darkblue'
        axes[1, 1].bar(['Double Compression Score'], [score], color=color)
        axes[1, 1].text(0, score + 2, f'{score:.1f}%', ha='center')
        axes[1, 1].set_title('Double Compression Detection')
        axes[1, 1].set_ylabel('Confidence Score')
        axes[1, 1].set_ylim(0, 105)

    # Overall summary
    axes[1, 2].axis('off')
    if 'overall_score' in jpeg_results:
        summary = jpeg_results['overall_score']
        summary_text = f"""**JPEG Analysis Summary**

**Overall Score**: {summary['overall_score']:.1f}/100
**Assessment**: {summary['assessment']}
**Confidence**: {summary['confidence_level']}

**Key Indicators**:
"""
        
        for indicator in summary['indicators'][:4]:
            summary_text += f"\n• {indicator[:45]}..." if len(indicator) > 45 else f"\n• {indicator}"
        
        axes[1, 2].text(0.0, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.9),
                        fontfamily='monospace')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"📊 JPEG analysis visualization saved as '{output_filename}'")
    return output_filename


# ======================= Export Functions =======================

def export_jpeg_analysis_report(jpeg_results, output_filename="jpeg_analysis_report.txt"):
    """Export detailed JPEG analysis report to text file"""
    
    report_content = f"""JPEG ANALYSIS DETAILED REPORT
{'='*50}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
{'-'*20}
Overall Score: {jpeg_results.get('overall_score', {}).get('overall_score', 0):.1f}/100
Assessment: {jpeg_results.get('overall_score', {}).get('assessment', 'Unknown')}
Confidence Level: {jpeg_results.get('overall_score', {}).get('confidence_level', 'Unknown')}

"""

    # Basic Analysis
    if 'basic_analysis' in jpeg_results:
        basic = jpeg_results['basic_analysis']
        report_content += f"""BASIC JPEG ANALYSIS:
{'-'*20}
Estimated Original Quality: {basic.get('estimated_original_quality', 'Unknown')}
Response Variance: {basic.get('response_variance', 0):.2f}
Double Compression Indicator: {basic.get('double_compression_indicator', 0):.3f}
Compression Inconsistency: {basic.get('compression_inconsistency', False)}

Quality Response Analysis:
"""
        
        if basic.get('quality_responses'):
            for qr in basic['quality_responses']:
                report_content += f"  Quality {qr['quality']}: Response {qr['response_mean']:.2f}\n"
    
    # Ghost Analysis
    if 'ghost_analysis' in jpeg_results:
        ghost = jpeg_results['ghost_analysis']
        report_content += f"""
JPEG GHOST ANALYSIS:
{'-'*20}
Ghost Coverage: {ghost.get('ghost_coverage', 0):.1%}
Ghost Intensity: {ghost.get('ghost_intensity', 0):.3f}
Total Ghost Score: {ghost.get('total_ghost_score', 0):.3f}
Number of Ghost Regions: {len(ghost.get('ghost_regions', []))}

"""
        
        if ghost.get('ghost_regions'):
            report_content += "Ghost Regions Detail:\n"
            for i, region in enumerate(ghost['ghost_regions'][:5]):  # Top 5 regions
                report_content += f"  Region {i+1}: Size={region['size']}, Strength={region['ghost_strength']:.3f}\n"
    
    # Block Analysis
    if 'block_analysis' in jpeg_results:
        blocks = jpeg_results['block_analysis']
        report_content += f"""
BLOCK ANALYSIS:
{'-'*20}
Overall Blocking Score: {blocks.get('overall_blocking_score', 0):.3f}
Blocking Variance: {blocks.get('blocking_variance', 0):.1f}
High Frequency Consistency: {blocks.get('high_freq_consistency', 0):.3f}
Quantization Consistency: {blocks.get('quantization_consistency', 0):.3f}
Outlier Blocks: {len(blocks.get('outlier_blocks', []))}

"""
    
    # Double Compression
    if 'double_compression' in jpeg_results:
        double_comp = jpeg_results['double_compression']
        report_content += f"""
DOUBLE COMPRESSION ANALYSIS:
{'-'*20}
Double Compression Score: {double_comp.get('double_compression_score', 0):.1f}/100
Confidence: {double_comp.get('confidence', 'Unknown')}
Is Double Compressed: {double_comp.get('is_double_compressed', False)}

Indicators:
"""
        
        for indicator in double_comp.get('indicators', []):
            report_content += f"  • {indicator}\n"
    
    # Overall Indicators
    if 'overall_score' in jpeg_results and 'indicators' in jpeg_results['overall_score']:
        report_content += f"""
OVERALL INDICATORS:
{'-'*20}
"""
        for indicator in jpeg_results['overall_score']['indicators']:
            report_content += f"• {indicator}\n"
    
    report_content += f"""

TECHNICAL NOTES:
{'-'*20}
• Analysis performed using multi-quality JPEG testing
• Ghost detection uses differential compression analysis
• Block analysis examines 8x8 DCT coefficient patterns
• Double compression detection combines multiple indicators
• Scores are calibrated based on empirical testing

END OF REPORT
{'='*50}
"""
    
    # Save report
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📄 JPEG analysis report saved as '{output_filename}'")
    return output_filename