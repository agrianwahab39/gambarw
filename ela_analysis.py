"""
Error Level Analysis (ELA) functions
"""

import os
import numpy as np
from PIL import Image, ImageChops, ImageStat
from config import ELA_QUALITIES, ELA_SCALE_FACTOR
from utils import detect_outliers_iqr

def perform_multi_quality_ela(image_pil, qualities=ELA_QUALITIES, scale_factor=ELA_SCALE_FACTOR):
    """Multi-quality ELA dengan analisis cross-quality"""
    temp_filename = "temp_ela_multi.jpg"
    
    if image_pil.mode != 'RGB':
        image_rgb = image_pil.convert('RGB')
    else:
        image_rgb = image_pil
    
    ela_results = []
    quality_stats = []
    
    for q in qualities:
        # Save and reload
        image_rgb.save(temp_filename, 'JPEG', quality=q)
        with Image.open(temp_filename) as compressed_rgb:
            # Calculate difference
            diff_rgb = ImageChops.difference(image_rgb, compressed_rgb)
            diff_l = diff_rgb.convert('L')
            ela_np = np.array(diff_l, dtype=float)

        # Scale
        scaled_ela = np.clip(ela_np * scale_factor, 0, 255)
        ela_results.append(scaled_ela)
        
        # Statistics for this quality
        stat = ImageStat.Stat(Image.fromarray(scaled_ela.astype(np.uint8)))
        quality_stats.append({
            'quality': q,
            'mean': stat.mean[0],
            'stddev': stat.stddev[0],
            'max': np.max(scaled_ela),
            'percentile_95': np.percentile(scaled_ela, 95)
        })
    
    # Cross-quality analysis
    ela_variance = np.var(ela_results, axis=0)
    
    # Final ELA (weighted average)
    weights = [0.2, 0.3, 0.3, 0.2]  # Give more weight to mid-qualities
    final_ela = np.average(ela_results, axis=0, weights=weights)
    final_ela_image = Image.fromarray(final_ela.astype(np.uint8), mode='L')
    
    # Enhanced regional analysis
    regional_stats = analyze_ela_regions_enhanced(final_ela, ela_variance)
    
    # Overall statistics
    final_stat = ImageStat.Stat(final_ela_image)
    
    try:
        os.remove(temp_filename)
    except:
        pass
    
    return (final_ela_image, final_stat.mean[0], final_stat.stddev[0],
            regional_stats, quality_stats, ela_variance)

def analyze_ela_regions_enhanced(ela_array, ela_variance, block_size=32):
    """Enhanced regional ELA analysis"""
    h, w = ela_array.shape
    regional_means = []
    regional_stds = []
    regional_variances = []
    suspicious_regions = []
    
    for i in range(0, h - block_size, block_size//2):
        for j in range(0, w - block_size, block_size//2):
            block = ela_array[i:i+block_size, j:j+block_size]
            var_block = ela_variance[i:i+block_size, j:j+block_size]
            
            block_mean = np.mean(block)
            block_std = np.std(block)
            block_var = np.mean(var_block)
            
            regional_means.append(block_mean)
            regional_stds.append(block_std)
            regional_variances.append(block_var)
            
            # Detect suspicious regions
            if block_mean > 15 or block_std > 25 or block_var > 100:
                suspicious_regions.append({
                    'position': (i, j),
                    'mean': block_mean,
                    'std': block_std,
                    'variance': block_var
                })
    
    # Statistical analysis
    regional_means = np.array(regional_means)
    regional_stds = np.array(regional_stds)
    
    return {
        'mean_variance': np.var(regional_means),
        'std_variance': np.var(regional_stds),
        'outlier_regions': len(detect_outliers_iqr(regional_means)) + len(detect_outliers_iqr(regional_stds)),
        'regional_inconsistency': np.std(regional_means) / (np.mean(regional_means) + 1e-6),
        'suspicious_regions': suspicious_regions,
        'cross_quality_variance': np.mean(regional_variances)
    }