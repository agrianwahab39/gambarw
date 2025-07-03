# --- START OF FILE advanced_analysis.py ---
"""
Advanced Analysis Module for Forensic Image Analysis System
Contains functions for noise, frequency, texture, edge, illumination, and statistical analysis
"""

import numpy as np
import cv2
try:
    from scipy import ndimage
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    class ndimage:
        @staticmethod
        def gaussian_filter(a, sigma):
            # Implement proper fallback using OpenCV's GaussianBlur
            # Convert sigma to kernel size (must be odd)
            ksize = int(2 * round(sigma * 3) + 1)
            ksize = max(3, ksize)  # Ensure minimum size of 3
            
            # Handle different array dimensions
            if len(a.shape) == 2:
                return cv2.GaussianBlur(a.astype(np.float32), (ksize, ksize), sigma).astype(a.dtype)
            elif len(a.shape) == 3:
                result = np.zeros_like(a)
                for i in range(a.shape[2]):
                    result[:,:,i] = cv2.GaussianBlur(a[:,:,i].astype(np.float32), (ksize, ksize), sigma).astype(a.dtype)
                return result
            else:
                return a  # Fallback for unsupported dimensions
    def entropy(arr):
        hist, _ = np.histogram(arr, bins=256, range=(0,255), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
import warnings

# Conditional imports dengan error handling
try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.filters import sobel, prewitt, roberts
    from skimage.measure import shannon_entropy
    SKIMAGE_AVAILABLE = True
except Exception:
    print("Warning: scikit-image not available. Some features will be limited.")
    SKIMAGE_AVAILABLE = False

# Import utilities dengan error handling
try:
    from utils import detect_outliers_iqr
except ImportError:
    print("Warning: utils module not found. Using fallback functions.")
    def detect_outliers_iqr(data, factor=1.5):
        """Fallback outlier detection"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return np.where((data < lower_bound) | (data > upper_bound))[0]

warnings.filterwarnings('ignore')

# ======================= Helper Functions =======================

def calculate_skewness(data):
    """Calculate skewness"""
    try:
        if len(data) == 0 or np.std(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        return float(np.mean(((data - mean) / std) ** 3))
    except Exception:
        return 0.0

def calculate_kurtosis(data):
    """Calculate kurtosis"""
    try:
        if len(data) == 0 or np.std(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        return float(np.mean(((data - mean) / std) ** 4) - 3)
    except Exception:
        return 0.0

def safe_entropy(data):
    """Safe entropy calculation with fallback"""
    try:
        if data.size == 0:
            return 0.0
        if SKIMAGE_AVAILABLE:
            return float(shannon_entropy(data))
        else:
            # Fallback entropy calculation
            hist, _ = np.histogram(data.flatten(), bins=256, range=(0, 255))
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]  # Remove zeros
            return float(-np.sum(hist * np.log2(hist + 1e-10)))
    except Exception:
        return 0.0

# ======================= Noise Analysis =======================

def analyze_noise_consistency(image_pil, block_size=32):
    """Advanced noise consistency analysis"""
    print("  - Advanced noise consistency analysis...")
    
    try:
        image_array = np.array(image_pil.convert('RGB'))
        
        # Convert to different color spaces for comprehensive analysis
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        
        h, w, c = image_array.shape
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        noise_characteristics = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                # Safe block extraction
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                rgb_block = image_array[y_start:y_end, x_start:x_end]
                lab_block = lab[y_start:y_end, x_start:x_end]
                
                if rgb_block.size == 0: # Skip empty blocks if they occur
                    continue

                # Noise estimation using Laplacian variance
                gray_block = cv2.cvtColor(rgb_block, cv2.COLOR_RGB2GRAY)
                try:
                    if gray_block.size == 0 or np.all(gray_block == gray_block[0,0]): # Skip if block is uniform
                        laplacian_var = 0.0
                    else:
                        laplacian = cv2.Laplacian(gray_block, cv2.CV_64F)
                        laplacian_var = laplacian.var()
                except Exception as cv_err:
                    # Fallback manual Laplacian
                    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                    if gray_block.ndim == 2 and gray_block.shape[0] >= 3 and gray_block.shape[1] >= 3:
                        laplacian = cv2.filter2D(gray_block.astype(np.float64), -1, kernel)
                        laplacian_var = laplacian.var()
                    else:
                        laplacian_var = 0.0 # Cannot compute for too small blocks
                
                # High frequency content analysis with safe indexing
                try:
                    f_transform = np.fft.fft2(gray_block)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
                    
                    # Safe frequency range calculation
                    h_block, w_block = magnitude_spectrum.shape
                    quarter_h, quarter_w = max(1, h_block//4), max(1, w_block//4)
                    three_quarter_h = min(h_block, 3*h_block//4)
                    three_quarter_w = min(w_block, 3*w_block//4)
                    
                    if three_quarter_h > quarter_h and three_quarter_w > quarter_w:
                        high_freq_energy = np.sum(magnitude_spectrum[quarter_h:three_quarter_h, 
                                                                   quarter_w:three_quarter_w])
                    else:
                        high_freq_energy = np.sum(magnitude_spectrum)
                except Exception:
                    high_freq_energy = 0.0
                
                # Color noise analysis
                rgb_std = np.std(rgb_block, axis=(0, 1)) if rgb_block.size > 0 else [0.0, 0.0, 0.0]
                lab_std = np.std(lab_block, axis=(0, 1)) if lab_block.size > 0 else [0.0, 0.0, 0.0]
                
                # Statistical moments
                mean_intensity = np.mean(gray_block) if gray_block.size > 0 else 0.0
                std_intensity = np.std(gray_block) if gray_block.size > 0 else 0.0
                skewness = calculate_skewness(gray_block.flatten())
                kurtosis = calculate_kurtosis(gray_block.flatten())
                
                noise_characteristics.append({
                    'position': (i, j),
                    'laplacian_var': float(laplacian_var),
                    'high_freq_energy': float(high_freq_energy),
                    'rgb_std': rgb_std.tolist(),
                    'lab_std': lab_std.tolist(),
                    'mean_intensity': float(mean_intensity),
                    'std_intensity': float(std_intensity),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis)
                })
        
        # Analyze consistency across blocks
        if noise_characteristics:
            laplacian_vars = [block['laplacian_var'] for block in noise_characteristics]
            high_freq_energies = [block['high_freq_energy'] for block in noise_characteristics]
            std_intensities = [block['std_intensity'] for block in noise_characteristics]
            
            # Filter out zero/nan values to prevent ZeroDivisionError
            laplacian_vars_filtered = [v for v in laplacian_vars if v != 0 and not np.isnan(v)]
            high_freq_energies_filtered = [v for v in high_freq_energies if v != 0 and not np.isnan(v)]
            std_intensities_filtered = [v for v in std_intensities if v != 0 and not np.isnan(v)]

            laplacian_consistency = np.std(laplacian_vars_filtered) / (np.mean(laplacian_vars_filtered) + 1e-6) if laplacian_vars_filtered else 0.0
            freq_consistency = np.std(high_freq_energies_filtered) / (np.mean(high_freq_energies_filtered) + 1e-6) if high_freq_energies_filtered else 0.0
            intensity_consistency = np.std(std_intensities_filtered) / (np.mean(std_intensities_filtered) + 1e-6) if std_intensities_filtered else 0.0
            
            # Overall inconsistency score
            overall_inconsistency = (laplacian_consistency + freq_consistency + intensity_consistency) / 3
            
            # Detect outlier blocks with error handling
            outliers = []
            try:
                # Convert to numpy array for outlier detection
                if laplacian_vars:
                    outlier_indices = detect_outliers_iqr(np.array(laplacian_vars))
                    for idx in outlier_indices:
                        if idx < len(noise_characteristics):
                            outliers.append(noise_characteristics[idx])
            except Exception:
                pass # Continue if outlier detection fails

            if not np.isfinite(laplacian_consistency): laplacian_consistency = 0.0
            if not np.isfinite(freq_consistency): freq_consistency = 0.0
            if not np.isfinite(intensity_consistency): intensity_consistency = 0.0
            if not np.isfinite(overall_inconsistency): overall_inconsistency = 0.0
            
        else:
            laplacian_consistency = 0.0
            freq_consistency = 0.0
            intensity_consistency = 0.0
            overall_inconsistency = 0.0
            outliers = []
        
        return {
            'noise_characteristics': noise_characteristics,
            'laplacian_consistency': float(laplacian_consistency),
            'frequency_consistency': float(freq_consistency),
            'intensity_consistency': float(intensity_consistency),
            'overall_inconsistency': float(overall_inconsistency),
            'outlier_blocks': outliers,
            'outlier_count': len(outliers)
        }
        
    except Exception as e:
        print(f"  Warning: Noise analysis failed: {e}")
        return {
            'noise_characteristics': [],
            'laplacian_consistency': 0.0,
            'frequency_consistency': 0.0,
            'intensity_consistency': 0.0,
            'overall_inconsistency': 0.0,
            'outlier_blocks': [],
            'outlier_count': 0
        }

# ======================= Frequency Domain Analysis =======================

def analyze_frequency_domain(image_pil):
    """Analyze DCT coefficients for manipulation detection"""
    try:
        image_array = np.array(image_pil.convert('L'))
        
        # DCT Analysis dengan multiple fallback methods
        dct_coeffs = None
        
        # Method 1: OpenCV DCT
        try:
            dct_coeffs = cv2.dct(image_array.astype(np.float32))
        except Exception:
            pass
        
        # Method 2: SciPy DCT fallback
        if dct_coeffs is None:
            if SCIPY_AVAILABLE:
                try:
                    from scipy.fft import dctn
                    dct_coeffs = dctn(image_array.astype(np.float64), type=2, norm='ortho')
                except Exception:
                    pass
        
        # Method 3: NumPy FFT fallback
        if dct_coeffs is None:
            try:
                # For consistency, use log of magnitude for visual comparison
                f_transform = np.fft.fft2(image_array)
                f_shift = np.fft.fftshift(f_transform)
                dct_coeffs = np.log(np.abs(f_shift) + 1)
            except Exception:
                dct_coeffs = np.zeros_like(image_array, dtype=np.float32)
        
        h, w = dct_coeffs.shape
        
        # Safe region calculation
        # Ensure dimensions are large enough to avoid negative indices or zero ranges
        low_h, low_w = min(16, h), min(16, w)
        mid_h_start, mid_w_start = min(8, h - 1), min(8, w - 1)
        mid_h_end, mid_w_end = min(24, h), min(24, w)

        dct_stats = {
            'low_freq_energy': float(np.sum(np.abs(dct_coeffs[:low_h, :low_w]))),
            'high_freq_energy': float(np.sum(np.abs(dct_coeffs[low_h:, low_w:]))),
            'mid_freq_energy': float(np.sum(np.abs(dct_coeffs[mid_h_start:mid_h_end, mid_w_start:mid_w_end]))),
        }
        
        dct_stats['freq_ratio'] = dct_stats['high_freq_energy'] / (dct_stats['low_freq_energy'] + 1e-6)
        
        # Block-wise DCT analysis
        block_size = 8
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        block_freq_variations = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                block = image_array[y_start:y_end, x_start:x_end]
                
                try:
                    # Only attempt DCT if block is valid
                    if block.shape[0] == block_size and block.shape[1] == block_size:
                        block_dct = cv2.dct(block.astype(np.float32))
                        block_energy = np.sum(np.abs(block_dct))
                    else: # Handle partial blocks or too-small images
                        block_energy = np.sum(np.abs(block.astype(np.float32))) # Sum magnitude for consistency
                except Exception:
                    block_energy = np.sum(np.abs(block.astype(np.float32))) # Fallback for other errors
                
                block_freq_variations.append(float(block_energy))
        
        # Calculate frequency inconsistency
        if len(block_freq_variations) > 0 and np.mean(block_freq_variations) != 0:
            freq_inconsistency = np.std(block_freq_variations) / (np.mean(block_freq_variations) + 1e-6)
        else:
            freq_inconsistency = 0.0

        if not np.isfinite(freq_inconsistency): freq_inconsistency = 0.0

        return {
            'dct_stats': dct_stats,
            'frequency_inconsistency': float(freq_inconsistency),
            'block_variations': float(np.var(block_freq_variations)) if block_freq_variations else 0.0
        }
        
    except Exception as e:
        print(f"  Warning: Frequency analysis failed: {e}")
        return {
            'dct_stats': {
                'low_freq_energy': 0.0,
                'high_freq_energy': 0.0,
                'mid_freq_energy': 0.0,
                'freq_ratio': 0.0
            },
            'frequency_inconsistency': 0.0,
            'block_variations': 0.0
        }

# ======================= Texture Analysis =======================

def analyze_texture_consistency(image_pil, block_size=64):
    """Analyze texture consistency using GLCM and LBP"""
    try:
        image_gray = np.array(image_pil.convert('L'))
        
        # Local Binary Pattern analysis dengan fallback (moved to internal block for localized LBP)
        
        # Block-wise texture analysis
        h, w = image_gray.shape
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        texture_features = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                block = image_gray[y_start:y_end, x_start:x_end]
                if block.size == 0 or block.shape[0] < 2 or block.shape[1] < 2: # Skip too small blocks
                    continue

                # GLCM analysis dengan fallback
                if SKIMAGE_AVAILABLE:
                    try:
                        # Ensure enough levels for GLCM
                        max_level = np.max(block)
                        levels = min(max_level + 1, 256) if max_level is not None else 256

                        # For small blocks, distances and angles might need to be adapted,
                        # or reduce levels if many bins are empty.
                        if levels > 1: # graycomatrix requires more than 1 unique gray level
                            glcm = graycomatrix(block, distances=[1], angles=[0, 45, 90, 135],
                                            levels=levels, symmetric=True, normed=True)
                            
                            contrast = graycoprops(glcm, 'contrast')[0, 0]
                            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                            energy = graycoprops(glcm, 'energy')[0, 0]
                        else: # Uniform block
                             contrast = 0.0
                             dissimilarity = 0.0
                             homogeneity = 1.0
                             energy = 1.0
                    except Exception:
                        # Fallback measures
                        contrast = float(np.var(block))
                        dissimilarity = float(np.std(block))
                        homogeneity = 1.0 / (1.0 + np.var(block)) if np.var(block) != 0 else 1.0
                        energy = float(np.mean(block ** 2) / 255**2)
                else:
                    # Fallback measures
                    contrast = float(np.var(block))
                    dissimilarity = float(np.std(block))
                    homogeneity = 1.0 / (1.0 + np.var(block)) if np.var(block) != 0 else 1.0
                    energy = float(np.mean(block ** 2) / 255**2)
                
                # LBP calculation for block
                radius = 1 # Use smaller radius for blocks
                n_points = 8 * radius
                lbp_value = 0.0
                if SKIMAGE_AVAILABLE:
                    try:
                        # Ensure block size is sufficient for LBP calculation (at least 3x3 for radius 1)
                        if block.shape[0] >= (2 * radius + 1) and block.shape[1] >= (2 * radius + 1):
                            block_lbp = local_binary_pattern(block, n_points, radius, method='uniform')
                            lbp_hist, _ = np.histogram(block_lbp, bins=range(n_points + 3)) # Max bins for uniform LBP is n_points + 2
                            lbp_hist = lbp_hist / np.sum(lbp_hist)
                            lbp_hist = lbp_hist[lbp_hist > 0]
                            if lbp_hist.size > 0:
                                lbp_value = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)) # Entropy of LBP histogram
                        else: # Too small for LBP
                            lbp_value = safe_entropy(block) # Fallback to block pixel entropy
                    except Exception:
                        lbp_value = safe_entropy(block) # Fallback if skimage.feature.local_binary_pattern fails
                else:
                    lbp_value = safe_entropy(block) # Fallback if skimage is not available

                texture_features.append([
                    float(contrast), 
                    float(dissimilarity), 
                    float(homogeneity), 
                    float(energy), 
                    float(lbp_value) # Using lbp_value now
                ])
        
        # Analyze consistency
        texture_consistency = {}
        feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'lbp_uniformity']
        
        if len(texture_features) > 0:
            texture_features = np.array(texture_features)
            for i, name in enumerate(feature_names):
                feature_values = texture_features[:, i]
                # Filter out zeros from mean calculation to prevent large consistency scores for uniform blocks
                filtered_values = [v for v in feature_values if v != 0 and not np.isnan(v)]
                consistency = np.std(filtered_values) / (np.mean(filtered_values) + 1e-6) if filtered_values else 0.0
                if not np.isfinite(consistency): consistency = 0.0
                texture_consistency[f'{name}_consistency'] = float(consistency)
            
            # Use mean of valid consistency scores
            overall_texture_inconsistency = np.mean([val for val in texture_consistency.values() if np.isfinite(val)])
            if not np.isfinite(overall_texture_inconsistency): overall_texture_inconsistency = 0.0

        else:
            for name in feature_names:
                texture_consistency[f'{name}_consistency'] = 0.0
            overall_texture_inconsistency = 0.0
        
        return {
            'texture_consistency': texture_consistency,
            'overall_inconsistency': float(overall_texture_inconsistency),
            'texture_features': texture_features.tolist() if len(texture_features) > 0 else []
        }
        
    except Exception as e:
        print(f"  Warning: Texture analysis failed: {e}")
        feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'lbp_uniformity']
        texture_consistency = {f'{name}_consistency': 0.0 for name in feature_names}
        
        return {
            'texture_consistency': texture_consistency,
            'overall_inconsistency': 0.0,
            'texture_features': []
        }

# ======================= Edge Analysis =======================

def analyze_edge_consistency(image_pil):
    """Analyze edge density consistency"""
    try:
        image_gray = np.array(image_pil.convert('L'))
        if image_gray.size == 0 or image_gray.shape[0] < 3 or image_gray.shape[1] < 3: # Handle too small images
            return {
                'edge_inconsistency': 0.0,
                'edge_densities': [],
                'edge_variance': 0.0
            }

        # Multiple edge detectors dengan fallback
        combined_edges = None

        if SKIMAGE_AVAILABLE:
            try:
                edges_sobel = sobel(image_gray.astype(np.float32))
                edges_prewitt = prewitt(image_gray.astype(np.float32))
                edges_roberts = roberts(image_gray.astype(np.float32))
                combined_edges = (edges_sobel + edges_prewitt + edges_roberts) / 3
            except Exception:
                # Fallback to OpenCV Sobel if skimage fails
                pass # Combined edges will be None, triggering the next fallback

        if combined_edges is None: # Fallback to OpenCV Sobel
            try:
                grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
                combined_edges = np.sqrt(grad_x**2 + grad_y**2)
            except Exception as sobel_err:
                # Manual gradient calculation fallback
                kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                if image_gray.shape[0] >= 3 and image_gray.shape[1] >= 3:
                    grad_x = cv2.filter2D(image_gray.astype(np.float64), -1, kernel_x)
                    grad_y = cv2.filter2D(image_gray.astype(np.float64), -1, kernel_y)
                    combined_edges = np.sqrt(grad_x**2 + grad_y**2)
                else: # Block too small even for manual 3x3 kernel
                    combined_edges = np.zeros_like(image_gray, dtype=np.float32)

        if combined_edges is None: # Last resort fallback
            combined_edges = np.zeros_like(image_gray, dtype=np.float32)

        # Normalize edge map to 0-255 if not already
        if np.max(combined_edges) > 0:
            combined_edges = (combined_edges / np.max(combined_edges)) * 255
        
        # Block-wise edge density
        block_size = 32
        h, w = image_gray.shape
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        edge_densities = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                block_edges = combined_edges[y_start:y_end, x_start:x_end]
                if block_edges.size > 0:
                    edge_density = np.mean(block_edges)
                    edge_densities.append(float(edge_density))
        
        edge_densities = np.array(edge_densities)
        
        if len(edge_densities) > 0 and np.mean(edge_densities) != 0:
            edge_inconsistency = np.std(edge_densities) / (np.mean(edge_densities) + 1e-6)
            edge_variance = np.var(edge_densities)
        else:
            edge_inconsistency = 0.0
            edge_variance = 0.0

        if not np.isfinite(edge_inconsistency): edge_inconsistency = 0.0
        if not np.isfinite(edge_variance): edge_variance = 0.0
        
        return {
            'edge_inconsistency': float(edge_inconsistency),
            'edge_densities': edge_densities.tolist(),
            'edge_variance': float(edge_variance)
        }
        
    except Exception as e:
        print(f"  Warning: Edge analysis failed: {e}")
        return {
            'edge_inconsistency': 0.0,
            'edge_densities': [],
            'edge_variance': 0.0
        }

# ======================= Illumination Analysis =======================

def analyze_illumination_consistency(image_pil):
    """Advanced illumination consistency analysis"""
    try:
        image_array = np.array(image_pil)
        if image_array.size == 0:
            return {
                'illumination_mean_consistency': 0.0,
                'illumination_std_consistency': 0.0,
                'gradient_consistency': 0.0,
                'overall_illumination_inconsistency': 0.0
            }
        
        # Convert to RGB if needed, then to LAB for L channel
        if len(image_array.shape) == 2 or image_array.shape[2] == 1:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB) # Convert grayscale to RGB for consistent LAB conversion
        
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        
        # Illumination map (L channel in LAB)
        illumination = lab[:, :, 0]
        
        # Gradient analysis with robust error handling
        try:
            if illumination.shape[0] >= 3 and illumination.shape[1] >= 3:
                grad_x = cv2.Sobel(illumination, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(illumination, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            else: # Image too small for Sobel
                gradient_magnitude = np.zeros_like(illumination, dtype=np.float32)
        except Exception as sobel_err:
            print(f"  Warning: Sobel operation failed: {sobel_err}")
            # Manual gradient calculation fallback
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            if illumination.shape[0] >= 3 and illumination.shape[1] >= 3:
                grad_x = cv2.filter2D(illumination.astype(np.float64), -1, kernel_x)
                grad_y = cv2.filter2D(illumination.astype(np.float64), -1, kernel_y)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            else: # Image too small for manual kernel
                gradient_magnitude = np.zeros_like(illumination, dtype=np.float32)
        
        # Block-wise illumination analysis
        block_size = 64
        h, w = illumination.shape
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        illumination_means = []
        illumination_stds = []
        gradient_means = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                block_illum = illumination[y_start:y_end, x_start:x_end]
                block_grad = gradient_magnitude[y_start:y_end, x_start:x_end]
                
                if block_illum.size > 0:
                    illumination_means.append(np.mean(block_illum))
                    illumination_stds.append(np.std(block_illum))
                else: # Skip empty blocks
                    illumination_means.append(0.0)
                    illumination_stds.append(0.0)
                
                if block_grad.size > 0:
                    gradient_means.append(np.mean(block_grad))
                else: # Skip empty blocks
                    gradient_means.append(0.0)


        # Consistency metrics
        illum_mean_consistency = 0.0
        illum_std_consistency = 0.0
        gradient_consistency = 0.0
        overall_inconsistency = 0.0

        if len(illumination_means) > 0 and np.mean(illumination_means) != 0:
            illum_mean_consistency = np.std(illumination_means) / (np.mean(illumination_means) + 1e-6)
        if len(illumination_stds) > 0 and np.mean(illumination_stds) != 0:
            illum_std_consistency = np.std(illumination_stds) / (np.mean(illumination_stds) + 1e-6)
        if len(gradient_means) > 0 and np.mean(gradient_means) != 0:
            gradient_consistency = np.std(gradient_means) / (np.mean(gradient_means) + 1e-6)
        
        overall_inconsistency = (illum_mean_consistency + gradient_consistency) / 2
        
        if not np.isfinite(illum_mean_consistency): illum_mean_consistency = 0.0
        if not np.isfinite(illum_std_consistency): illum_std_consistency = 0.0
        if not np.isfinite(gradient_consistency): gradient_consistency = 0.0
        if not np.isfinite(overall_inconsistency): overall_inconsistency = 0.0

        return {
            'illumination_mean_consistency': float(illum_mean_consistency),
            'illumination_std_consistency': float(illum_std_consistency),
            'gradient_consistency': float(gradient_consistency),
            'overall_illumination_inconsistency': float(overall_inconsistency)
        }
        
    except Exception as e:
        print(f"  Warning: Illumination analysis failed: {e}")
        return {
            'illumination_mean_consistency': 0.0,
            'illumination_std_consistency': 0.0,
            'gradient_consistency': 0.0,
            'overall_illumination_inconsistency': 0.0
        }

# ======================= Statistical Analysis =======================

def perform_statistical_analysis(image_pil):
    """Comprehensive statistical analysis"""
    try:
        image_array = np.array(image_pil)
        stats = {}
        
        if image_array.ndim != 3 or image_array.shape[2] not in [3, 4]:
            print("  Warning: Image is not a standard RGB/RGBA image, performing grayscale stats.")
            gray_channel = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if image_array.ndim == 3 else image_array
            
            gray_data = gray_channel.flatten()
            stats['R_mean'] = stats['G_mean'] = stats['B_mean'] = float(np.mean(gray_data)) if gray_data.size > 0 else 0.0
            stats['R_std'] = stats['G_std'] = stats['B_std'] = float(np.std(gray_data)) if gray_data.size > 0 else 0.0
            stats['R_skewness'] = stats['G_skewness'] = stats['B_skewness'] = calculate_skewness(gray_data)
            stats['R_kurtosis'] = stats['G_kurtosis'] = stats['B_kurtosis'] = calculate_kurtosis(gray_data)
            stats['R_entropy'] = stats['G_entropy'] = stats['B_entropy'] = safe_entropy(gray_channel)
            
            stats['rg_correlation'] = 1.0 # Or 0.0, depends on interpretation for grayscale
            stats['rb_correlation'] = 1.0
            stats['gb_correlation'] = 1.0
            stats['overall_entropy'] = safe_entropy(gray_channel)
            return stats

        # Per-channel statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image_array[:, :, i].flatten()
            stats[f'{channel}_mean'] = float(np.mean(channel_data)) if channel_data.size > 0 else 0.0
            stats[f'{channel}_std'] = float(np.std(channel_data)) if channel_data.size > 0 else 0.0
            stats[f'{channel}_skewness'] = calculate_skewness(channel_data)
            stats[f'{channel}_kurtosis'] = calculate_kurtosis(channel_data)
            stats[f'{channel}_entropy'] = safe_entropy(image_array[:, :, i])
        
        # Cross-channel correlation
        r_channel = image_array[:, :, 0].flatten()
        g_channel = image_array[:, :, 1].flatten()
        b_channel = image_array[:, :, 2].flatten()
        
        if r_channel.size > 1 and g_channel.size > 1: # ensure at least two elements for correlation
            rg_corr = np.corrcoef(r_channel, g_channel)[0, 1]
            stats['rg_correlation'] = float(rg_corr if np.isfinite(rg_corr) else 0.0)
        else: stats['rg_correlation'] = 0.0

        if r_channel.size > 1 and b_channel.size > 1:
            rb_corr = np.corrcoef(r_channel, b_channel)[0, 1]
            stats['rb_correlation'] = float(rb_corr if np.isfinite(rb_corr) else 0.0)
        else: stats['rb_correlation'] = 0.0

        if g_channel.size > 1 and b_channel.size > 1:
            gb_corr = np.corrcoef(g_channel, b_channel)[0, 1]
            stats['gb_correlation'] = float(gb_corr if np.isfinite(gb_corr) else 0.0)
        else: stats['gb_correlation'] = 0.0
        
        # Overall statistics
        stats['overall_entropy'] = safe_entropy(image_array)
        
        return stats
        
    except Exception as e:
        print(f"  Warning: Statistical analysis failed: {e}")
        # Return safe defaults
        channels = ['R', 'G', 'B']
        stats = {}
        for ch in channels:
            stats[f'{ch}_mean'] = 0.0
            stats[f'{ch}_std'] = 0.0
            stats[f'{ch}_skewness'] = 0.0
            stats[f'{ch}_kurtosis'] = 0.0
            stats[f'{ch}_entropy'] = 0.0
        
        stats['rg_correlation'] = 0.0
        stats['rb_correlation'] = 0.0
        stats['gb_correlation'] = 0.0
        stats['overall_entropy'] = 0.0
        
        return stats

# --- END OF FILE advanced_analysis.py ---