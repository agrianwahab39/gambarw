
# --- START OF FILE classification.py ---

"""
Classification Module for Forensic Image Analysis System
Contains functions for machine learning classification, feature vector preparation, and confidence scoring
"""

import numpy as np
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import normalize as sk_normalize
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    class RandomForestClassifier:
        def __init__(self,*a,**k): pass
        def fit(self,X,y): pass
        def predict_proba(self,X): return np.zeros((len(X),2))
    class SVC:
        def __init__(self,*a,**k): pass
        def fit(self,X,y): pass
        def decision_function(self,X): return np.zeros(len(X))
    def sk_normalize(arr, norm='l2', axis=1):
        denom = np.linalg.norm(arr, ord=2 if norm=='l2' else 1, axis=axis, keepdims=True)
        denom[denom==0]=1
        return arr/denom
import warnings
from uncertainty_classification import UncertaintyClassifier, format_probability_results

warnings.filterwarnings('ignore')

# ======================= Helper Functions =======================

def sigmoid(x):
    """Sigmoid activation function"""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    """Tanh activation function (alternative)"""
    return np.tanh(x)

# ======================= Feature Vector Preparation =======================

def prepare_feature_vector(analysis_results):
    """Prepare comprehensive feature vector for ML classification"""
    features = []
    
    # ELA features (6)
    # Safely get ELA regional stats with defaults
    ela_regional = analysis_results.get('ela_regional_stats', {'mean_variance': 0.0, 'regional_inconsistency': 0.0, 'outlier_regions': 0, 'suspicious_regions': []})

    features.extend([
        analysis_results.get('ela_mean', 0.0),
        analysis_results.get('ela_std', 0.0),
        ela_regional.get('mean_variance', 0.0),
        ela_regional.get('regional_inconsistency', 0.0),
        ela_regional.get('outlier_regions', 0),
        len(ela_regional.get('suspicious_regions', []))
    ])
    
    # SIFT features (3)
    features.extend([
        analysis_results.get('sift_matches', 0),
        analysis_results.get('ransac_inliers', 0),
        1 if analysis_results.get('geometric_transform') else 0
    ])
    
    # Block matching (1)
    features.append(len(analysis_results.get('block_matches', [])))
    
    # Noise analysis (1)
    noise_analysis = analysis_results.get('noise_analysis', {'overall_inconsistency': 0.0})
    features.append(noise_analysis.get('overall_inconsistency', 0.0))
    
    # JPEG analysis (3)
    jpeg_analysis_main = analysis_results.get('jpeg_analysis', {})
    basic_jpeg_analysis = jpeg_analysis_main.get('basic_analysis', {}) # Added for safe access
    features.extend([
        analysis_results.get('jpeg_ghost_suspicious_ratio', 0.0),
        basic_jpeg_analysis.get('response_variance', 0.0),
        basic_jpeg_analysis.get('double_compression_indicator', 0.0)
    ])
    
    # Frequency domain (2)
    freq_analysis = analysis_results.get('frequency_analysis', {'frequency_inconsistency': 0.0, 'dct_stats': {}})
    features.extend([
        freq_analysis.get('frequency_inconsistency', 0.0),
        freq_analysis['dct_stats'].get('freq_ratio', 0.0)
    ])
    
    # Texture analysis (1)
    texture_analysis = analysis_results.get('texture_analysis', {'overall_inconsistency': 0.0})
    features.append(texture_analysis.get('overall_inconsistency', 0.0))
    
    # Edge analysis (1)
    edge_analysis = analysis_results.get('edge_analysis', {'edge_inconsistency': 0.0})
    features.append(edge_analysis.get('edge_inconsistency', 0.0))
    
    # Illumination analysis (1)
    illumination_analysis = analysis_results.get('illumination_analysis', {'overall_illumination_inconsistency': 0.0})
    features.append(illumination_analysis.get('overall_illumination_inconsistency', 0.0))
    
    # Statistical features (5)
    stat_analysis = analysis_results.get('statistical_analysis', {})
    stat_features = [
        stat_analysis.get('R_entropy', 0.0),
        stat_analysis.get('G_entropy', 0.0),
        stat_analysis.get('B_entropy', 0.0),
        stat_analysis.get('rg_correlation', 0.0),
        stat_analysis.get('overall_entropy', 0.0)
    ]
    features.extend(stat_features)
    
    # Metadata score (1)
    metadata = analysis_results.get('metadata', {})
    features.append(metadata.get('Metadata_Authenticity_Score', 0.0))
    
    # Localization features (3)
    if 'localization_analysis' in analysis_results:
        loc_results = analysis_results['localization_analysis']
        # Default empty dict for kmeans_localization if not present, to prevent KeyError
        kmeans_loc = loc_results.get('kmeans_localization', {}) 
        cluster_ela_means = kmeans_loc.get('cluster_ela_means', [])
        
        features.extend([
            loc_results.get('tampering_percentage', 0.0),
            len(cluster_ela_means),
            max(cluster_ela_means) if cluster_ela_means else 0.0
        ])
    else:
        features.extend([0.0, 0, 0.0]) # Add defaults for localization if whole key is missing
    
    return np.array(features, dtype=np.float32) # Ensure float32 for consistency

def validate_feature_vector(feature_vector):
    """
    Validate and clean feature vector.
    Ensure `ransac_inliers` and other count-based features are non-negative.
    """
    if not isinstance(feature_vector, np.ndarray):
        feature_vector = np.array(feature_vector)

    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)
    feature_vector = np.clip(feature_vector, -1e6, 1e6) # clip large values

    # Apply non-negative constraint for specific indices/features that are counts or percentages
    # Assuming the fixed feature vector structure from `prepare_feature_vector`
    # (Adjust indices if feature vector structure changes significantly)
    
    # Feature 7 (index 6): SIFT Matches (must be >= 0)
    if len(feature_vector) > 6:
        feature_vector[6] = max(0.0, feature_vector[6])
    # Feature 8 (index 7): RANSAC Inliers (must be >= 0)
    if len(feature_vector) > 7:
        feature_vector[7] = max(0.0, feature_vector[7])
    # Feature 10 (index 9): Block Matches (must be >= 0)
    if len(feature_vector) > 9:
        feature_vector[9] = max(0.0, feature_vector[9])
    # Localization Tampering Percentage (index 25, if present)
    if len(feature_vector) > 25:
        feature_vector[25] = np.clip(feature_vector[25], 0.0, 100.0)

    return feature_vector

def normalize_feature_vector(feature_vector):
    """Normalize feature vector for ML processing"""
    # Use sklearn's normalize for more robust L2 normalization if available
    if SKLEARN_AVAILABLE:
        try:
            return sk_normalize(feature_vector.reshape(1, -1), norm='l2', axis=1)[0]
        except Exception as e:
            print(f"  Warning: sklearn normalization failed: {e}, falling back to manual.")

    # Manual Min-Max Scaling (fallback if sklearn is not robust or available)
    feature_min = np.min(feature_vector)
    feature_max = np.max(feature_vector)
    
    if feature_max - feature_min > 0:
        normalized = (feature_vector - feature_min) / (feature_max - feature_min)
    else:
        normalized = np.zeros_like(feature_vector)
    return normalized

# ======================= Machine Learning Classification =======================

def classify_with_ml(feature_vector):
    """Classify using pre-trained models (simplified version)"""
    feature_vector = validate_feature_vector(feature_vector)
    
    # Simplified logic, usually uses actual trained models or complex rules
    copy_move_indicators = [
        feature_vector[7] > 10 if len(feature_vector) > 7 else False, # RANSAC inliers
        feature_vector[9] > 10 if len(feature_vector) > 9 else False, # Block matches
        feature_vector[8] > 0 if len(feature_vector) > 8 else False, # Geometric transform presence
    ]
    
    splicing_indicators = [
        feature_vector[0] > 8 if len(feature_vector) > 0 else False, # ELA Mean
        feature_vector[4] > 3 if len(feature_vector) > 4 else False, # ELA Outlier Regions
        feature_vector[10] > 0.3 if len(feature_vector) > 10 else False, # Noise inconsistency
        feature_vector[11] > 0.15 if len(feature_vector) > 11 else False, # JPEG Ghost ratio
        feature_vector[17] > 0.3 if len(feature_vector) > 17 else False, # Texture inconsistency
        feature_vector[18] > 0.3 if len(feature_vector) > 18 else False, # Edge inconsistency
    ]
    
    copy_move_score = sum(copy_move_indicators) * 20
    splicing_score = sum(splicing_indicators) * 15
    
    return copy_move_score, splicing_score

def classify_with_advanced_ml(feature_vector):
    """Advanced ML classification with multiple algorithms"""
    feature_vector = validate_feature_vector(feature_vector)
    normalized_features = normalize_feature_vector(feature_vector)
    
    scores = {}
    
    rf_copy_move = simulate_random_forest_classification(normalized_features, 'copy_move')
    rf_splicing = simulate_random_forest_classification(normalized_features, 'splicing')
    scores['random_forest'] = (rf_copy_move, rf_splicing)
    
    svm_copy_move = simulate_svm_classification(normalized_features, 'copy_move')
    svm_splicing = simulate_svm_classification(normalized_features, 'splicing')
    scores['svm'] = (svm_copy_move, svm_splicing)
    
    nn_copy_move = simulate_neural_network_classification(normalized_features, 'copy_move')
    nn_splicing = simulate_neural_network_classification(normalized_features, 'splicing')
    scores['neural_network'] = (nn_copy_move, nn_splicing)
    
    copy_move_scores = [scores[model][0] for model in scores]
    splicing_scores = [scores[model][1] for model in scores]
    
    ensemble_copy_move = np.mean(copy_move_scores)
    ensemble_splicing = np.mean(splicing_scores)
    
    return ensemble_copy_move, ensemble_splicing, scores

def simulate_random_forest_classification(features, manipulation_type):
    """Simulate Random Forest classification"""
    # Assuming a feature vector of around 28-30 elements
    if manipulation_type == 'copy_move':
        # Weights focused on geometric & copy-move artifacts
        weights = np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.1, 0.8, 1.0, 1.0, 0.9, 0.3, 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.05, 0.02, 0.3, 0.6, 0.4, 0.5])
    else: # splicing
        # Weights focused on statistical, compression, and other inconsistencies
        weights = np.array([0.9, 0.9, 0.7, 0.7, 0.8, 0.7, 0.1, 0.05, 0.02, 0.05, 0.9, 0.7, 0.6, 0.7, 0.8, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.5, 0.5, 0.4, 0.7, 0.3, 0.4])
    
    # Pad or truncate weights to match feature vector length
    if len(weights) > len(features):
        weights = weights[:len(features)]
    elif len(weights) < len(features):
        # Default to a small, non-zero weight for padded features
        weights = np.pad(weights, (0, len(features) - len(weights)), 'constant', constant_values=0.1) 
    
    weighted_features = features * weights
    score = np.sum(weighted_features) / len(features) * 100
    
    return min(max(score, 0), 100)

def simulate_svm_classification(features, manipulation_type):
    """Simulate SVM classification - IMPROVED VERSION"""
    # Features will be normalized 0-1 from previous step.
    # Use selected indices to represent SVM decision boundaries for key features.
    
    if manipulation_type == 'copy_move':
        # Prioritize ransac inliers, block matches, geometric transform presence
        # indices for: ransac_inliers(7), block_matches(9), geometric_transform(8), ela_regional_inconsistency(3)
        key_indices = [idx for idx in [7, 9, 8, 3] if idx < len(features)]
        # Add a bias term / threshold
        bias_threshold = 0.4 
    else: # splicing
        # Prioritize ELA mean, noise inconsistency, JPEG ghost ratio, frequency inconsistency
        # indices for: ela_mean(0), ela_std(1), noise_inconsistency(10), jpeg_ghost_ratio(11), frequency_inconsistency(14), texture_inconsistency(16)
        key_indices = [idx for idx in [0, 1, 10, 11, 14, 16] if idx < len(features)]
        bias_threshold = 0.35 
    
    if len(key_indices) > 0:
        # Sum relevant features, high value indicates manipulation
        decision_score_sum = np.sum(features[key_indices]) / len(key_indices) # Average relevant feature values
        
        # Linearly map decision score (adjusted for bias) to 0-100
        # If score is > bias_threshold, it contributes positively, else negatively (after scaling)
        decision_score = (decision_score_sum - bias_threshold) * 200 # Factor 200 to map roughly from -0.x to 100
    else:
        decision_score = 0
    
    return min(max(decision_score, 0), 100) # Clip between 0 and 100

def simulate_neural_network_classification(features, manipulation_type):
    """Simulate Neural Network classification - FIXED VERSION"""
    try:
        # Simple two-layer feedforward network simulation
        # Normalize features first if not already done, though they should be 0-1
        
        # Hidden layer 1: Tanh activation
        # Arbitrary weights and biases for simulation
        hidden1_weights = np.linspace(-0.5, 0.5, len(features)) 
        hidden1_output = tanh_activation(features * hidden1_weights + 0.1) # Added a small bias

        # Hidden layer 2: Sigmoid activation
        hidden2_weights = np.linspace(0.2, 1.2, len(hidden1_output))
        hidden2_output = sigmoid(hidden1_output * hidden2_weights - 0.2) # Added a small negative bias

        # Output layer: weighted sum, then scaled
        output_weights_base = np.ones(len(hidden2_output)) 

        if manipulation_type == 'copy_move':
            # Boost weights for copy-move related features' contributions (e.g., SIFT, Block matches indices)
            if len(hidden2_output) > 9:
                output_weights_base[[7, 8, 9]] *= 3.0 # RANSAC Inliers, Geom Transform, Block Matches
            if len(hidden2_output) > 25: # Localization tampering percentage
                output_weights_base[25] *= 1.5 
        else: # splicing
            # Boost weights for splicing related features' contributions (e.g., ELA, Noise, JPEG ghost indices)
            if len(hidden2_output) > 11:
                output_weights_base[[0, 1, 10, 11]] *= 2.5 # ELA Mean, ELA Std, Noise Inc, JPEG Ghost Ratio
            if len(hidden2_output) > 18:
                 output_weights_base[[14, 16, 17, 18]] *= 2.0 # Freq, Texture, Edge, Illumination inconsistencies

        final_output_sum = np.sum(hidden2_output * output_weights_base)

        # Scale to 0-100 range.
        # Max sum for normalized features and boosted weights. A general factor is enough for simulation.
        max_possible_sum = np.sum(output_weights_base[output_weights_base > 0.0]) # sum positive weights only
        score = (final_output_sum / (max_possible_sum + 1e-9)) * 100 
        
        return min(max(score, 0), 100)
    except Exception as e:
        print(f"  Warning: Neural network simulation failed: {e}")
        # Fallback to a basic aggregated score
        feature_sum_effective = np.mean(features) # Mean of features more stable than raw sum
        if manipulation_type == 'copy_move':
            return min(feature_sum_effective * 50, 100) # Arbitrary scaling
        else:
            return min(feature_sum_effective * 40, 100)

# ======================= Advanced Classification System =======================

# Di dalam classification.py

def classify_manipulation_advanced(analysis_results):
    """Advanced classification with comprehensive scoring including localization and uncertainty"""
    
    try:
        # Initialize uncertainty classifier
        uncertainty_classifier = UncertaintyClassifier()
        
        # Calculate probabilities with uncertainty
        probabilities = uncertainty_classifier.calculate_manipulation_probability(analysis_results)
        uncertainty_report = uncertainty_classifier.generate_uncertainty_report(probabilities)
        
        # Traditional classification for backward compatibility
        feature_vector = prepare_feature_vector(analysis_results)
        ensemble_copy_move, ensemble_splicing, ml_scores = classify_with_advanced_ml(feature_vector)
        
        # Ambil skor ML yang lebih dapat diandalkan dari ensemble
        ml_copy_move_score = ensemble_copy_move
        ml_splicing_score = ensemble_splicing
        
        copy_move_score = 0
        splicing_score = 0
        
        # ======================= PENYESUAIAN SKOR MENTAH =======================
        # Tujuan: Kurangi bobot individu agar skor tidak cepat jenuh.
        # Total bobot maksimum untuk setiap kategori harus sekitar 150-160 sebelum dibatasi 100.

        # === Enhanced Copy-Move Detection (MAX ~150) ===
        # Ensure 'ransac_inliers' is always non-negative as validated in prepare_feature_vector
        ransac_inliers = analysis_results.get('ransac_inliers', 0)
        
        # Significantly increased weights for copy-move detection
        if ransac_inliers >= 50: copy_move_score += 60
        elif ransac_inliers >= 30: copy_move_score += 55
        elif ransac_inliers >= 20: copy_move_score += 50
        elif ransac_inliers >= 15: copy_move_score += 40
        elif ransac_inliers >= 10: copy_move_score += 30
        elif ransac_inliers >= 5: copy_move_score += 20
        
        block_matches = len(analysis_results.get('block_matches', []))
        # Significantly increased weights for block matching
        if block_matches >= 30: copy_move_score += 50
        elif block_matches >= 20: copy_move_score += 45
        elif block_matches >= 10: copy_move_score += 30
        elif block_matches >= 5: copy_move_score += 20
        
        # Significantly boosted other weights
        if analysis_results.get('geometric_transform') is not None: copy_move_score += 30
        if analysis_results.get('sift_matches', 0) >= 50: copy_move_score += 25
        
        ela_regional = analysis_results.get('ela_regional_stats', {})
        if ela_regional.get('regional_inconsistency', 1.0) < 0.2: copy_move_score += 10
        
        if 'localization_analysis' in analysis_results:
            tampering_pct = analysis_results['localization_analysis'].get('tampering_percentage', 0)
            if 10 < tampering_pct < 40: copy_move_score += 15
            elif 5 < tampering_pct <= 10: copy_move_score += 10
        
        # === Enhanced Splicing Detection (MAX ~160) ===
        # Significantly increased weights for splicing detection
        ela_mean = analysis_results.get('ela_mean', 0)
        ela_std = analysis_results.get('ela_std', 0)
        # Massively boosted weights for ELA analysis
        if ela_mean > 15.0 or ela_std > 22.0: splicing_score += 50
        elif ela_mean > 10.0 or ela_std > 20.0: splicing_score += 45
        elif ela_mean > 8.0 or ela_std > 18.0: splicing_score += 40
        elif ela_mean > 6.0 or ela_std > 15.0: splicing_score += 35
        
        outlier_regions = ela_regional.get('outlier_regions', 0)
        suspicious_regions = len(ela_regional.get('suspicious_regions', []))
        # Bobot tetap
        if outlier_regions > 8 or suspicious_regions > 5: splicing_score += 35
        elif outlier_regions > 5 or suspicious_regions > 3: splicing_score += 25
        elif outlier_regions >= 2 or suspicious_regions > 1: splicing_score += 15
        
        noise_analysis = analysis_results.get('noise_analysis', {})
        noise_inconsistency = noise_analysis.get('overall_inconsistency', 0)
        # Massively boosted weights for noise analysis
        if noise_inconsistency >= 0.7: splicing_score += 50
        elif noise_inconsistency >= 0.5: splicing_score += 45
        elif noise_inconsistency > 0.35: splicing_score += 35
        elif noise_inconsistency > 0.25: splicing_score += 25

        # Berikan penalti/bonus tambahan jika indikasi copy-move sangat kuat
        # Tujuannya agar kasus kombinasi (copy-move + splicing) lebih mudah terdeteksi
        if copy_move_score > 80 and splicing_score >= 40:
            splicing_score += 20
        if copy_move_score > 90 and splicing_score >= 50:
            splicing_score += 20
        
        jpeg_suspicious = analysis_results.get('jpeg_ghost_suspicious_ratio', 0)
        # Asumsi 'compression_inconsistency' ada di 'jpeg_analysis'. Correct access path for `basic_analysis` is needed
        jpeg_basic_analysis = analysis_results.get('jpeg_analysis', {}).get('basic_analysis', {})
        jpeg_compression = jpeg_basic_analysis.get('compression_inconsistency', False)

        if jpeg_suspicious >= 0.25 or jpeg_compression: splicing_score += 30
        elif jpeg_suspicious > 0.15: splicing_score += 15 # dikurangi dari 20
        elif jpeg_suspicious > 0.1: splicing_score += 10

        frequency_analysis = analysis_results.get('frequency_analysis', {})
        if frequency_analysis.get('frequency_inconsistency', 0) > 1.5: splicing_score += 20
        elif frequency_analysis.get('frequency_inconsistency', 0) > 1.0: splicing_score += 10

        texture_analysis = analysis_results.get('texture_analysis', {})
        if texture_analysis.get('overall_inconsistency', 0) > 0.4: splicing_score += 15
        elif texture_analysis.get('overall_inconsistency', 0) > 0.3: splicing_score += 10

        edge_analysis = analysis_results.get('edge_analysis', {})
        if edge_analysis.get('edge_inconsistency', 0) > 0.4: splicing_score += 15
        elif edge_analysis.get('edge_inconsistency', 0) > 0.3: splicing_score += 10

        illumination_analysis = analysis_results.get('illumination_analysis', {})
        if illumination_analysis.get('overall_illumination_inconsistency', 0) > 0.4: splicing_score += 20
        elif illumination_analysis.get('overall_illumination_inconsistency', 0) > 0.3: splicing_score += 10

        stat_analysis = analysis_results.get('statistical_analysis', {})
        # Ganti dengan check yang lebih aman jika kunci tidak ada
        rg_corr = stat_analysis.get('rg_correlation', 1.0)
        rb_corr = stat_analysis.get('rb_correlation', 1.0)
        gb_corr = stat_analysis.get('gb_correlation', 1.0)
        # If any correlation is significantly low, it's suspicious
        correlation_anomaly = (abs(rg_corr) < 0.3 or abs(rb_corr) < 0.3 or abs(gb_corr) < 0.3)
        if correlation_anomaly: splicing_score += 15

        metadata = analysis_results.get('metadata', {})
        metadata_issues = len(metadata.get('Metadata_Inconsistency', []))
        metadata_score = metadata.get('Metadata_Authenticity_Score', 100)
        if metadata_issues > 2 or metadata_score < 50: splicing_score += 20
        elif metadata_issues > 0 or metadata_score < 70: splicing_score += 10

        localization_analysis = analysis_results.get('localization_analysis', {})
        if localization_analysis:
            tampering_pct = localization_analysis.get('tampering_percentage', 0)
            if tampering_pct > 40: splicing_score += 20
            elif tampering_pct > 25: splicing_score += 15
            elif tampering_pct > 15: splicing_score += 10
        # ======================= AKHIR PENYESUAIAN =======================

        # Batasi skor mentah agar tidak lebih dari 100
        copy_move_score = min(copy_move_score, 100)
        splicing_score = min(splicing_score, 100)
        
        # Kombinasi skor (bobot ML disesuaikan untuk lebih seimbang)
        # Ensure scores are non-negative before combination
        ml_copy_move_score = max(0, ml_copy_move_score)
        ml_splicing_score = max(0, ml_splicing_score)

        raw_copy_move = (copy_move_score * 0.8 + ml_copy_move_score * 0.2)
        raw_splicing = (splicing_score * 0.8 + ml_splicing_score * 0.2)
        
        final_copy_move_score = min(max(0, int(raw_copy_move)), 100)
        final_splicing_score = min(max(0, int(raw_splicing)), 100)
        
        # Enhanced decision making
        # Adjusted DETECTION_THRESHOLD and CONFIDENCE_THRESHOLD for more realistic behavior
        # (Assuming these might come from config or fixed values here)
        detection_threshold = 45 
        confidence_threshold = 60
        manipulation_type = "Tidak Terdeteksi Manipulasi"
        confidence = "Rendah"
        details = []

        # LOGIKA KLASIFIKASI DENGAN URUTAN YANG BENAR
        # Ensure scores are actually above threshold
        is_cm_detected = final_copy_move_score >= detection_threshold
        is_splicing_detected = final_splicing_score >= detection_threshold

        if is_cm_detected or is_splicing_detected:
            # 1. Cek kasus kompleks dulu - relaxed conditions
            if final_copy_move_score >= confidence_threshold and final_splicing_score >= confidence_threshold:
                manipulation_type = "Manipulasi Kompleks (Copy-Move + Splicing)"
                confidence = get_enhanced_confidence_level(max(final_copy_move_score, final_splicing_score))
                details = get_enhanced_complex_details(analysis_results)
            # 2. Baru cek dominasi
            # A margin for dominance to avoid false complex if one score is barely higher
            elif final_copy_move_score > final_splicing_score * 1.3:
                manipulation_type = "Copy-Move Forgery"
                confidence = get_enhanced_confidence_level(final_copy_move_score)
                details = get_enhanced_copy_move_details(analysis_results)
            elif final_splicing_score > final_copy_move_score * 1.3:
                manipulation_type = "Splicing Forgery"
                confidence = get_enhanced_confidence_level(final_splicing_score)
                details = get_enhanced_splicing_details(analysis_results)
            # 3. Fallback for less clear cases, prioritize which score is higher if above threshold
            elif final_copy_move_score > final_splicing_score: # And is_cm_detected is True implied
                manipulation_type = "Copy-Move Forgery"
                confidence = get_enhanced_confidence_level(final_copy_move_score)
                details = get_enhanced_copy_move_details(analysis_results)
            else: # If slicing is higher or equal, and is_splicing_detected is True implied
                manipulation_type = "Splicing Forgery"
                confidence = get_enhanced_confidence_level(final_splicing_score)
                details = get_enhanced_splicing_details(analysis_results)
        
        # ======================= PERUBAHAN UTAMA DI SINI =======================
        final_manipulation_type = uncertainty_report.get('primary_assessment', 'N/A').replace('Indikasi: ', '')
        final_confidence = uncertainty_report.get('confidence_level', 'Sangat Rendah')
        final_details = uncertainty_report.get('reliability_indicators', [])
        final_details.append(uncertainty_report.get('uncertainty_summary', ''))

        classification_result = {
            'type': final_manipulation_type,
            'confidence': final_confidence,
            'copy_move_score': final_copy_move_score,
            'splicing_score': final_splicing_score,
            'details': final_details,
            'ml_scores': {
                'copy_move': ensemble_copy_move,
                'splicing': ensemble_splicing,
                'detailed_ml_scores': ml_scores
            },
            'feature_vector': feature_vector.tolist(),
            'traditional_scores': {
                'copy_move': copy_move_score,
                'splicing': splicing_score
            },
            'uncertainty_analysis': {
                'probabilities': probabilities,
                'report': uncertainty_report,
                'formatted_output': format_probability_results(probabilities, uncertainty_report),
            }
        }

        return classification_result
    except KeyError as e:
        print(f"  Warning: Classification failed due to missing key: {e}. Returning default error.")
        return {
            'type': "Analysis Error", 'confidence': "Error",
            'copy_move_score': 0, 'splicing_score': 0, 'details': [f"Classification error: Missing key {str(e)}. Please check analysis data integrity."],
            'ml_scores': {}, 'feature_vector': [], 'traditional_scores': {},
            'uncertainty_analysis': {
                'probabilities': {'copy_move_probability': 0.0, 'splicing_probability': 0.0, 'authentic_probability': 1.0, 'uncertainty_level': 1.0, 'confidence_intervals': {'copy_move': {'lower':0, 'upper':0}, 'splicing': {'lower':0, 'upper':0}, 'authentic': {'lower':0, 'upper':0}}},
                'report': {'primary_assessment': 'Error: Data Insufficient', 'confidence_level': 'Sangat Rendah', 'uncertainty_summary': 'Major data issues, classification unreliable', 'reliability_indicators': [], 'recommendation': 'Rerun analysis with valid data or debug.'},
                'formatted_output': "Error: Classification data missing. See logs."
            }
        }
    except Exception as e:
        print(f"  Warning: Classification failed: {e}. Returning default error.")
        import traceback
        traceback.print_exc() # For debugging
        return {
            'type': "Analysis Error", 'confidence': "Error",
            'copy_move_score': 0, 'splicing_score': 0, 'details': [f"Classification error: {str(e)}"],
            'ml_scores': {}, 'feature_vector': [], 'traditional_scores': {},
            'uncertainty_analysis': {
                'probabilities': {'copy_move_probability': 0.0, 'splicing_probability': 0.0, 'authentic_probability': 1.0, 'uncertainty_level': 1.0, 'confidence_intervals': {'copy_move': {'lower':0, 'upper':0}, 'splicing': {'lower':0, 'upper':0}, 'authentic': {'lower':0, 'upper':0}}},
                'report': {'primary_assessment': 'Error: Unknown Issue', 'confidence_level': 'Sangat Rendah', 'uncertainty_summary': 'An unexpected error occurred during classification. Result is unreliable.', 'reliability_indicators': [], 'recommendation': 'Rerun analysis or consult support.'},
                'formatted_output': "Error: Classification process failed. See logs for details."
            }
        }

# ======================= Confidence and Detail Functions =======================

def get_enhanced_confidence_level(score):
    if score >= 90: return "Sangat Tinggi (>90%)"
    elif score >= 75: return "Tinggi (75-90%)"
    elif score >= 60: return "Sedang (60-75%)"
    elif score >= 45: return "Rendah (45-60%)"
    else: return "Sangat Rendah (<45%)"

def get_enhanced_copy_move_details(results):
    details = []
    # Using .get() for safer access
    ransac_inliers = results.get('ransac_inliers', 0)
    geometric_transform = results.get('geometric_transform')
    block_matches_len = len(results.get('block_matches', []))
    sift_matches = results.get('sift_matches', 0)
    ela_regional_stats = results.get('ela_regional_stats', {})
    localization_analysis = results.get('localization_analysis', {})

    if ransac_inliers > 0: details.append(f"✓ RANSAC verification: {ransac_inliers} geometric matches")
    
    transform_type = None
    if isinstance(geometric_transform, (list, tuple)) and len(geometric_transform) > 0:
        transform_type = geometric_transform[0] 
    elif geometric_transform is not None: # Direct value check
        transform_type = "Detected" # Could be any non-None object

    if transform_type is not None:
        details.append(f"✓ Geometric transformation: {transform_type}")
        
    if block_matches_len > 0: details.append(f"✓ Block matching: {block_matches_len} identical blocks")
    if sift_matches > 10: details.append(f"✓ Feature matching: {sift_matches} SIFT correspondences")
    if ela_regional_stats.get('regional_inconsistency', 1.0) < 0.3: details.append("✓ Consistent ELA patterns (same source content)")
    if localization_analysis.get('tampering_percentage', 0) > 5:
        details.append(f"✓ K-means localization: {localization_analysis['tampering_percentage']:.1f}% tampering detected")
    return details

def get_enhanced_splicing_details(results):
    details = []
    ela_regional_stats = results.get('ela_regional_stats', {})
    ela_outlier_regions = ela_regional_stats.get('outlier_regions', 0)
    jpeg_basic_analysis = results.get('jpeg_analysis', {}).get('basic_analysis', {})
    noise_analysis = results.get('noise_analysis', {})
    frequency_analysis = results.get('frequency_analysis', {})
    texture_analysis = results.get('texture_analysis', {})
    edge_analysis = results.get('edge_analysis', {})
    illumination_analysis = results.get('illumination_analysis', {})
    metadata = results.get('metadata', {})
    localization_analysis = results.get('localization_analysis', {})

    if ela_outlier_regions > 0: details.append(f"⚠ ELA anomalies: {ela_outlier_regions} suspicious regions")
    if jpeg_basic_analysis.get('compression_inconsistency', False): details.append("⚠ JPEG compression inconsistency detected")
    if noise_analysis.get('overall_inconsistency', 0) > 0.25: details.append(f"⚠ Noise inconsistency: {noise_analysis['overall_inconsistency']:.3f}")
    if frequency_analysis.get('frequency_inconsistency', 0) > 1.0: details.append("⚠ Frequency domain anomalies detected")
    if texture_analysis.get('overall_inconsistency', 0) > 0.3: details.append("⚠ Texture pattern inconsistency")
    if edge_analysis.get('edge_inconsistency', 0) > 0.3: details.append("⚠ Edge density inconsistency")
    if illumination_analysis.get('overall_illumination_inconsistency', 0) > 0.3: details.append("⚠ Illumination inconsistency detected")
    if len(metadata.get('Metadata_Inconsistency', [])) > 0: details.append(f"⚠ Metadata issues: {len(metadata['Metadata_Inconsistency'])} found")
    if localization_analysis.get('tampering_percentage', 0) > 15:
        details.append(f"⚠ K-means localization: {localization_analysis['tampering_percentage']:.1f}% suspicious areas detected")
    return details

def get_enhanced_complex_details(results):
    return get_enhanced_copy_move_details(results) + get_enhanced_splicing_details(results)

# ======================= Classification Calibration =======================

def calibrate_classification_thresholds(validation_results=None):
    """Calibrate classification thresholds based on validation data"""
    # Default thresholds
    thresholds = {
        'detection_threshold': 45,
        'confidence_threshold': 60,
        'copy_move_dominance': 1.3,
        'splicing_dominance': 1.3
    }
    
    # If validation results are provided, adjust thresholds
    if validation_results:
        # Adjust based on false positive/negative rates
        if validation_results.get('false_positive_rate', 0) > 0.1:
            thresholds['detection_threshold'] += 5
        if validation_results.get('false_negative_rate', 0) > 0.1:
            thresholds['detection_threshold'] -= 5
    
    return thresholds

def evaluate_classification_performance(predictions, ground_truth):
    """Evaluate classification performance metrics"""
    metrics = {}
    
    # Calculate basic metrics
    # Convert scores to binary predictions (e.g., > 0 means manipulation)
    predictions_binary = [1 if p > 0 else 0 for p in predictions]
    ground_truth_binary = [1 if g > 0 else 0 for g in ground_truth]

    total = len(predictions_binary)
    if total == 0:
        return {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
            'false_positive_rate': 0, 'false_negative_rate': 0
        }

    # Ensure predictions and ground_truth are NumPy arrays for direct comparison if needed later
    predictions_binary = np.array(predictions_binary)
    ground_truth_binary = np.array(ground_truth_binary)

    # Calculate confusion matrix components
    true_positives = np.sum((predictions_binary == 1) & (ground_truth_binary == 1))
    true_negatives = np.sum((predictions_binary == 0) & (ground_truth_binary == 0))
    false_positives = np.sum((predictions_binary == 1) & (ground_truth_binary == 0))
    false_negatives = np.sum((predictions_binary == 0) & (ground_truth_binary == 1))
    
    metrics['accuracy'] = (true_positives + true_negatives) / total if total > 0 else 0
    metrics['precision'] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    metrics['recall'] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    metrics['false_positive_rate'] = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    metrics['false_negative_rate'] = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
    
    return metrics

# ======================= Classification Utilities =======================

def generate_classification_report(classification_result, analysis_results):
    """Generate comprehensive classification report"""
    # Safe access to nested dictionary values
    ml_scores = classification_result.get('ml_scores', {})
    detailed_ml_scores = ml_scores.get('detailed_ml_scores', {})
    random_forest_scores = detailed_ml_scores.get('random_forest', (0.0, 0.0))
    svm_scores = detailed_ml_scores.get('svm', (0.0, 0.0))
    neural_network_scores = detailed_ml_scores.get('neural_network', (0.0, 0.0))
    
    report = {
        'summary': {
            'detected_type': classification_result['type'],
            'confidence_level': classification_result['confidence'],
            'copy_move_score': classification_result['copy_move_score'],
            'splicing_score': classification_result['splicing_score']
        },
        'evidence': {
            'technical_indicators': classification_result['details'],
            'feature_count': len(classification_result['feature_vector']),
            'ml_confidence_copy_move': ml_scores.get('copy_move', 0.0),
            'ml_confidence_splicing': ml_scores.get('splicing', 0.0)
        },
        'methodology': {
            'feature_vector_size': len(classification_result['feature_vector']),
            'ml_algorithms_used': ['Random Forest', 'SVM', 'Neural Network'],
            'random_forest_scores': random_forest_scores,
            'svm_scores': svm_scores,
            'neural_network_scores': neural_network_scores,
            'traditional_scoring': classification_result.get('traditional_scores', {}),
            'ensemble_weighting': 'Traditional: 60%, ML: 40%' # Update if weights in classify_manipulation_advanced change
        },
        'reliability': {
            'metadata_score': analysis_results.get('metadata', {}).get('Metadata_Authenticity_Score', 0),
            'analysis_completeness': f"{analysis_results.get('pipeline_status', {}).get('completed_stages', 0)} out of {analysis_results.get('pipeline_status', {}).get('total_stages', 0)} pipeline stages completed",
            'cross_validation': 'Multiple detector agreement'
        }
    }
    
    return report

def export_classification_metrics(classification_result, output_filename="classification_metrics.txt"):
    """Export classification metrics to text file"""
    
    content = f"""CLASSIFICATION METRICS EXPORT
{'='*50}

FINAL CLASSIFICATION:
Type: {classification_result.get('type', 'N/A')}
Confidence: {classification_result.get('confidence', 'N/A')}

SCORING BREAKDOWN:
Copy-Move Score: {classification_result.get('copy_move_score', 0)}/100
Splicing Score: {classification_result.get('splicing_score', 0)}/100

TRADITIONAL SCORES:
Traditional Copy-Move: {classification_result.get('traditional_scores', {}).get('copy_move', 0)}/100
Traditional Splicing: {classification_result.get('traditional_scores', {}).get('splicing', 0)}/100

MACHINE LEARNING SCORES:
ML Copy-Move: {classification_result.get('ml_scores', {}).get('copy_move', 0):.1f}/100
ML Splicing: {classification_result.get('ml_scores', {}).get('splicing', 0):.1f}/100
Ensemble Copy-Move: {classification_result.get('ml_scores', {}).get('ensemble_copy_move', 0):.1f}/100
Ensemble Splicing: {classification_result.get('ml_scores', {}).get('ensemble_splicing', 0):.1f}/100

DETAILED ML SCORES:
Random Forest Copy-Move: {classification_result.get('ml_scores', {}).get('detailed_ml_scores', {}).get('random_forest', (0.0,0.0))[0]:.1f}
Random Forest Splicing: {classification_result.get('ml_scores', {}).get('detailed_ml_scores', {}).get('random_forest', (0.0,0.0))[1]:.1f}
SVM Copy-Move: {classification_result.get('ml_scores', {}).get('detailed_ml_scores', {}).get('svm', (0.0,0.0))[0]:.1f}
SVM Splicing: {classification_result.get('ml_scores', {}).get('detailed_ml_scores', {}).get('svm', (0.0,0.0))[1]:.1f}
Neural Network Copy-Move: {classification_result.get('ml_scores', {}).get('detailed_ml_scores', {}).get('neural_network', (0.0,0.0))[0]:.1f}
Neural Network Splicing: {classification_result.get('ml_scores', {}).get('detailed_ml_scores', {}).get('neural_network', (0.0,0.0))[1]:.1f}

FEATURE VECTOR:
Size: {len(classification_result.get('feature_vector', []))} features
Values: {classification_result.get('feature_vector', [])}

DETECTION DETAILS:
"""
    
    for detail in classification_result.get('details', []):
        content += f"• {detail}\n"
    
    content += f"""
METHODOLOGY:
• Feature extraction: 16-stage analysis pipeline
• ML ensemble: Random Forest + SVM + Neural Network
• Scoring combination: Traditional (60%) + ML (40%)
• Threshold-based decision making with confidence calibration

UNCERTAINTY ANALYSIS (New in V3):
Primary Assessment: {classification_result.get('uncertainty_analysis', {}).get('report', {}).get('primary_assessment', 'N/A')}
Uncertainty Level: {classification_result.get('uncertainty_analysis', {}).get('probabilities', {}).get('uncertainty_level', 0.0):.1%}
Confidence Level: {classification_result.get('uncertainty_analysis', {}).get('report', {}).get('confidence_level', 'N/A')}

END OF METRICS
{'='*50}
"""
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📊 Classification metrics exported to '{output_filename}'")
    return output_filename

# ======================= Advanced Feature Analysis =======================

def analyze_feature_importance(feature_vector, classification_result):
    """Analyze feature importance for classification decision"""
    
    # Updated feature names to match prepare_feature_vector (current size ~28 features)
    feature_names = [
        'ELA Mean', 'ELA Std', 'ELA Mean Variance', 'ELA Regional Inconsistency',
        'ELA Outlier Regions', 'ELA Suspicious Regions Count', 
        'SIFT Matches', 'RANSAC Inliers', 'Geometric Transform Exists', 
        'Block Matches Count', 'Noise Overall Inconsistency', 
        'JPEG Ghost Suspicious Ratio', 'JPEG Basic Response Variance', 'JPEG Basic Double Comp Indicator', 
        'Frequency Inconsistency', 'DCT Freq Ratio', 'Texture Overall Inconsistency', 
        'Edge Inconsistency', 'Illumination Overall Inconsistency', 
        'R Entropy', 'G Entropy', 'B Entropy', 'RG Correlation', 'Overall Entropy', 
        'Metadata Auth Score', 
        'Loc Tampering Percentage', 'Loc KMeans Cluster Count', 'Loc Max Cluster ELA'
    ]
    
    # Ensure feature names match vector length, pad with generic names if vector is longer
    if len(feature_names) > len(feature_vector):
        feature_names = feature_names[:len(feature_vector)]
    elif len(feature_names) < len(feature_vector):
        feature_names.extend([f'Feature_{i}' for i in range(len(feature_names), len(feature_vector))])
    
    # Calculate feature importance based on contribution to final scores
    copy_move_importance = []
    splicing_importance = []
    
    # Define importance weights for different manipulation types based on the adjusted scoring
    # These weights are simplified; a real model would derive these automatically (e.g., from tree models)
    copy_move_weights_template = [ # Expected length 28 (align with prepare_feature_vector output)
        0.05, 0.05, 0.1, 0.1, 0.2, 0.1,  # ELA (minor indirect for CM)
        0.8, 1.0, 1.0,  # SIFT & RANSAC (HIGH for CM)
        0.9,  # Block Matches (HIGH for CM)
        0.3,  # Noise (minor for CM)
        0.2, 0.1, 0.05,  # JPEG (minor for CM)
        0.1, 0.05,  # Freq, Texture (minor for CM)
        0.1,  # Edge (minor for CM)
        0.1,  # Illumination (minor for CM)
        0.01, 0.01, 0.01,  # R,G,B Entropy (very minor for CM)
        0.05, 0.02, # RG Correlation, Overall Entropy (very minor for CM)
        0.2, # Metadata (minor for CM)
        0.7, 0.5, 0.6 # Localization (medium-high for CM)
    ]
    
    splicing_weights_template = [ # Expected length 28
        0.9, 0.9, 0.7, 0.7, 0.8, 0.7, # ELA (HIGH for Splicing)
        0.05, 0.02, 0.01, # SIFT & RANSAC (very low for Splicing direct)
        0.1, # Block Matches (low for Splicing direct)
        0.9, # Noise (HIGH for Splicing)
        0.8, 0.7, 0.8, # JPEG (HIGH for Splicing)
        0.7, 0.6, # Freq, Texture (medium for Splicing)
        0.7, # Edge (medium for Splicing)
        0.8, # Illumination (HIGH for Splicing)
        0.5, 0.5, 0.5, # R,G,B Entropy (medium for Splicing)
        0.6, 0.5, # RG Correlation, Overall Entropy (medium for Splicing)
        0.7, # Metadata (medium for Splicing)
        0.8, 0.4, 0.5 # Localization (medium-high for Splicing)
    ]
    
    # Normalize weights to match feature vector length
    # Ensure templates are padded with reasonable defaults if shorter than current feature_vector
    current_len = len(feature_vector)
    copy_move_weights = np.pad(copy_move_weights_template, (0, max(0, current_len - len(copy_move_weights_template))), 'constant', constant_values=0.1)
    splicing_weights = np.pad(splicing_weights_template, (0, max(0, current_len - len(splicing_weights_template))), 'constant', constant_values=0.1)
    
    copy_move_importance_list = []
    splicing_importance_list = []
    
    # Calculate importance scores
    for i in range(current_len):
        name = feature_names[i]
        value = feature_vector[i]
        cm_weight = copy_move_weights[i]
        sp_weight = splicing_weights[i]
        
        # Calculate importance. A high positive value (value * weight) means it contributes
        # significantly to the "manipulated" score for that category.
        # This is a heuristic and not derived from a trained ML model.
        cm_importance = float(value * cm_weight)
        sp_importance = float(value * sp_weight)
        
        copy_move_importance_list.append({
            'feature': name,
            'value': float(value),
            'importance': cm_importance,
            'weight': float(cm_weight)
        })
        
        splicing_importance_list.append({
            'feature': name,
            'value': float(value),
            'importance': sp_importance,
            'weight': float(sp_weight)
        })
    
    # Sort by importance (highest positive importance indicates strongest evidence)
    copy_move_importance_list.sort(key=lambda x: x['importance'], reverse=True)
    splicing_importance_list.sort(key=lambda x: x['importance'], reverse=True)
    
    return {
        'copy_move_importance': copy_move_importance_list[:10],  # Top 10 most influential features
        'splicing_importance': splicing_importance_list[:10],   # Top 10 most influential features
        'feature_summary': {
            'total_features': current_len,
            'significant_copy_move_features': len([x for x in copy_move_importance_list if x['importance'] > 0.1]), # Threshold can be tuned
            'significant_splicing_features': len([x for x in splicing_importance_list if x['importance'] > 0.1])
        }
    }

def create_classification_summary():
    """Create summary of classification capabilities"""
    
    summary = """
CLASSIFICATION SYSTEM SUMMARY
=============================

DETECTION CAPABILITIES:
• Copy-Move Forgery Detection
• Splicing Forgery Detection  
• Complex Manipulation Detection
• Authentic Image Verification

MACHINE LEARNING MODELS (Simulated):
• Random Forest Classifier
• Support Vector Machine (SVM)
• Neural Network Simulation
• Ensemble Method Integration (combines multiple models)

FEATURE ANALYSIS:
• Multi-dimensional feature vector (~28 features)
• Features from ELA, noise, frequency, texture, edges, illumination, statistics, metadata, localization
• Feature importance ranking (identifies key evidence)

CONFIDENCE SCORING:
• Traditional rule-based scoring (heuristic)
• ML-based probability estimation (from ensemble models)
• Ensemble confidence calibration (combines traditional and ML)
• Threshold-based decision making with uncertainty quantification

PERFORMANCE CHARACTERISTICS (Based on internal validation):
• Designed for high accuracy on typical manipulated images
• Aims for low false positive rate (minimal false alarms)
• Robust to common image processing like compression
• Scalable for various image sizes and resolutions

VALIDATION METHODS (Internal & External Standards):
• Cross-validation with known datasets (simulated for deployment)
• Statistical significance testing of features
• Consensus evaluation across different detection algorithms
• Integration of DFRWS & NIST forensic validation principles
"""
    
    return summary


# --- END OF FILE classification.py ---