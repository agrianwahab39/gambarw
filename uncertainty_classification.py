"""
Advanced Uncertainty-Based Classification System
Sistem klasifikasi dengan probabilitas dan ketidakpastian untuk forensik gambar
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class UncertaintyClassifier:
    """
    Klasifikasi dengan model ketidakpastian yang mempertimbangkan:
    1. Confidence intervals
    2. Probabilitas manipulasi
    3. Indikator keraguan
    4. Multiple evidence weighting
    """
    
    def __init__(self):
        # Base uncertainty represents inherent ambiguity in forensic image analysis, capped
        self.base_uncertainty = 0.08  # Lower base to reflect potential for clear cases
        self.confidence_thresholds = {
            'very_high': 0.90,
            'high': 0.75,
            'medium': 0.60,
            'low': 0.45,
            'very_low': 0.30
        }
        
    def calculate_manipulation_probability(self, analysis_results: Dict) -> Dict:
        """
        Hitung probabilitas manipulasi dengan mempertimbangkan ketidakpastian
        """
        # Extract scores dari hasil analisis
        copy_move_indicators = self._extract_copy_move_indicators(analysis_results)
        splicing_indicators = self._extract_splicing_indicators(analysis_results)
        authenticity_indicators = self._extract_authenticity_indicators(analysis_results)
        
        # Hitung probabilitas dasar (skor gabungan, belum dinormalisasi atau diatur untuk konflik)
        copy_move_raw_prob = self._calculate_weighted_probability(copy_move_indicators)
        splicing_raw_prob = self._calculate_weighted_probability(splicing_indicators)
        authentic_raw_prob = self._calculate_weighted_probability(authenticity_indicators)
        
        # Apply mutual exclusivity logic if conflicting high scores
        # If strong evidence for both CM and Splicing, treat as "Complex" but reduce
        # the certainty that one is *only* CM or Splicing.
        if copy_move_raw_prob > 0.6 and splicing_raw_prob > 0.6:
            # Adjust to represent complex manipulation
            # Reduce individual pure probabilities but keep high total manipulation
            temp_cm_raw = copy_move_raw_prob * 0.7
            temp_sp_raw = splicing_raw_prob * 0.7
            temp_au_raw = authentic_raw_prob * 0.3 # Less likely to be authentic if both high
            copy_move_raw_prob, splicing_raw_prob, authentic_raw_prob = temp_cm_raw, temp_sp_raw, temp_au_raw
            print("  [Uncertainty] Detected high conflicting signals, adjusting raw probabilities for 'Complex'.")

        # Now, normalize the probabilities so they sum to 1.
        # This acts as the softmax layer in a neural network.
        exp_cm = np.exp(copy_move_raw_prob * 3) # Scale up for better differentiation after sigmoid
        exp_sp = np.exp(splicing_raw_prob * 3)
        exp_au = np.exp(authentic_raw_prob * 3)

        # Normalize to get final probabilities (using exponential for soft decision boundary)
        sum_exp = exp_cm + exp_sp + exp_au
        
        if sum_exp > 0:
            copy_move_prob_final = exp_cm / sum_exp
            splicing_prob_final = exp_sp / sum_exp
            authentic_prob_final = exp_au / sum_exp
        else: # Fallback for no data
            copy_move_prob_final = splicing_prob_final = authentic_prob_final = 1/3
            
        # Recalculate uncertainty factors based on ALL evidence, not just the "raw" scores
        # This will include how sparse or ambiguous the *actual input evidence* was.
        uncertainty_factors = self._calculate_uncertainty_factors(
            analysis_results, copy_move_indicators, splicing_indicators, authenticity_indicators
        )
        
        # Apply uncertainty *after* normalization to show confidence around the final probability
        # The probability is what the model believes; uncertainty is how sure it is of that belief.
        
        return {
            'copy_move_probability': copy_move_prob_final,
            'splicing_probability': splicing_prob_final,
            'authentic_probability': authentic_prob_final,
            'uncertainty_level': self._calculate_overall_uncertainty(uncertainty_factors),
            'confidence_intervals': self._calculate_confidence_intervals(
                copy_move_prob_final, splicing_prob_final, authentic_prob_final, uncertainty_factors
            )
        }
    
    def _extract_copy_move_indicators(self, results: Dict) -> List[Tuple[float, float]]:
        """Extract copy-move indicators dengan weights"""
        indicators = [] # Format: (score, weight)
        
        # RANSAC inliers (high weight) - higher inliers means more confident CM
        ransac_inliers = results.get('ransac_inliers', 0)
        # Using a non-linear scaling (log/sqrt) or threshold-based score for features for better discrimination
        if ransac_inliers >= 5: 
            score = min(np.sqrt(ransac_inliers) / np.sqrt(50), 1.0) # Sqrt scale for large range of inliers
            indicators.append((score, 0.30))  # High weight, as RANSAC is strong evidence
        
        # Block matches
        block_matches_count = len(results.get('block_matches', []))
        if block_matches_count >= 3:
            score = min(block_matches_count / 30.0, 1.0) # Direct ratio up to 30 matches
            indicators.append((score, 0.25))
        
        # Geometric transform existence (strong, but 0/1 indicator)
        if results.get('geometric_transform') is not None:
            indicators.append((1.0, 0.20)) # High weight for transform
        
        # SIFT raw matches (supportive)
        sift_matches_raw = results.get('sift_matches', 0)
        if sift_matches_raw > 20:
            score = min(sift_matches_raw / 200.0, 1.0)
            indicators.append((score, 0.10))
        
        # ELA regional inconsistency (indirectly, if low means consistency implying same source for CM)
        ela_regional_inconsistency = results.get('ela_regional_stats', {}).get('regional_inconsistency', 1.0)
        if ela_regional_inconsistency < 0.25: # Low inconsistency implies possible copy-move
            indicators.append((1.0 - ela_regional_inconsistency / 0.5, 0.10)) # Inverted score for consistency

        # Tampering localization percentage (medium-high percentage can indicate CM areas)
        tampering_pct = results.get('localization_analysis', {}).get('tampering_percentage', 0.0)
        if 5.0 < tampering_pct < 60.0: # Range likely for tampering (not too small, not too large)
             score = min(tampering_pct / 50.0, 1.0)
             indicators.append((score, 0.05))

        return indicators
    
    def _extract_splicing_indicators(self, results: Dict) -> List[Tuple[float, float]]:
        """Extract splicing indicators dengan weights"""
        indicators = []
        
        # ELA analysis (main indicator for splicing due to compression difference)
        ela_mean = results.get('ela_mean', 0)
        ela_std = results.get('ela_std', 0)
        # Higher ELA mean/std indicates compression inconsistencies likely from splicing
        if ela_mean > 10 or ela_std > 18:
            score = min(max(ela_mean / 25.0, ela_std / 35.0), 1.0)
            indicators.append((score, 0.25))
        
        # Noise inconsistency (non-uniform noise patterns often due to splicing)
        noise_inconsistency = results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        if noise_inconsistency > 0.2:
            score = min(noise_inconsistency / 0.6, 1.0)
            indicators.append((score, 0.20))
        
        # JPEG Ghost analysis (direct evidence of double compression or pasting)
        jpeg_ghost_suspicious_ratio = results.get('jpeg_ghost_suspicious_ratio', 0)
        if jpeg_ghost_suspicious_ratio > 0.05: # Even small ratio can be indicative
            score = min(jpeg_ghost_suspicious_ratio / 0.3, 1.0)
            indicators.append((score, 0.20))
        
        # Frequency inconsistency (spectral artifacts from pasting)
        freq_inconsistency = results.get('frequency_analysis', {}).get('frequency_inconsistency', 0)
        if freq_inconsistency > 0.8: # Threshold could be lower, more common artifact
            score = min(freq_inconsistency / 2.0, 1.0)
            indicators.append((score, 0.10))
        
        # Illumination inconsistency (very strong sign of splicing)
        illum_inconsistency = results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0)
        if illum_inconsistency > 0.2: # Significant illumination change
            score = min(illum_inconsistency / 0.5, 1.0)
            indicators.append((score, 0.15)) # High weight
        
        # Edge inconsistency (blurriness or sharp edges mismatches)
        edge_inconsistency = results.get('edge_analysis', {}).get('edge_inconsistency', 0)
        if edge_inconsistency > 0.2:
            score = min(edge_inconsistency / 0.5, 1.0)
            indicators.append((score, 0.05))

        # Metadata issues (soft indicator, often tampered in conjunction with image manipulation)
        metadata_inconsistencies = len(results.get('metadata', {}).get('Metadata_Inconsistency', []))
        if metadata_inconsistencies > 0:
            score = min(metadata_inconsistencies / 5.0, 1.0) # Up to 5 issues is max score
            indicators.append((score, 0.05))

        # Statistical anomalies (entropy/correlation changes)
        stat_analysis = results.get('statistical_analysis', {})
        # If any correlation is far from 1 (or -1), might be suspicious
        rg_corr = stat_analysis.get('rg_correlation', 1.0)
        rb_corr = stat_analysis.get('rb_correlation', 1.0)
        gb_corr = stat_analysis.get('gb_correlation', 1.0)
        # Anomaly if absolute correlation is low (0.0 to 0.5 typically for manipulated areas)
        if abs(rg_corr) < 0.7 or abs(rb_corr) < 0.7 or abs(gb_corr) < 0.7:
            # Score higher for lower correlations
            score = max(max(0, 1.0 - abs(rg_corr)), max(0, 1.0 - abs(rb_corr)), max(0, 1.0 - abs(gb_corr)))
            indicators.append((score, 0.05))

        return indicators
    
    def _extract_authenticity_indicators(self, results: Dict) -> List[Tuple[float, float]]:
        """
        Extract authenticity indicators. These are inverse to manipulation indicators.
        Absence of manipulation evidence or positive integrity checks.
        """
        indicators = [] # (score, weight)
        
        # PENINGKATAN #1: Beri bobot sangat tinggi jika TIDAK ADA bukti manipulasi sama sekali
        ransac_inliers = results.get('ransac_inliers', 0)
        block_matches_count = len(results.get('block_matches', []))
        jpeg_ghost_suspicious_ratio = results.get('jpeg_ghost_suspicious_ratio', 0)
        noise_inconsistency = results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        
        # Jika tidak ada satupun bukti copy-move, ini adalah sinyal keaslian yang kuat
        if ransac_inliers == 0 and block_matches_count == 0:
            indicators.append((1.0, 0.40)) # Bobot ditingkatkan dari 0.25 menjadi 0.40
        
        # PENINGKATAN #2: Tambahkan kondisi untuk noise dan ELA yang sangat rendah
        ela_mean = results.get('ela_mean', 0)
        if noise_inconsistency < 0.15 and ela_mean < 5:
             indicators.append((1.0 - noise_inconsistency, 0.25)) # Sinyal kuat lainnya
        
        # High Metadata score (direct indicator of authenticity)
        metadata_auth_score = results.get('metadata', {}).get('Metadata_Authenticity_Score', 0)
        if metadata_auth_score > 80: # Tingkatkan ambang batas
            indicators.append((metadata_auth_score / 100.0, 0.20)) # Bobot sedikit diturunkan karena bisa dimanipulasi
        
        # Low ELA mean/std (no compression artifacts that indicate tampering)
        ela_std = results.get('ela_std', 0)
        if ela_mean < 8 and ela_std < 15: # Healthy low ELA
            indicators.append((1.0 - (ela_mean / 10.0), 0.15)) # Bobot disesuaikan
        
        # Low JPEG Ghost ratio
        if jpeg_ghost_suspicious_ratio < 0.02: # Very low ghost means good
            indicators.append((1.0 - jpeg_ghost_suspicious_ratio / 0.1, 0.10))

        # Low Tampering percentage (localization shows clean image)
        tampering_pct = results.get('localization_analysis', {}).get('tampering_percentage', 0.0)
        if tampering_pct < 2.0: # Almost no detected tampering regions
            indicators.append((1.0 - tampering_pct / 5.0, 0.10)) # Bobot ditingkatkan

        return indicators
    
    def _calculate_weighted_probability(self, indicators: List[Tuple[float, float]]) -> float:
        """Hitung probabilitas dengan weighted average"""
        if not indicators:
            return 0.1 # Small baseline for empty indicators, allows other categories to dominate
        
        total_weight = sum(weight for _, weight in indicators)
        if total_weight == 0:
            return 0.1 # Avoid division by zero
        
        weighted_sum = sum(score * weight for score, weight in indicators)
        # Scale sum based on maximum possible weighted sum for the given weights
        # max_possible_sum_if_all_scores_1 = total_weight
        # To get a value that ranges from ~0 to ~1 (or max possible for sum of weights)
        # Using sigmoid to squash the sum if raw weighted sum gets very high
        prob_sum = weighted_sum / total_weight # Max score could be 1.0 (if all scores are 1)
        return float(prob_sum)
    
    def _calculate_uncertainty_factors(self, results: Dict, cm_indicators: List, sp_indicators: List, au_indicators: List) -> Dict[str, float]:
        """
        Hitung faktor ketidakpastian untuk setiap kategori.
        Ketidakpastian meningkat jika:
        1. Bukti ambigu (misalnya, ELA di zona abu-abu)
        2. Bukti jarang (sedikit indikator ditemukan)
        3. Konteks gambar membuat deteksi sulit (e.g., noise tinggi alami, terlalu mulus, kecil)
        """
        factors = {}
        
        # General image context for overall uncertainty contribution
        overall_uncertainty_context = 0.0
        
        # If image is very small, more uncertainty
        img_width = results.get('metadata', {}).get('Dimensions', (0,0))[0] # Get width, or fallback
        if isinstance(img_width, str) and 'x' in img_width: # Sometimes comes as "WxH" string
             img_width = int(img_width.split('x')[0])
        else: # Try as int/float
             img_width = int(img_width)
        
        if img_width < 300: # Small image means more uncertainty for all
            overall_uncertainty_context += 0.08
        
        # If too simple (e.g., solid color) or too complex (e.g., highly textured natural images), affects some
        ela_mean = results.get('ela_mean', 0)
        ela_std = results.get('ela_std', 0)
        if ela_mean < 3 and ela_std < 5: # Very uniform/smooth image
            overall_uncertainty_context += 0.05
        elif ela_mean > 20 or ela_std > 30: # Extremely noisy/highly compressed
             overall_uncertainty_context += 0.03 # Moderate noise can confuse, but extreme noise makes everything random, hard to judge.

        # --- Copy-move uncertainty ---
        cm_uncertainty = self.base_uncertainty + overall_uncertainty_context
        # If RANSAC inliers or block matches are found but are few and not very confident:
        if 0 < results.get('ransac_inliers', 0) < 10: cm_uncertainty += 0.05
        if 0 < len(results.get('block_matches', [])) < 5: cm_uncertainty += 0.05
        
        # If number of CM indicators is low, it makes it less certain.
        if len(cm_indicators) < 2: cm_uncertainty += 0.10 # Not enough evidence
        factors['copy_move'] = min(cm_uncertainty, 0.4) # Cap at 40%

        # --- Splicing uncertainty ---
        sp_uncertainty = self.base_uncertainty + overall_uncertainty_context
        # If ELA mean/std are in the ambiguous "grey zone"
        if 5 < ela_mean < 12 or 10 < ela_std < 20: sp_uncertainty += 0.08 # Ambiguous ELA zone
        noise_inconsistency = results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        if 0.15 < noise_inconsistency < 0.3: sp_uncertainty += 0.06 # Ambiguous noise
        
        if len(sp_indicators) < 2: sp_uncertainty += 0.10 # Not enough evidence
        factors['splicing'] = min(sp_uncertainty, 0.45) # Splicing can be more complex to be certain

        # --- Authentic uncertainty ---
        au_uncertainty = self.base_uncertainty + overall_uncertainty_context
        metadata_score = results.get('metadata', {}).get('Metadata_Authenticity_Score', 0)
        if 40 < metadata_score < 70: au_uncertainty += 0.08 # Ambiguous metadata
        
        # If the authentic indicators are very few (could just be lucky)
        if len(au_indicators) < 2: au_uncertainty += 0.10
        factors['authentic'] = min(au_uncertainty, 0.35)

        return factors
    
    def _apply_uncertainty(self, base_prob: float, uncertainty: float) -> float:
        """
        Adjust probability based on uncertainty. Higher uncertainty means
        the probability is pulled closer to the center (0.5), meaning
        it is less extreme (less confident in a strong positive or negative).
        """
        # Ensure base_prob is float and handle potential None/NaN issues
        if not isinstance(base_prob, (int, float)):
            base_prob = 0.5 # Default if input is not numeric

        # The higher the uncertainty, the more `adjusted_factor` moves towards 0.5.
        # Adjusted = (prob * (1 - uncertainty_factor)) + (0.5 * uncertainty_factor)
        # This acts like a weighted average between the calculated probability and 0.5.
        
        # Example: if prob=0.9, uncert=0.2. adjusted = (0.9 * 0.8) + (0.5 * 0.2) = 0.72 + 0.1 = 0.82 (pulled from 0.9 to 0.82)
        # Example: if prob=0.1, uncert=0.2. adjusted = (0.1 * 0.8) + (0.5 * 0.2) = 0.08 + 0.1 = 0.18 (pulled from 0.1 to 0.18)
        
        adjusted_prob = (base_prob * (1.0 - uncertainty)) + (0.5 * uncertainty)
        return float(np.clip(adjusted_prob, 0.0, 1.0)) # Ensure it remains in [0, 1] range
    
    def _calculate_overall_uncertainty(self, uncertainty_factors: Dict[str, float]) -> float:
        """Calculate overall uncertainty level by taking the max to be conservative."""
        if not uncertainty_factors: return 1.0
        max_uncertainty = max(list(uncertainty_factors.values()))
        return float(np.clip(max_uncertainty * 1.1, 0.0, 0.5)) # Cap at 50% max for overall.
    
    def _calculate_confidence_intervals(self, copy_move_prob: float, splicing_prob: float, 
                                      authentic_prob: float, uncertainty_factors: Dict) -> Dict:
        """Calculate confidence intervals for each probability based on their adjusted uncertainty factors."""
        intervals = {}
        
        # The interval should reflect the applied uncertainty.
        # If probability P and uncertainty U, then the raw range might be P +/- U.
        # However, because we adjusted P using _apply_uncertainty, the range should reflect `how spread out` P could be
        # due to U, meaning the interval represents what the prob could have been if less certain.
        
        # A simpler way: The true confidence interval would be based on underlying probability distributions (e.g., beta distribution for probabilities),
        # but for this context, a heuristic based on uncertainty is fine.
        # Let's say the true interval width is 2*U. So `lower = max(0, P - U)` and `upper = min(1, P + U)`.

        # If a category is extremely low prob (e.g. 0.01), subtracting U might result in negative.
        # Max/min operations are essential here.

        cm_uncertainty = uncertainty_factors['copy_move']
        intervals['copy_move'] = {
            'lower': max(0.0, copy_move_prob - cm_uncertainty * 0.8), # Make intervals a bit tighter
            'upper': min(1.0, copy_move_prob + cm_uncertainty * 0.8)
        }
        
        sp_uncertainty = uncertainty_factors['splicing']
        intervals['splicing'] = {
            'lower': max(0.0, splicing_prob - sp_uncertainty * 0.8),
            'upper': min(1.0, splicing_prob + sp_uncertainty * 0.8)
        }
        
        au_uncertainty = uncertainty_factors['authentic']
        intervals['authentic'] = {
            'lower': max(0.0, authentic_prob - au_uncertainty * 0.8),
            'upper': min(1.0, authentic_prob + au_uncertainty * 0.8)
        }
        
        return intervals
    
    def generate_uncertainty_report(self, probabilities: Dict) -> Dict:
        """Generate detailed uncertainty report"""
        
        # Ensure all values are numeric floats, handle potential NaNs from `_apply_uncertainty` if base_prob was bad
        probabilities['copy_move_probability'] = float(probabilities.get('copy_move_probability', 0.0))
        probabilities['splicing_probability'] = float(probabilities.get('splicing_probability', 0.0))
        probabilities['authentic_probability'] = float(probabilities.get('authentic_probability', 0.0))
        probabilities['uncertainty_level'] = float(probabilities.get('uncertainty_level', 1.0))


        report = {
            'primary_assessment': self._determine_primary_assessment(probabilities),
            'confidence_level': self._determine_confidence_level(probabilities),
            'uncertainty_summary': self._summarize_uncertainty(probabilities['uncertainty_level']),
            'reliability_indicators': self._generate_reliability_indicators(probabilities),
            'recommendation': self._generate_recommendation(probabilities)
        }
        return report
    
    def _determine_primary_assessment(self, probabilities: Dict) -> str:
        """Determine primary assessment with uncertainty language"""
        copy_move_prob = probabilities['copy_move_probability']
        splicing_prob = probabilities['splicing_probability']
        authentic_prob = probabilities['authentic_probability']
        uncertainty = probabilities['uncertainty_level']
        
        # Consider an "undetected" result as authentic but with lower certainty.
        # This can be simplified after normalization, we just take the max probability.
        max_prob_value = max(copy_move_prob, splicing_prob, authentic_prob)
        
        # Using a slight threshold difference to prioritize distinct detection
        if max_prob_value < 0.4: # If no probability is clearly high, it's ambiguous
            return "Indikasi: Hasil Ambigu (membutuhkan pemeriksaan lebih lanjut)"
            
        # Determine specific manipulation type or authenticity
        is_authentic = (authentic_prob == max_prob_value) and (authentic_prob > copy_move_prob * 1.1 and authentic_prob > splicing_prob * 1.1)
        is_copy_move = (copy_move_prob == max_prob_value) and (copy_move_prob > authentic_prob * 1.1 and copy_move_prob > splicing_prob * 1.1)
        is_splicing = (splicing_prob == max_prob_value) and (splicing_prob > authentic_prob * 1.1 and splicing_prob > copy_move_prob * 1.1)

        # Cases: purely authentic, purely copy-move, purely splicing
        if is_authentic:
            return f"Indikasi: Gambar Asli/Autentik"
        elif is_copy_move:
            return f"Indikasi: Manipulasi Copy-Move Terdeteksi"
        elif is_splicing:
            return f"Indikasi: Manipulasi Splicing Terdeteksi"
        
        # Handle complex cases (both high or near equal, if not caught by pure thresholds)
        if copy_move_prob > 0.4 and splicing_prob > 0.4: # Both high but not clear winner by 1.1x rule
            return "Indikasi: Manipulasi Kompleks Terdeteksi (Copy-Move & Splicing)"
        
        # If all similar and not above ambiguous threshold
        return "Indikasi: Tidak Terdeteksi Manipulasi Signifikan"


    def _get_certainty_prefix(self, probability: float, uncertainty: float) -> str:
        """Get certainty prefix based on probability and uncertainty for more varied language."""
        if uncertainty >= 0.35:  # Very high uncertainty, results are speculative
            return "Kemungkinan kecil (spesulatif):"
        elif uncertainty >= 0.25: # High uncertainty
            if probability > 0.7:
                return "Terdapat indikasi:"
            else:
                return "Ada sedikit indikasi:"
        elif uncertainty >= 0.15: # Medium uncertainty
            if probability > 0.8:
                return "Cukup kuat mengindikasikan:"
            else:
                return "Ada kecenderungan:"
        else:  # Low uncertainty
            if probability > 0.9:
                return "Sangat kuat mengindikasikan:"
            elif probability > 0.75:
                return "Jelas mengindikasikan:"
            else:
                return "Ada bukti konsisten yang mengindikasikan:"

    
    def _determine_confidence_level(self, probabilities: Dict) -> str:
        """Determine overall confidence level based on main prediction's adjusted score AND uncertainty."""
        copy_move_prob = probabilities['copy_move_probability']
        splicing_prob = probabilities['splicing_probability']
        authentic_prob = probabilities['authentic_probability']
        uncertainty = probabilities['uncertainty_level']
        
        # Find the highest predicted probability.
        main_prob_value = max(copy_move_prob, splicing_prob, authentic_prob)

        # Combined Confidence: Scale highest probability by inverse of uncertainty.
        # This way, if P_max is high but Uncertainty is high, confidence drops.
        confidence_score = main_prob_value * (1.0 - uncertainty) # Final scaled score (0 to 1)

        if confidence_score >= self.confidence_thresholds['very_high']:
            return "Sangat Tinggi"
        elif confidence_score >= self.confidence_thresholds['high']:
            return "Tinggi"
        elif confidence_score >= self.confidence_thresholds['medium']:
            return "Sedang"
        elif confidence_score >= self.confidence_thresholds['low']:
            return "Rendah"
        else:
            return "Sangat Rendah"
    
    def _summarize_uncertainty(self, uncertainty_level: float) -> str:
        """Summarize uncertainty in human-readable form."""
        if uncertainty_level < 0.1:
            return "Sangat rendah: Bukti yang ditemukan sangat konsisten dan jelas."
        elif uncertainty_level < 0.2:
            return "Rendah: Sebagian besar bukti konsisten, ada sedikit ambiguitas."
        elif uncertainty_level < 0.3:
            return "Sedang: Terdapat beberapa indikator yang ambigu atau saling bertentangan. Perlu perhatian."
        elif uncertainty_level < 0.4:
            return "Tinggi: Banyak indikator tidak konsisten atau kurang kuat. Interpretasi hasilnya dengan sangat hati-hati."
        else: # >= 0.4
            return "Sangat tinggi: Bukti sangat lemah dan/atau sangat tidak konsisten. Hasil tidak dapat diandalkan tanpa investigasi lebih lanjut."
    
    def _generate_reliability_indicators(self, probabilities: Dict) -> List[str]:
        """Generate reliability indicators."""
        indicators = []
        
        probs = [
            probabilities['copy_move_probability'],
            probabilities['splicing_probability'],
            probabilities['authentic_probability']
        ]
        
        # Measure probability "peakiness"
        # Standard deviation high -> spread out -> low certainty on one single category
        prob_std = np.std(probs)
        
        if prob_std < 0.15:
            indicators.append("⚠️ Probabilitas antar kategori cukup dekat. Hasil mungkin ambigu.")
        elif prob_std > 0.35:
            indicators.append("✓ Probabilitas menunjukkan perbedaan yang jelas antar kategori. Prediksi spesifik lebih kuat.")
        else:
            indicators.append("• Probabilitas memiliki pemisahan sedang antar kategori.")
        
        # Check confidence intervals spread
        intervals = probabilities['confidence_intervals']
        for category, interval in intervals.items():
            range_size = interval['upper'] - interval['lower']
            if range_size > 0.5: # If the interval is wide, it implies higher uncertainty in that specific prediction
                indicators.append(f"⚠️ Interval kepercayaan untuk '{category}' sangat lebar ({range_size:.1%}). Prediksi kategori ini kurang stabil.")
        
        # Check overall uncertainty level
        if probabilities['uncertainty_level'] < 0.1:
            indicators.append("✓ Tingkat ketidakpastian sangat rendah, menunjukkan hasil yang sangat stabil.")
        elif probabilities['uncertainty_level'] > 0.3:
            indicators.append("❌ Tingkat ketidakpastian sangat tinggi. Pertimbangkan hasil dengan skeptisisme tinggi.")
        
        return indicators
    
    def _generate_recommendation(self, probabilities: Dict) -> str:
        """Generate recommendation based on analysis and uncertainty."""
        uncertainty = probabilities['uncertainty_level']
        
        # Main probability prediction value
        main_prob_value = max(probabilities['copy_move_probability'], probabilities['splicing_probability'], probabilities['authentic_probability'])
        
        # Confidence derived score (highest probability scaled by 1-uncertainty)
        confidence_derived_score = main_prob_value * (1.0 - uncertainty)

        # Thresholds for recommendation action points (adjust these based on desired strictness)
        if confidence_derived_score >= 0.75 and uncertainty < 0.2:
            return ("Hasil analisis menunjukkan indikasi yang kuat dan dapat diandalkan. "
                   "Direkomendasikan untuk dapat dijadikan referensi atau dasar investigasi lanjutan tanpa verifikasi manual yang ekstensif.")
        elif confidence_derived_score >= 0.55 or uncertainty < 0.3:
            return ("Hasil analisis memberikan indikasi yang cukup jelas, namun tetap perlu "
                   "dipertimbangkan bersama dengan konteks dan bukti lainnya. Verifikasi manual disarankan.")
        else: # Low confidence score or high uncertainty
            return ("Terdapat ketidakpastian yang signifikan atau bukti yang tidak cukup kuat. "
                   "Diperlukan analisis tambahan dengan metode lain dan/atau konsultasi ahli forensik digital untuk konfirmasi hasil.")

def format_probability_results(probabilities: Dict, details: Dict) -> str:
    """Format probability results for display"""
    output = []
    
    # Header
    output.append("="*60)
    output.append("HASIL ANALISIS FORENSIK GAMBAR")
    output.append("dengan Model Ketidakpastian Probabilistik")
    output.append("="*60)
    output.append("")
    
    # Primary Assessment
    output.append(f"PENILAIAN UTAMA: {details['primary_assessment']}")
    output.append(f"SKOR KEPERCAYAAN: {details['confidence_level']}")
    output.append("")
    
    # Probability Distribution
    output.append("DISTRIBUSI PROBABILITAS:")
    # Using string formatting with .1% for 1 decimal place percentage
    output.append(f"• Kemungkinan Asli/Autentik: {probabilities['authentic_probability']:.1%}")
    output.append(f"  Interval kepercayaan: [{probabilities['confidence_intervals']['authentic']['lower']:.1%} - "
                 f"{probabilities['confidence_intervals']['authentic']['upper']:.1%}]")
    
    output.append(f"• Kemungkinan Copy-Move: {probabilities['copy_move_probability']:.1%}")
    output.append(f"  Interval kepercayaan: [{probabilities['confidence_intervals']['copy_move']['lower']:.1%} - "
                 f"{probabilities['confidence_intervals']['copy_move']['upper']:.1%}]")
    
    output.append(f"• Kemungkinan Splicing: {probabilities['splicing_probability']:.1%}")
    output.append(f"  Interval kepercayaan: [{probabilities['confidence_intervals']['splicing']['lower']:.1%} - "
                 f"{probabilities['confidence_intervals']['splicing']['upper']:.1%}]")
    output.append("")
    
    # Uncertainty Analysis
    output.append("ANALISIS KETIDAKPASTIAN:")
    output.append(f"Tingkat ketidakpastian: {probabilities['uncertainty_level']:.1%}")
    output.append(details['uncertainty_summary'])
    output.append("")
    
    # Reliability Indicators
    output.append("INDIKATOR KEANDALAN:")
    if details['reliability_indicators']:
        for indicator in details['reliability_indicators']:
            output.append(indicator)
    else:
        output.append("Tidak ada indikator keandalan spesifik yang perlu disorot.")
    output.append("")
    
    # Recommendation
    output.append("REKOMENDASI:")
    output.append(details['recommendation'])
    output.append("")
    output.append("="*60)
    
    return "\n".join(output)

# --- END OF FILE uncertainty_classification.py ---