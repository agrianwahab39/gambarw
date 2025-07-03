# validator.py

from PIL import Image
import numpy as np # Import numpy

# Diambil dari app2.py
class ForensicValidator:
    def __init__(self):
        # Bobot algoritma (harus berjumlah 1.0)
        self.weights = {
            'clustering': 0.30,  # K-Means (metode utama)
            'localization': 0.30,  # Lokalisasi tampering (metode utama)
            'ela': 0.20,  # Error Level Analysis (metode pendukung)
            'feature_matching': 0.20,  # SIFT (metode pendukung)
        }
        
        # Threshold minimum untuk setiap teknik (0-1 scale)
        self.thresholds = {
            'clustering': 0.60,
            'localization': 0.60,
            'ela': 0.60,
            'feature_matching': 0.60,
        }
    
    def validate_clustering(self, analysis_results):
        """Validasi kualitas clustering K-Means"""
        # Access `kmeans_localization` inside `localization_analysis`
        kmeans_data = analysis_results.get('localization_analysis', {}).get('kmeans_localization', {})

        if not kmeans_data or 'cluster_ela_means' not in kmeans_data:
            return 0.0, "Data clustering tidak tersedia atau tidak lengkap"
            
        cluster_ela_means = kmeans_data.get('cluster_ela_means', [])
        cluster_count = len(cluster_ela_means)

        if cluster_count < 2:
            return 0.4, "Diferensiasi cluster tidak memadai (kurang dari 2 cluster teridentifikasi)"
            
        # 2. Periksa pemisahan cluster (semakin tinggi selisih mean ELA antar cluster semakin baik)
        mean_diff = max(cluster_ela_means) - min(cluster_ela_means) if cluster_ela_means else 0
        mean_diff_score = min(1.0, mean_diff / 20.0)  # Normalisasi: a diff of 20 implies a score of 1.0
        
        # 3. Periksa identifikasi cluster tampering (jika ada cluster dengan ELA tinggi yang ditandai)
        tampering_cluster_id = kmeans_data.get('tampering_cluster_id', -1)
        tampering_identified = (tampering_cluster_id >= 0 and tampering_cluster_id < cluster_count and cluster_ela_means[tampering_cluster_id] > 5) # ELA mean of identified cluster should be somewhat high
        
        # 4. Periksa area tampering berukuran wajar (tidak terlalu kecil atau terlalu besar)
        tampering_pct = analysis_results.get('localization_analysis', {}).get('tampering_percentage', 0)
        size_score = 0.0
        if 1.0 < tampering_pct < 50.0:  # Ideal size range for actual tampering
            size_score = 1.0
        elif tampering_pct <= 1.0 and tampering_pct > 0.0:  # Too small but exists
            size_score = tampering_pct # linear interpolation from 0 to 1
        elif tampering_pct >= 50.0: # Too large (might be global effect or full image replacement)
            size_score = max(0.0, 1.0 - ((tampering_pct - 50) / 50.0)) # Linear falloff from 1.0 to 0.0 for 50-100%

        # Gabungkan skor dengan faktor berbobot
        confidence = (
            0.3 * min(cluster_count / 5.0, 1.0)  # Up to 5 clusters, more means better differentiation up to a point
            + 0.3 * mean_diff_score
            + 0.2 * float(tampering_identified)
            + 0.2 * size_score
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        details = (
            f"Jumlah cluster: {cluster_count}, "
            f"Pemisahan cluster (Max-Min ELA): {mean_diff:.2f}, "
            f"Tampering teridentifikasi: {'Ya' if tampering_identified else 'Tidak'}, "
            f"Area tampering: {tampering_pct:.1f}%"
        )
        
        return confidence, details
    
    def validate_localization(self, analysis_results):
        """Validasi efektivitas lokalisasi tampering"""
        localization_data = analysis_results.get('localization_analysis', {})

        if not localization_data:
            return 0.0, "Data lokalisasi tidak tersedia"
            
        # 1. Periksa apakah mask tampering yang digabungkan telah dihasilkan
        has_combined_mask = 'combined_tampering_mask' in localization_data and localization_data['combined_tampering_mask'] is not None and localization_data['combined_tampering_mask'].size > 0
        if not has_combined_mask:
            return 0.0, "Tidak ada mask tampering gabungan yang dihasilkan"
            
        # 2. Periksa persentase area yang ditandai (harus wajar untuk manipulasi)
        tampering_pct = localization_data.get('tampering_percentage', 0.0)
        area_score = 0.0
        if 0.5 < tampering_pct < 40.0:  # Common range for effective tampering, neither too small nor too large
            area_score = 1.0
        elif 0.0 < tampering_pct <= 0.5:  # Too small, might be noise
            area_score = tampering_pct / 0.5 # Scale from 0 to 1 as it gets to 0.5%
        else: # tampering_pct >= 40.0: # Too large, could be entire image replaced or a global filter
            area_score = max(0.0, 1.0 - ((tampering_pct - 40.0) / 60.0)) # Drops from 1 to 0 for 40% to 100%
        
        # 3. Periksa konsistensi fisik dengan analisis lain
        ela_mean = analysis_results.get('ela_mean', 0.0)
        ela_std = analysis_results.get('ela_std', 0.0)
        noise_inconsistency = analysis_results.get('noise_analysis', {}).get('overall_inconsistency', 0.0)
        jpeg_ghost_ratio = analysis_results.get('jpeg_ghost_suspicious_ratio', 0.0) # Check this exists
        
        # High ELA means stronger splicing signal in general.
        ela_consistency = min(1.0, max(0.0, (ela_mean - 5.0) / 10.0)) # Scores 0 at ELA mean 5, 1 at 15
        ela_consistency = ela_consistency * min(1.0, max(0.0, (ela_std - 10.0) / 15.0)) # Add std influence (scores 0 at 10, 1 at 25)

        # High noise inconsistency (for areas, or globally near manipulated regions)
        noise_consistency = min(1.0, max(0.0, (noise_inconsistency - 0.1) / 0.3)) # Scores 0 at 0.1, 1 at 0.4

        # High JPEG ghost ratio
        jpeg_consistency = min(1.0, max(0.0, jpeg_ghost_ratio / 0.2)) # Scores 0 at 0, 1 at 0.2

        # Combine physical consistency. Max implies if one is strong, it still lends credence.
        physical_consistency = max(ela_consistency, noise_consistency, jpeg_consistency)
        
        # Skor gabungan dengan faktor berbobot
        confidence = (
            0.4 * float(has_combined_mask) # Must have a mask
            + 0.3 * area_score # Quality of area percentage
            + 0.3 * physical_consistency # Agreement with other physical anomalies
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Kalibrasi ke rentang [0,1]
        
        details = (
            f"Mask tampering: {'Ada' if has_combined_mask else 'Tidak ada'}, "
            f"Persentase area: {tampering_pct:.1f}%, "
            f"Konsistensi ELA (Mean, Std): {ela_consistency:.2f}, "
            f"Konsistensi noise: {noise_consistency:.2f}, "
            f"JPEG ghost: {jpeg_consistency:.2f}"
        )
        
        return confidence, details
    
    def validate_ela(self, analysis_results):
        """Validasi kualitas Error Level Analysis"""
        ela_image_obj = analysis_results.get('ela_image')
        # Check if ela_image object itself is a valid PIL Image or has image-like properties that can be converted
        if ela_image_obj is None or (not isinstance(ela_image_obj, Image.Image) and not hasattr(ela_image_obj, 'size') and not hasattr(ela_image_obj, 'ndim')):
             return 0.0, "Tidak ada gambar ELA yang tersedia atau format tidak valid"
            
        ela_mean = analysis_results.get('ela_mean', 0.0)
        ela_std = analysis_results.get('ela_std', 0.0)
        
        # 1. Normalisasi ELA mean (tinggi ~manipulasi, sangat rendah ~normal, tengah ~ambigu)
        # Penalize values very low and values very high (as sometimes artifacts are subtle)
        # Score higher for ambiguous/mid-range or distinctly high ELA
        mean_score = 0.0
        if 5.0 <= ela_mean <= 20.0: # Good range for detection
            mean_score = 1.0
        elif ela_mean > 20.0: # Very high, might be over-exposure or unusual image, can reduce score slightly
            mean_score = max(0.0, 1.0 - (ela_mean - 20.0) / 10.0)
        elif ela_mean < 5.0 and ela_mean > 0.0: # Too low, harder to confirm manipulation with ELA
            mean_score = ela_mean / 5.0 # Scales 0 to 1 up to ELA mean 5.0
        
        # 2. Normalisasi ELA std (tinggi ~inkonsistensi)
        std_score = 0.0
        if 10.0 <= ela_std <= 30.0: # Good range for std indicating inconsistency
            std_score = 1.0
        elif ela_std > 30.0: # Too high
            std_score = max(0.0, 1.0 - (ela_std - 30.0) / 10.0)
        elif ela_std < 10.0 and ela_std > 0.0: # Too low, likely uniform or not complex manipulation
            std_score = ela_std / 10.0
            
        # 2. Periksa inkonsistensi regional (more significant if outliers or high inconsistency)
        regional_stats = analysis_results.get('ela_regional_stats', {})
        regional_inconsistency = regional_stats.get('regional_inconsistency', 0.0)
        outlier_regions = regional_stats.get('outlier_regions', 0)
        
        inconsistency_score = min(1.0, regional_inconsistency / 0.5) # Normalized score (0.5 means full score)
        outlier_score = min(1.0, outlier_regions / 5.0) # Up to 5 outlier regions is full score
        
        # 3. Periksa metrik kualitas ELA (response across qualities)
        quality_stats = analysis_results.get('ela_quality_stats', [])
        quality_variation = 0.0
        if quality_stats:
            means = [q.get('mean', 0) for q in quality_stats if 'mean' in q]
            if len(means) > 1:
                # The maximum difference between mean responses across qualities
                quality_variation = max(means) - min(means) 
                quality_variation = min(1.0, quality_variation / 15.0) # Normalizes if total variation over 15

        # Gabungkan skor dengan bobot
        confidence = (
            0.3 * mean_score
            + 0.2 * std_score
            + 0.2 * inconsistency_score
            + 0.2 * outlier_score
            + 0.1 * quality_variation
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        details = (
            f"ELA mean: {ela_mean:.2f} (score: {mean_score:.2f}), "
            f"ELA std: {ela_std:.2f} (score: {std_score:.2f}), "
            f"Inkonsistensi regional: {regional_inconsistency:.3f} (score: {inconsistency_score:.2f}), "
            f"Region outlier: {outlier_regions} (score: {outlier_score:.2f})"
        )
        
        return confidence, details
    
    def validate_feature_matching(self, analysis_results):
        """Validasi kualitas pencocokan fitur SIFT/ORB/AKAZE"""
        ransac_inliers = analysis_results.get('ransac_inliers', 0)
        sift_matches = analysis_results.get('sift_matches', 0) # Raw matches before RANSAC
        
        # Ensure ransac_inliers are not negative, if by any error they were introduced (should be handled earlier)
        if ransac_inliers < 0: ransac_inliers = 0
            
        if sift_matches < 5: # Minimal raw matches needed
             return 0.0, "Tidak ada data pencocokan fitur yang signifikan (kurang dari 5 raw matches)"
            
        # 1. Periksa kecocokan yang signifikan (RANSAC inliers sebagai indikator kuat)
        # Normalisasi inlier: A good amount (e.g. 20-30 inliers) is strong.
        inlier_score = min(1.0, ransac_inliers / 25.0) # Score 1.0 at 25 inliers
        
        # Raw matches count (provides context for potential matching opportunities)
        match_score = min(1.0, sift_matches / 150.0) # Score 1.0 at 150 raw matches
        
        # 2. Periksa transformasi geometris yang ditemukan oleh RANSAC
        has_transform = analysis_results.get('geometric_transform') is not None
        transform_type = None
        if has_transform: # geometric_transform format is (type_string, matrix)
            try: # Robust access for tuple/list
                transform_type = analysis_results['geometric_transform'][0]
            except (TypeError, IndexError): # In case it's not a tuple or is empty
                transform_type = "Unknown_Type"
        
        # 3. Periksa kecocokan blok (harus berkorelasi dengan kecocokan fitur untuk copy-move)
        block_matches = len(analysis_results.get('block_matches', []))
        block_score = min(1.0, block_matches / 15.0) # Score 1.0 at 15 block matches
        
        # Cross-algorithm correlation: High RANSAC and Block matches
        correlation_score = 0.0
        if ransac_inliers > 10 and block_matches > 5: # Both strong: high correlation
            correlation_score = 1.0
        elif ransac_inliers > 0 and block_matches > 0: # Both exist: some correlation
            correlation_score = 0.5
        
        # Gabungkan skor dengan bobot
        confidence = (
            0.35 * inlier_score # Highest weight for RANSAC inliers
            + 0.20 * match_score # Medium for overall matches
            + 0.20 * float(has_transform) # Medium for detecting transform type
            + 0.10 * block_score # Lower for general block matching
            + 0.15 * correlation_score # Consistency score between two detection methods
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        details = (
            f"RANSAC inliers: {ransac_inliers} (score: {inlier_score:.2f}), "
            f"Raw SIFT matches: {sift_matches} (score: {match_score:.2f}), "
            f"Tipe transformasi: {transform_type if transform_type else 'Tidak ada'}, "
            f"Kecocokan blok: {block_matches} (score: {block_score:.2f})"
        )
        
        return confidence, details
    
    def validate_cross_algorithm(self, analysis_results):
        """Validasi konsistensi silang algoritma"""
        if not analysis_results:
            return [], 0.0, "Tidak ada hasil analisis yang tersedia", []
        
        validation_results = {}
        for technique, validate_func in [
            ('clustering', self.validate_clustering),
            ('localization', self.validate_localization),
            ('ela', self.validate_ela),
            ('feature_matching', self.validate_feature_matching)
        ]:
            confidence, details = validate_func(analysis_results)
            # Ensure confidence is a float, especially important from fallback paths
            confidence = float(confidence)
            validation_results[technique] = {
                'confidence': confidence,
                'details': details,
                'weight': self.weights[technique],
                'threshold': self.thresholds[technique],
                'passed': confidence >= self.thresholds[technique]
            }
        
        # Prepare textual results for console/logging
        process_results_list = []
        
        for technique, result in validation_results.items():
            status = "[LULUS]" if result['passed'] else "[GAGAL]"
            emoji = "✅" if result['passed'] else "❌"
            process_results_list.append(f"{emoji} {status:10} | Validasi {technique.capitalize()} - Skor: {result['confidence']:.2f}")
            
        # Calculate weighted individual technique scores
        weighted_scores = {
            technique: result['confidence'] * result['weight']
            for technique, result in validation_results.items()
        }
        
        # Calculate inter-algorithm agreement ratio
        agreement_pairs = 0
        total_pairs = 0
        techniques_list = list(validation_results.keys()) # Convert to list to iterate
        
        for i in range(len(techniques_list)):
            for j in range(i + 1, len(techniques_list)):
                t1, t2 = techniques_list[i], techniques_list[j]
                total_pairs += 1
                # If both passed or both failed, they "agree"
                if validation_results[t1]['passed'] == validation_results[t2]['passed']:
                    agreement_pairs += 1
        
        if total_pairs > 0:
            agreement_ratio = float(agreement_pairs) / total_pairs
        else: # Handle case of 0 total pairs (e.g., less than 2 techniques or specific edge cases)
            agreement_ratio = 1.0 # If nothing to compare, assume perfect agreement logically
        
        # Combine weighted score and agreement bonus
        raw_weighted_total = sum(weighted_scores.values())
        consensus_boost = agreement_ratio * 0.10 # Add max 10% bonus for perfect agreement (tuned)
        
        final_score = (raw_weighted_total * 100) + (consensus_boost * 100)
        
        # Clamp final score between 0 and 100
        final_score = min(100.0, max(0.0, final_score))
        
        # Collect failed validations for detailed reporting
        failed_validations_detail = [
            {
                'name': f"Validasi {technique.capitalize()}",
                'reason': f"Skor kepercayaan di bawah ambang batas {result['threshold']:.2f}",
                'rule': f"LULUS = (Kepercayaan >= {result['threshold']:.2f})",
                'values': f"Nilai aktual: Kepercayaan = {result['confidence']:.2f}\nDetail: {result['details']}"
            }
            for technique, result in validation_results.items()
            if not result['passed']
        ]
        
        # Determine confidence level description for summary text
        if final_score >= 95:
            confidence_level = "Sangat Tinggi (Very High)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Semua metode analisis menunjukkan konsistensi dan kualitas tinggi."
        elif final_score >= 90:
            confidence_level = "Tinggi (High)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Sebagian besar metode analisis menunjukkan konsistensi dan kualitas baik."
        elif final_score >= 85:
            confidence_level = "Sedang (Medium)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Beberapa metode analisis menunjukkan inkonsistensi minor."
        else:
            confidence_level = "Rendah (Low)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Terdapat inkonsistensi signifikan antar metode analisis yang memerlukan perhatian."
        
        return process_results_list, final_score, summary_text, failed_validations_detail

# --- END OF FILE validator.py ---