"""
Feature detection and matching functions
"""

import numpy as np
import cv2
try:
    from sklearn.preprocessing import normalize as sk_normalize
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    def sk_normalize(arr, norm='l2', axis=1):
        denom = np.linalg.norm(arr, ord=2 if norm=='l2' else 1, axis=axis, keepdims=True)
        denom[denom == 0] = 1
        return arr / denom
from config import *

def extract_multi_detector_features(image_pil, ela_image_pil, ela_mean, ela_stddev):
    """Extract features using multiple detectors (SIFT, ORB, SURF)"""
    ela_np = np.array(ela_image_pil)
    
    # Dynamic thresholding
    threshold = ela_mean + 1.5 * ela_stddev
    threshold = max(min(threshold, 180), 30)
    
    # Enhanced ROI mask
    roi_mask = (ela_np > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
    
    # Convert to grayscale with enhancement
    original_image_np = np.array(image_pil.convert('RGB'))
    gray_original = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2GRAY)
    
    # Multiple enhancement techniques
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray_original)
    
    # Extract features using multiple detectors
    feature_sets = {}
    
    # 1. SIFT
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES, 
                          contrastThreshold=SIFT_CONTRAST_THRESHOLD, 
                          edgeThreshold=SIFT_EDGE_THRESHOLD)
    kp_sift, desc_sift = sift.detectAndCompute(gray_enhanced, mask=roi_mask)
    feature_sets['sift'] = (kp_sift, desc_sift)
    
    # 2. ORB
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, 
                        scaleFactor=ORB_SCALE_FACTOR, 
                        nlevels=ORB_LEVELS)
    kp_orb, desc_orb = orb.detectAndCompute(gray_enhanced, mask=roi_mask)
    feature_sets['orb'] = (kp_orb, desc_orb)
    
    # 3. AKAZE
    try:
        akaze = cv2.AKAZE_create()
        kp_akaze, desc_akaze = akaze.detectAndCompute(gray_enhanced, mask=roi_mask)
        feature_sets['akaze'] = (kp_akaze, desc_akaze)
    except:
        feature_sets['akaze'] = ([], None)
    
    return feature_sets, roi_mask, gray_enhanced

def match_sift_features(keypoints, descriptors, ratio_thresh, min_distance, ransac_thresh, min_inliers):
    """Enhanced SIFT matching"""
    # Handle empty descriptors
    if descriptors is None or len(descriptors) == 0:
        return [], 0, None
        
    descriptors_norm = sk_normalize(descriptors, norm='l2', axis=1)
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors_norm, descriptors_norm, k=8)
    
    good_matches = []
    match_pairs = []
    
    for i, match_list in enumerate(matches):
        for m in match_list[1:]:  # Skip self-match
            pt1 = keypoints[i].pt
            pt2 = keypoints[m.trainIdx].pt
            
            spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            if spatial_dist > min_distance and m.distance < ratio_thresh:
                good_matches.append(m)
                match_pairs.append((i, m.trainIdx))
    
    if len(match_pairs) < min_inliers:
        return good_matches, 0, None
    
    # RANSAC verification
    src_pts = np.float32([keypoints[i].pt for i, _ in match_pairs]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[j].pt for _, j in match_pairs]).reshape(-1, 1, 2)
    
    best_inliers = 0
    best_transform = None
    best_mask = None
    
    # Try different transformations
    for transform_type in ['affine', 'homography', 'similarity']:
        try:
            if transform_type == 'affine':
                M, mask = cv2.estimateAffine2D(src_pts, dst_pts,
                                             method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh)
            elif transform_type == 'homography':
                M, mask = cv2.findHomography(src_pts, dst_pts,
                                           cv2.RANSAC, ransac_thresh)
            else:  # similarity
                M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                                    method=cv2.RANSAC,
                                                    ransacReprojThreshold=ransac_thresh)
            
            if M is not None:
                inliers = np.sum(mask)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_transform = (transform_type, M)
                    best_mask = mask
        except:
            continue
    
    if best_mask is not None and best_inliers >= min_inliers:
        ransac_matches = [good_matches[i] for i in range(len(good_matches))
                         if best_mask[i][0] == 1]
        return ransac_matches, best_inliers, best_transform
    
    return good_matches, 0, None

def match_orb_features(keypoints, descriptors, min_distance, ransac_thresh, min_inliers):
    """ORB feature matching"""
    # Hamming distance matcher for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=6)
    
    good_matches = []
    match_pairs = []
    
    for i, match_list in enumerate(matches):
        for m in match_list[1:]:  # Skip self-match
            pt1 = keypoints[i].pt
            pt2 = keypoints[m.trainIdx].pt
            
            spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            if spatial_dist > min_distance and m.distance < 80:  # Hamming distance threshold
                good_matches.append(m)
                match_pairs.append((i, m.trainIdx))
    
    if len(match_pairs) < min_inliers:
        return good_matches, 0, None
    
    # Simple geometric verification
    return good_matches, len(match_pairs), ('orb_matches', None)

def match_akaze_features(keypoints, descriptors, min_distance, ransac_thresh, min_inliers):
    """AKAZE feature matching"""
    if descriptors is None:
        return [], 0, None
    
    # Hamming distance for AKAZE
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=6)
    
    good_matches = []
    match_pairs = []
    
    for i, match_list in enumerate(matches):
        if len(match_list) > 1:
            for m in match_list[1:]:  # Skip self-match
                pt1 = keypoints[i].pt
                pt2 = keypoints[m.trainIdx].pt
                
                spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                
                if spatial_dist > min_distance and m.distance < 100:
                    good_matches.append(m)
                    match_pairs.append((i, m.trainIdx))
    
    return good_matches, len(match_pairs), ('akaze_matches', None)