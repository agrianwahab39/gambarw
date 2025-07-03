"""
Configuration file for Forensic Image Analysis System
"""

# Analysis parameters
ELA_QUALITIES = [70, 80, 90, 95]
ELA_SCALE_FACTOR = 20
BLOCK_SIZE = 16
NOISE_BLOCK_SIZE = 32
TEXTURE_BLOCK_SIZE = 64

# Feature detection parameters
SIFT_FEATURES = 3000
SIFT_CONTRAST_THRESHOLD = 0.02
SIFT_EDGE_THRESHOLD = 10

ORB_FEATURES = 2000
ORB_SCALE_FACTOR = 1.2
ORB_LEVELS = 8

# Copy-move detection parameters
RATIO_THRESH = 0.7
MIN_DISTANCE = 40
RANSAC_THRESH = 5.0
MIN_INLIERS = 8

# Classification thresholds
DETECTION_THRESHOLD = 45
CONFIDENCE_THRESHOLD = 60

# File format support
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
MIN_FILE_SIZE = 50000  # 50KB

# Processing parameters
TARGET_MAX_DIM = 1500
MAX_SAMPLES_DBSCAN = 50000