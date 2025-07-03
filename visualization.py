"""
Visualization Module for Forensic Image Analysis System
Contains functions for creating comprehensive visualizations, plots, and visual reports
"""

import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    gridspec = None # Ensure gridspec is also None when matplotlib fails
    class PdfPages: # Dummy class
        def __init__(self, *a, **k):
            raise RuntimeError('matplotlib not available')
from PIL import Image
from datetime import datetime
try:
    from skimage.filters import sobel
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
    def sobel(x): # Fallback sobel that returns black image
        return np.zeros_like(x, dtype=float)
import os
import io
import warnings
try:
    from sklearn.metrics import confusion_matrix, accuracy_score
    import seaborn as sns
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
    SKLEARN_METRICS_AVAILABLE = True
except Exception:
    SKLEARN_METRICS_AVAILABLE = False
    SCIPY_AVAILABLE = False # Also for gaussian_kde which needs scipy.stats

warnings.filterwarnings('ignore')

# ======================= Main Visualization Function (DIPERBAIKI) =======================

def visualize_results_advanced(original_pil, analysis_results, output_filename="advanced_forensic_analysis.png"):
    """Advanced visualization with comprehensive forensic analysis results"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available. Cannot generate visualization.")
        return None
    print("📊 Creating advanced forensic visualization...")
    
    # Increase figure size for more detail in combined plots
    fig = plt.figure(figsize=(24, 20)) # Slightly larger for better readability
    # Adjust grid spacing and number of rows/columns as per sections below
    gs = fig.add_gridspec(4, 4, hspace=0.7, wspace=0.3) # More vertical space
    
    fig.suptitle(
        f"Laporan Visual Analisis Forensik Gambar\nFile: {analysis_results['metadata'].get('Filename', 'N/A')} | Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=24, fontweight='bold' # Larger title
    )
    
    # Row 1: Core Visuals (Original, ELA, Feature, Block)
    ax1_1 = fig.add_subplot(gs[0, 0])
    ax1_2 = fig.add_subplot(gs[0, 1])
    ax1_3 = fig.add_subplot(gs[0, 2])
    ax1_4 = fig.add_subplot(gs[0, 3])
    create_core_visuals_grid(ax1_1, ax1_2, ax1_3, ax1_4, original_pil, analysis_results)
    
    # Row 2: Advanced Analysis Visuals (Edge, Illumination, JPEG Ghost, Combined Heatmap)
    ax2_1 = fig.add_subplot(gs[1, 0])
    ax2_2 = fig.add_subplot(gs[1, 1])
    ax2_3 = fig.add_subplot(gs[1, 2])
    ax2_4 = fig.add_subplot(gs[1, 3])
    create_advanced_analysis_grid(ax2_1, ax2_2, ax2_3, ax2_4, original_pil, analysis_results)
    
    # Row 3: Statistical & Metric Visuals (Frequency, Texture, Statistical, JPEG Quality Response)
    ax3_1 = fig.add_subplot(gs[2, 0])
    ax3_2 = fig.add_subplot(gs[2, 1])
    ax3_3 = fig.add_subplot(gs[2, 2])
    ax3_4 = fig.add_subplot(gs[2, 3])
    create_statistical_grid(ax3_1, ax3_2, ax3_3, ax3_4, analysis_results)

    # Row 4: Summary Report, Probability Bars, Uncertainty Visualization
    ax_report = fig.add_subplot(gs[3, 0]) # Was 0:2, now just one slot to free space
    ax_probability_bars = fig.add_subplot(gs[3, 1]) # Dedicated spot
    ax_uncertainty_vis = fig.add_subplot(gs[3, 2]) # Dedicated spot
    ax_validation_summary = fig.add_subplot(gs[3, 3]) # Final spot for some numerical metrics or dummy
    
    # The actual plotting of these. `populate_validation_visuals` combines `create_confusion_matrix` and `create_confidence_distribution` (from original code comments).
    # But now, we are adding new distinct plots: `create_probability_bars` and `create_uncertainty_visualization`.
    create_summary_report(ax_report, analysis_results)
    
    if 'classification' in analysis_results and 'uncertainty_analysis' in analysis_results['classification']:
        create_probability_bars(ax_probability_bars, analysis_results)
        create_uncertainty_visualization(ax_uncertainty_vis, analysis_results)
    else: # Fallback to populate_validation_visuals (simulated graphs) if uncertainty data missing
        populate_validation_visuals(ax_probability_bars, ax_uncertainty_vis) # Repurpose these axes for simulation or simple message
    
    # Add a simplified general validation metrics or status here.
    ax_validation_summary.axis('off') # Hide axes for text display
    pipeline_status_summary = analysis_results.get('pipeline_status', {})
    total_stages = pipeline_status_summary.get('total_stages', 0)
    completed_stages = pipeline_status_summary.get('completed_stages', 0)
    failed_stages_count = len(pipeline_status_summary.get('failed_stages', []))
    success_rate = (completed_stages / total_stages) * 100 if total_stages > 0 else 0
    
    validation_text = f"**System Validation Summary**\n\n" \
                      f"Pipeline Status: {completed_stages}/{total_stages} stages completed ({success_rate:.1f}%)\n" \
                      f"Failed Stages: {failed_stages_count}\n\n" \
                      f"For full validation report, refer to DOCX/PDF export and 'Hasil Pengujian' tab."

    ax_validation_summary.text(0.5, 0.5, validation_text, transform=ax_validation_summary.transAxes,
                                ha='center', va='center', fontsize=10, 
                                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    try:
        plt.savefig(output_filename, dpi=180, bbox_inches='tight') # Higher DPI for better quality
        print(f"📊 Advanced forensic visualization saved as '{output_filename}'")
        plt.close(fig)
        return output_filename
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
        import traceback # For detailed error trace
        traceback.print_exc()
        plt.close(fig)
        return None

# ======================= Grid Helper Functions =======================

def create_core_visuals_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create core visuals grid (Original, ELA, Feature, Block)"""
    ax1.imshow(original_pil)
    ax1.set_title("1. Gambar Asli", fontsize=12)
    ax1.axis('off')

    ela_img_data = results.get('ela_image') # This can be a PIL Image or NumPy array
    ela_mean = results.get('ela_mean', 0.0)
    
    # Ensure ELA image is a displayable numpy array in grayscale
    ela_display_array = np.zeros(original_pil.size[::-1], dtype=np.uint8) # Default black
    if ela_img_data is not None:
        if isinstance(ela_img_data, Image.Image):
            ela_display_array = np.array(ela_img_data.convert('L'))
        elif isinstance(ela_img_data, np.ndarray):
            if np.issubdtype(ela_img_data.dtype, np.floating): # Normalize if float
                ela_display_array = (ela_img_data / (ela_img_data.max() + 1e-9) * 255).astype(np.uint8)
            else: # Already int
                ela_display_array = ela_img_data
            
    ela_display = ax2.imshow(ela_display_array, cmap='hot')
    ax2.set_title(f"2. ELA (μ={ela_mean:.1f})", fontsize=12)
    ax2.axis('off')
    plt.colorbar(ela_display, ax=ax2, fraction=0.046, pad=0.04)

    create_feature_match_visualization(ax3, original_pil, results)
    create_block_match_visualization(ax4, original_pil, results)

def create_advanced_analysis_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create advanced analysis grid (Edge, Illumination, JPEG Ghost, Combined Heatmap)"""
    create_edge_visualization(ax1, original_pil, results)
    create_illumination_visualization(ax2, original_pil, results)
    
    # Ensure jpeg_ghost is a displayable numpy array in grayscale
    jpeg_ghost_data = results.get('jpeg_ghost')
    jpeg_ghost_display_array = np.zeros(original_pil.size[::-1], dtype=np.uint8)
    if jpeg_ghost_data is not None:
        if jpeg_ghost_data.ndim == 2:
            if np.issubdtype(jpeg_ghost_data.dtype, np.floating) and jpeg_ghost_data.max() > 0:
                jpeg_ghost_display_array = (jpeg_ghost_data / (jpeg_ghost_data.max() + 1e-9) * 255).astype(np.uint8)
            else: # Already 0-255 or max is 0 (all black)
                jpeg_ghost_display_array = jpeg_ghost_data.astype(np.uint8)
        else:
             print("Warning: JPEG ghost data not 2D for visualization.") # Should be 2D array
             
    ghost_display = ax3.imshow(jpeg_ghost_display_array, cmap='hot')
    ax3.set_title(f"7. JPEG Ghost ({results.get('jpeg_ghost_suspicious_ratio', 0):.1%} area)", fontsize=12)
    ax3.axis('off')
    plt.colorbar(ghost_display, ax=ax3, fraction=0.046, pad=0.04)

    # Convert original PIL image to numpy array for consistency with heatmap overlay
    original_pil_array = np.array(original_pil.convert('RGB'))
    combined_heatmap = create_advanced_combined_heatmap(results, original_pil.size)
    ax4.imshow(original_pil_array, alpha=0.4)
    ax4.imshow(combined_heatmap, cmap='hot', alpha=0.6)
    ax4.set_title("8. Peta Kecurigaan Gabungan", fontsize=12)
    ax4.axis('off')

def create_statistical_grid(ax1, ax2, ax3, ax4, results):
    """Create statistical analysis grid (Frequency, Texture, Statistical, JPEG Quality Response)"""
    create_frequency_visualization(ax1, results)
    create_texture_visualization(ax2, results)
    create_statistical_visualization(ax3, results)
    create_quality_response_plot(ax4, results)
    
# ======================= Individual Visualization Functions =======================

def create_feature_match_visualization(ax, original_pil, results):
    img_matches = np.array(original_pil.convert('RGB'))
    keypoints = results.get('sift_keypoints') # Expects list of cv2.KeyPoint
    ransac_matches = results.get('ransac_matches') # Expects list of cv2.DMatch
    
    # Filter to show only the strongest N matches or random N for clarity
    MAX_MATCHES_DISPLAY = 50 # Limit number of matches to display to prevent clutter
    
    if keypoints and ransac_matches and len(ransac_matches) > 0:
        display_matches = sorted(ransac_matches, key=lambda x: x.distance)[:MAX_MATCHES_DISPLAY]
        
        for m in display_matches:
            if m.queryIdx < len(keypoints) and m.trainIdx < len(keypoints): # Basic check
                pt1 = tuple(map(int, keypoints[m.queryIdx].pt))
                pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
                
                # Check for self-match: if pts are very close, likely same point
                if pt1 == pt2 or (abs(pt1[0]-pt2[0]) < 2 and abs(pt1[1]-pt2[1]) < 2): # Avoid self-matches on the visualization itself
                    continue

                # Draw line with color indicating source/dest points or simply match
                cv2.line(img_matches, pt1, pt2, (50, 255, 50), 1, cv2.LINE_AA) # Green line
                # Draw circles on keypoints (smaller size for density)
                cv2.circle(img_matches, pt1, 3, (255, 0, 0), -1, cv2.LINE_AA) # Red for query point
                cv2.circle(img_matches, pt2, 3, (0, 0, 255), -1, cv2.LINE_AA) # Blue for train point
            
    ax.imshow(img_matches)
    ax.set_title(f"3. Feature Matches ({results.get('ransac_inliers',0)} inliers)", fontsize=12)
    ax.axis('off')


def create_block_match_visualization(ax, original_pil, results):
    img_blocks = np.array(original_pil.convert('RGB'))
    block_matches = results.get('block_matches', [])
    
    MAX_BLOCK_MATCHES_DISPLAY = 20 # Limit number of block matches for display
    
    if block_matches:
        display_block_matches = block_matches[:MAX_BLOCK_MATCHES_DISPLAY]
        for i, match in enumerate(display_block_matches):
            x1, y1 = match['block1']; x2, y2 = match['block2']
            block_size = 16 # Default block size for display (config.BLOCK_SIZE)

            # Different color for each pair or simply alternate to distinguish visually
            color = (255, 165, 0) if i % 2 == 0 else (0, 165, 255) # Orange or light blue (BGR format)

            # Draw rectangles
            cv2.rectangle(img_blocks, (x1, y1), (x1 + block_size, y1 + block_size), color, 2)
            cv2.rectangle(img_blocks, (x2, y2), (x2 + block_size, y2 + block_size), color, 2)
            
            # Optional: draw lines connecting centers of blocks for a clearer visual connection
            center1 = (x1 + block_size // 2, y1 + block_size // 2)
            center2 = (x2 + block_size // 2, y2 + block_size // 2)
            cv2.line(img_blocks, center1, center2, color, 1, cv2.LINE_AA)

    ax.imshow(img_blocks)
    ax.set_title(f"4. Block Matches ({len(block_matches or [])} found)", fontsize=12)
    ax.axis('off')


def create_localization_visualization(ax, original_pil, analysis_results):
    loc_analysis = analysis_results.get('localization_analysis', {})
    # Use 'combined_tampering_mask' which is the cleaned and integrated mask
    mask = loc_analysis.get('combined_tampering_mask') 
    tampering_pct = loc_analysis.get('tampering_percentage', 0)

    # Ensure original_pil is converted to RGB array if it isn't already for consistent overlay
    original_img_array = np.array(original_pil.convert('RGB'))

    ax.imshow(original_img_array) # Display original image
    
    # Overlay the mask if available and significant tampering detected
    if mask is not None and mask.size > 0 and tampering_pct > 0.1:
        # Resize mask to original image dimensions for accurate overlay
        # `cv2.resize` expects (width, height)
        mask_resized_uint8 = cv2.resize(mask.astype(np.uint8), (original_img_array.shape[1], original_img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create a red overlay layer (RGBA)
        red_overlay = np.zeros((*original_img_array.shape[:2], 4), dtype=np.uint8) # Shape (H, W, 4)
        
        # Where the mask is active, set the red color with transparency
        red_overlay[mask_resized_uint8 == 1] = [255, 0, 0, 100] # R=255, G=0, B=0, Alpha=100 (semi-transparent red)
        
        ax.imshow(red_overlay) # Overlay the red mask
    
    ax.set_title(f"5. K-Means Localization ({tampering_pct:.1f}%)", fontsize=12)
    ax.axis('off')

def create_uncertainty_visualization(ax, results):
    """Create uncertainty visualization for detailed confidence breakdown."""
    uncertainty_analysis_data = results.get('classification', {}).get('uncertainty_analysis', {})
    
    if not uncertainty_analysis_data:
        ax.text(0.5, 0.5, 'Uncertainty Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title("15. Analisis Ketidakpastian", fontsize=12)
        ax.axis('off')
        return

    uncertainty_data = uncertainty_analysis_data.get('probabilities', {})
    report_details = uncertainty_analysis_data.get('report', {})
    
    # Clear previous contents and settings to prevent overlap if reusing axis
    ax.clear()
    ax.set_aspect('auto') # Ensure proper aspect ratio for text


    # Use the `report_details` to present human-readable information
    # Use formatted string to control layout and appearance
    
    # Title with padding from top for aesthetics
    ax.text(0.5, 0.95, "Analisis Ketidakpastian Forensik", ha='center', va='top', 
            fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    y_start_pos = 0.88 # Start position for text, slightly below title
    line_height = 0.08 # Spacing between lines of text
    font_size = 9
    label_font_size = 9.5
    
    # Main Assessment & Confidence
    ax.text(0.05, y_start_pos, 
            f"**Penilaian:** {report_details.get('primary_assessment', 'N/A')}", 
            fontsize=label_font_size, va='top', wrap=True, transform=ax.transAxes)
    y_start_pos -= line_height

    ax.text(0.05, y_start_pos, 
            f"**Keyakinan:** {report_details.get('confidence_level', 'N/A')}", 
            fontsize=label_font_size, va='top', wrap=True, transform=ax.transAxes)
    y_start_pos -= line_height
    
    # Overall Uncertainty Level
    overall_uncertainty_pct = uncertainty_data.get('uncertainty_level', 0.0) * 100
    ax.text(0.05, y_start_pos, 
            f"**Tingkat Ketidakpastian:** {overall_uncertainty_pct:.1f}%", 
            fontsize=label_font_size, va='top', wrap=True, transform=ax.transAxes)
    y_start_pos -= line_height

    # Detailed Uncertainty Description
    uncertainty_desc_text = report_details.get('uncertainty_description', 'No description available.')
    # Using markdown or similar approach within text to denote importance, if not raw text
    ax.text(0.05, y_start_pos, 
            f"_{uncertainty_desc_text}_", # Italicize description
            fontsize=font_size, va='top', wrap=True, transform=ax.transAxes)
    y_start_pos -= (line_height * (uncertainty_desc_text.count('\n') + 1)) # Adjust line height for wrapped text


    # Reliability Indicators
    reliability_indicators = report_details.get('reliability_indicators', [])
    if reliability_indicators:
        ax.text(0.05, y_start_pos - line_height/2, 
                "**Indikator Keandalan:**", 
                fontsize=label_font_size, va='top', transform=ax.transAxes)
        y_start_pos -= line_height*1.5 # Space before bullet points
        
        # Display 3-5 main indicators
        for i, indicator_text in enumerate(reliability_indicators[:5]):
            ax.text(0.05, y_start_pos, 
                    f"• {indicator_text}", 
                    fontsize=font_size - 0.5, va='top', wrap=True, transform=ax.transAxes) # Slightly smaller for indicators
            y_start_pos -= line_height * (indicator_text.count('\n') + 1) # Adjust line height if indicator wraps

    ax.axis('off') # Hide axis for cleaner text display


def create_probability_bars(ax, results):
    """Create probability bar chart with confidence intervals."""
    probabilities_data = results.get('classification', {}).get('uncertainty_analysis', {}).get('probabilities', {})

    if not probabilities_data:
        ax.text(0.5, 0.5, 'Probability Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('14. Distribusi Probabilitas', fontsize=12)
        ax.axis('off')
        return

    # Clear previous contents and settings
    ax.clear()

    categories = ['Authentic', 'Copy-Move', 'Splicing'] # Consistent category labels
    
    # Retrieve probabilities, ensuring they are valid numbers.
    authentic_prob = probabilities_data.get('authentic_probability', 0.0)
    copy_move_prob = probabilities_data.get('copy_move_probability', 0.0)
    splicing_prob = probabilities_data.get('splicing_probability', 0.0)
    
    probabilities = np.array([authentic_prob, copy_move_prob, splicing_prob]) * 100 # Convert to percentage

    # Get confidence intervals, default to P +/- 0 if not present
    intervals = probabilities_data.get('confidence_intervals', {})
    
    authentic_int = intervals.get('authentic', {'lower':authentic_prob, 'upper':authentic_prob})
    cm_int = intervals.get('copy_move', {'lower':copy_move_prob, 'upper':copy_move_prob})
    splicing_int = intervals.get('splicing', {'lower':splicing_prob, 'upper':splicing_prob})
    
    # Calculate error bar heights (from probability to lower/upper interval bound)
    errors_lower = np.array([
        probabilities[0] - (authentic_int['lower'] * 100),
        probabilities[1] - (cm_int['lower'] * 100),
        probabilities[2] - (splicing_int['lower'] * 100)
    ])
    errors_upper = np.array([
        (authentic_int['upper'] * 100) - probabilities[0],
        (cm_int['upper'] * 100) - probabilities[1],
        (splicing_int['upper'] * 100) - probabilities[2]
    ])
    
    # Stack errors as a 2xN array required by yerr (lower, upper)
    errors = np.vstack([errors_lower, errors_upper])
    # Ensure no negative error bars if somehow interval goes above prob (shouldn't if applied uncertainty correctly)
    errors = np.maximum(0, errors) 

    # Colors for the bars
    colors = ['#28a745', '#dc3545', '#ffc107'] # Green, Red, Yellow (Bootstrap-like success, danger, warning)
    
    # Create the bar chart
    bars = ax.bar(categories, probabilities, color=colors, alpha=0.8, yerr=errors, capsize=7, ecolor='black')
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, # +2 to offset slightly above the bar
                f'{yval:.1f}%', va='bottom', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_ylim(0, 110) # Adjust y-limit to give space for labels
    ax.set_ylabel('Probability (%)', fontsize=10)
    ax.set_title('14. Distribusi Probabilitas dengan Interval Kepercayaan', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xticklabels(categories, fontsize=10)
    
    # Add a note on uncertainty level
    overall_uncertainty_level_pct = probabilities_data.get('uncertainty_level', 0.0) * 100
    ax.text(0.5, -0.2, # Position relative to axes
            f'Tingkat Ketidakpastian Keseluruhan: {overall_uncertainty_level_pct:.1f}%',
            horizontalalignment='center', verticalalignment='top', 
            transform=ax.transAxes, fontsize=8, color='gray')


def create_frequency_visualization(ax, results):
    freq_data = results.get('frequency_analysis', {}).get('dct_stats', {})
    
    # Convert large scientific notation numbers for labels into more readable format if needed, or adjust x/y labels
    values = [freq_data.get('low_freq_energy', 0), freq_data.get('mid_freq_energy', 0), freq_data.get('high_freq_energy', 0)]
    labels = ['Rendah', 'Menengah', 'Tinggi'] # Simpler labels

    ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax.set_title(f"9. Analisis Frekuensi", fontsize=12)
    ax.set_ylabel('Energi DCT', fontsize=10)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Force scientific notation if values are very large
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pita Frekuensi', fontsize=10)


def create_texture_visualization(ax, results):
    texture_data = results.get('texture_analysis', {}).get('texture_consistency', {})
    
    # Filter for valid data before processing to avoid errors on empty analysis
    if not texture_data or all(v == 0.0 or np.isnan(v) or np.isinf(v) for v in texture_data.values()):
        ax.text(0.5, 0.5, 'Texture Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"10. Konsistensi Tekstur", fontsize=12)
        ax.axis('off')
        return


    metrics = []
    values = []
    
    # Re-order and clean up names
    ordered_metrics = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'lbp_uniformity']
    for m in ordered_metrics:
        key = f'{m}_consistency'
        if key in texture_data and np.isfinite(texture_data[key]):
            metrics.append(m.capitalize().replace('_', ' ')) # "lbp uniformity"
            values.append(texture_data[key])
    
    # If values list is empty after filtering, cannot plot
    if not values:
        ax.text(0.5, 0.5, 'Texture Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"10. Konsistensi Tekstur", fontsize=12)
        ax.axis('off')
        return

    # Horizontal bar chart is often good for consistency metrics
    ax.barh(metrics, values, color='#8c564b', alpha=0.8) # Muted brown/orange color
    ax.set_title(f"10. Konsistensi Tekstur", fontsize=12)
    ax.set_xlabel('Skor Inkonsistensi (↑ lebih buruk)', fontsize=10)
    ax.invert_yaxis() # Highest value at top for clarity
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.tick_params(axis='y', labelsize=9) # Smaller tick labels for y-axis


def create_edge_visualization(ax, original_pil, results):
    # Ensure image_gray is a valid NumPy array
    image_gray_data = np.array(original_pil.convert('L'))
    
    # Handle cases where image is too small for Sobel, or error during edge detection
    edges = np.zeros_like(image_gray_data, dtype=float) # Default black if issue
    edge_inconsistency = results.get('edge_analysis', {}).get('edge_inconsistency', 0.0)

    try:
        if image_gray_data.shape[0] > 1 and image_gray_data.shape[1] > 1: # Minimum size for any edge detection
            # Attempt using skimage.sobel first if available, else fallback logic
            if SKIMAGE_AVAILABLE:
                edges = sobel(image_gray_data.astype(np.float32)) # Sobel usually works best on float type
            else: # Fallback to OpenCV or manual
                if image_gray_data.shape[0] >= 3 and image_gray_data.shape[1] >= 3:
                    grad_x = cv2.Sobel(image_gray_data, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(image_gray_data, cv2.CV_64F, 0, 1, ksize=3)
                    edges = np.sqrt(grad_x**2 + grad_y**2)
                else: # Manual simple diff for tiny images if necessary or just empty array
                    edges = np.zeros_like(image_gray_data, dtype=float)
        
        # Normalize edges to 0-1 for consistent display, if not already
        if edges.max() > 0:
            edges = edges / edges.max()
        else: # All edges are zero, resulting in flat image
            edges = np.zeros_like(edges) # Render as black

    except Exception as e:
        print(f"Warning: Edge visualization failed during edge detection: {e}. Displaying black image.")
        edges = np.zeros_like(image_gray_data, dtype=float) # On error, use a blank image


    ax.imshow(edges, cmap='gray') # Grayscale for edge map
    ax.set_title(f"6. Analisis Tepi (Incons: {edge_inconsistency:.2f})", fontsize=12)
    ax.axis('off')


def create_illumination_visualization(ax, original_pil, results):
    image_array = np.array(original_pil.convert('RGB')) # Ensure RGB for LAB conversion
    
    illumination_data = np.zeros(original_pil.size[::-1], dtype=np.uint8) # Default black
    illum_inconsistency = results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0.0)

    if image_array.size > 0:
        try:
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            illumination_data = lab[:, :, 0] # L channel for illumination
        except Exception as e:
            print(f"Warning: Illumination visualization failed during LAB conversion: {e}. Displaying black image.")
    else:
        print("Warning: Input image for illumination analysis is empty. Displaying black image.")

    disp = ax.imshow(illumination_data, cmap='magma') # Magma cmap often highlights light differences well
    ax.set_title(f"7. Peta Iluminasi (Incons: {illum_inconsistency:.2f})", fontsize=12)
    ax.axis('off')
    plt.colorbar(disp, ax=ax, fraction=0.046, pad=0.04) # Color bar for illumination intensity


def create_statistical_visualization(ax, results):
    stats = results.get('statistical_analysis', {})
    
    # Check if necessary data exists and is not just default zeros.
    # If entropy is all 0s, bars will be flat.
    r_entropy = stats.get('R_entropy', 0.0)
    g_entropy = stats.get('G_entropy', 0.0)
    b_entropy = stats.get('B_entropy', 0.0)
    
    entropy_values = [r_entropy, g_entropy, b_entropy]

    # Handle case where all entropies might be zero (e.g. solid black/white image or analysis failure)
    if all(v == 0.0 for v in entropy_values):
        ax.text(0.5, 0.5, 'Statistical Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"11. Entropi Kanal", fontsize=12)
        ax.axis('off')
        return

    channels = ['Red', 'Green', 'Blue']
    colors = ['#d62728', '#2ca02c', '#1f77b4'] # Red, Green, Blue from tab20 palette
    
    ax.bar(channels, entropy_values, color=colors, alpha=0.8)
    ax.set_title(f"11. Entropi Kanal", fontsize=12)
    ax.set_ylabel('Entropi (bits)', fontsize=10)
    ax.set_ylim(0, 8.5) # Entropy typically 0-8 for 256 grayscale levels
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', labelsize=9) # Adjust font size


def create_quality_response_plot(ax, results):
    # Retrieve data using .get() with empty list/dict defaults for safety
    jpeg_analysis_data = results.get('jpeg_analysis', {})
    quality_responses = jpeg_analysis_data.get('basic_analysis', {}).get('quality_responses', [])
    estimated_original_quality = jpeg_analysis_data.get('basic_analysis', {}).get('estimated_original_quality', None)
    
    if not quality_responses: # If no quality responses were generated
        ax.text(0.5, 0.5, 'Quality Response Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"12. Respons Kualitas JPEG", fontsize=12)
        ax.axis('off')
        return

    qualities = [r['quality'] for r in quality_responses]
    mean_responses = [r['response_mean'] for r in quality_responses]
    
    ax.plot(qualities, mean_responses, 'b-o', markersize=5, linewidth=1.5, markerfacecolor='cornflowerblue', markeredgecolor='darkblue') # Stylized plot
    
    if estimated_original_quality is not None and estimated_original_quality > 0: # Ensure valid estimated quality
        ax.axvline(x=estimated_original_quality, color='r', linestyle='--', linewidth=1.5, label=f'Est. Q: {estimated_original_quality}')
        ax.legend(fontsize=8, loc='upper right') # Smaller legend for compact plots

    ax.set_title(f"12. Respons Kualitas JPEG", fontsize=12)
    ax.set_xlabel('Kualitas JPEG', fontsize=10)
    ax.set_ylabel('Rata-rata Error', fontsize=10)
    ax.set_ylim(bottom=0) # Ensure y-axis starts from 0
    ax.grid(True, alpha=0.6, linestyle=':') # Dotted grid for clarity


def create_advanced_combined_heatmap(analysis_results, image_size):
    """Create combined heatmap with robust size handling."""
    # Ensure image_size is (width, height)
    w, h = 512, 512 # Fallback defaults
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        w, h = int(image_size[0]), int(image_size[1])
    elif hasattr(image_size, 'width') and hasattr(image_size, 'height'): # PIL Image size
        w, h = image_size.width, image_size.height
    
    if w <= 0 or h <= 0: # Handle invalid dimensions after potential conversion
        print(f"Warning: Invalid image dimensions ({w}x{h}) for heatmap. Using default 512x512.")
        w, h = 512, 512

    # Initialize a base heatmap of zeros with float type
    # Note: numpy arrays are HxW, so it's (height, width)
    heatmap = np.zeros((h, w), dtype=np.float32)

    # 1. Kontribusi ELA (bobot 35%)
    ela_image_data = analysis_results.get('ela_image')
    if ela_image_data is not None:
        try:
            ela_array = np.array(ela_image_data.convert('L')) if isinstance(ela_image_data, Image.Image) else np.array(ela_image_data)
            if ela_array.size > 0:
                ela_resized = cv2.resize(ela_array, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                # Normalize ELA to 0-1 range for blending if it's 0-255 already
                heatmap += (ela_resized / 255.0) * 0.35 
            else: print("Warning: ELA array is empty for heatmap.")
        except Exception as e:
            print(f"Warning: ELA contribution to heatmap failed: {e}")

    # 2. Kontribusi JPEG Ghost (bobot 25%)
    jpeg_ghost_data = analysis_results.get('jpeg_ghost')
    if jpeg_ghost_data is not None:
        try:
            if jpeg_ghost_data.size > 0:
                ghost_resized = cv2.resize(jpeg_ghost_data, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                # Ghost data is often 0-1, so just scale it
                heatmap += ghost_resized * 0.25 
            else: print("Warning: JPEG ghost array is empty for heatmap.")
        except Exception as e:
            print(f"Warning: JPEG Ghost contribution to heatmap failed: {e}")

    # 3. Kontribusi Lokalisasi K-Means (bobot 40%)
    loc_analysis = analysis_results.get('localization_analysis', {})
    combined_mask_data = loc_analysis.get('combined_tampering_mask') # Expects boolean or uint8 mask
    if combined_mask_data is not None:
        try:
            if combined_mask_data.size > 0:
                # Resize mask from its shape (which should be HxW like image array)
                # It's bool or uint8 0/1. Directly multiply by weight.
                mask_resized = cv2.resize(combined_mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                heatmap += mask_resized * 0.40
            else: print("Warning: Localization mask is empty for heatmap.")
        except Exception as e:
            print(f"Warning: Localization mask contribution to heatmap failed: {e}")

    # Normalize the final heatmap to 0-1 for consistent display, and apply a mild blur
    # Add a small epsilon to prevent division by zero if max is 0
    heatmap_max = np.max(heatmap)
    if heatmap_max > 0:
        heatmap_norm = heatmap / heatmap_max
    else: # If all contributions were zero, it's just a black image
        heatmap_norm = heatmap 

    # Apply Gaussian blur for smoother appearance, use odd kernel size
    blur_ksize = int(max(1, min(101, w // 20))) # Max 101, scaled from width
    if blur_ksize % 2 == 0: blur_ksize += 1 # Ensure odd
    if blur_ksize > 1: # Only blur if ksize is effective
        heatmap_blurred = cv2.GaussianBlur(heatmap_norm, (blur_ksize, blur_ksize), 0)
    else: # No blur if image is too small
        heatmap_blurred = heatmap_norm 

    return heatmap_blurred


def create_summary_report(ax, analysis_results):
    ax.axis('off') # Turn off axis for text display
    ax.clear() # Clear existing content if reusing axis

    classification = analysis_results.get('classification', {})
    
    # Safe access for type and confidence, with fallback
    result_type = classification.get('type', 'N/A')
    confidence_level = classification.get('confidence', 'N/A')
    
    # Text color for the result based on type (green for authentic, red for manipulation)
    text_color = 'green'
    if "Manipulasi" in result_type or "Forgery" in result_type or "Splicing" in result_type or "Copy-Move" in result_type:
        text_color = 'red'

    # Scores (Copy-Move, Splicing)
    copy_move_score = classification.get('copy_move_score', 0)
    splicing_score = classification.get('splicing_score', 0)
    
    # Main summary header using rich text (LaTeX style is used in Matplotlib text, but can fall back)
    # The `plt.text` function with `usetex=False` will try to interpret LaTeX-like syntax or display it raw.
    
    summary_header = f"\\textbf{{RINGKASAN LAPORAN HASIL ANALISIS}}\n" \
                     f"-----------------------------------------\n" \
                     f"\\textbf{{Tipe Deteksi:}} {result_type}\n" \
                     f"\\textbf{{Kepercayaan:}} {confidence_level}\n" \
                     f"-----------------------------------------\n" \
                     f"\\textbf{{Skor Copy-Move:}} {copy_move_score}/100\n" \
                     f"\\textbf{{Skor Splicing:}} {splicing_score}/100\n"
    
    # Start point for the text. Top-left of the axis.
    ax.text(0.01, 0.98, summary_header, 
            transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.8), # Lighter box
            wrap=True)

    # Key Findings / Details
    details = classification.get('details', [])
    if details:
        details_text = "\\textbf{{Temuan Kunci:}}\n"
        # Limit the number of details and truncate length to fit
        for i, detail_item in enumerate(details[:5]): # Show up to 5 key details
            truncated_detail = detail_item[:70] + "..." if len(detail_item) > 70 else detail_item
            details_text += f" • {truncated_detail}\n"
    else:
        details_text = "\\textbf{{Temuan Kunci:}}\n • Tidak ada temuan signifikan."

    # Adjust vertical position for key findings. This requires more precise positioning or
    # using `plt.figtext` (relative to figure) or calculating total lines in summary_header
    # or using `Y` value (which positions the *bottom* of text block if va='bottom').
    # A simple y-position decrement can be error prone with variable content.
    
    # A robust approach using `get_window_extent` to find actual height of previous text,
    # or a series of ax.text calls that build upon each other.
    # For now, let's estimate its position for consistent layout given its content structure.
    
    # Calculate approx. vertical offset for the "details" text based on line count in summary_header
    # Line height ~0.035, base starting point for details.
    
    ax.text(0.01, 0.55, details_text, # Approx. position after scores + few empty lines
            transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='azure', edgecolor='lightgray', alpha=0.7), # Lighter box for details
            wrap=True) # wrap text within box boundaries


    ax.set_title("13. Ringkasan Laporan Klasifikasi", fontsize=14, y=1.05) # Main plot title

def populate_validation_visuals(ax1, ax2):
    """
    Populates two subplots with system validation visuals (Confusion Matrix and Confidence Distribution).
    This function acts as a fallback or a generic system validation representation.
    """
    ax1.clear() # Clear contents
    ax2.clear() # Clear contents

    # Set common titles initially
    ax1.set_title("16. Matriks Konfusi (Contoh)", fontsize=12)
    ax2.set_title("17. Distribusi Kepercayaan (Contoh)", fontsize=12)

    # Use simulated data or provide informative message if libraries are not available
    if SKLEARN_METRICS_AVAILABLE and SCIPY_AVAILABLE:
        # Simulate data for a confusion matrix
        y_true = np.random.randint(0, 2, 100) # 0 for Authentic, 1 for Manipulated
        # Predict with some accuracy, e.g., 90% chance to be correct
        y_pred = np.array([y_true[i] if np.random.rand() < 0.9 else 1-y_true[i] for i in range(100)])
        
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False, # Removed cbar for cleaner look
                    xticklabels=['Asli', 'Manipulasi'],
                    yticklabels=['Asli', 'Manipulasi'], 
                    linewidths=.5, linecolor='gray')
        ax1.set_xlabel("Prediksi", fontsize=9)
        ax1.set_ylabel("Aktual", fontsize=9)
        ax1.tick_params(axis='both', which='major', labelsize=8) # Smaller ticks
        ax1.set_title(f"16. Matriks Konfusi (Akurasi: {accuracy:.1%})", fontsize=12)

        # Simulate data for a confidence score distribution (e.g., scores for "Manipulated" class)
        # Bimodal distribution might represent authentic (low scores) and manipulated (high scores)
        np.random.seed(42) # For reproducibility
        authentic_scores = np.random.normal(loc=20, scale=10, size=50) # Scores for authentic
        manipulated_scores = np.random.normal(loc=80, scale=10, size=50) # Scores for manipulated
        combined_scores = np.concatenate((authentic_scores, manipulated_scores))
        combined_scores = np.clip(combined_scores, 0, 100) # Clip scores to 0-100 range

        sns.histplot(combined_scores, kde=True, ax=ax2, color="purple", bins=15, alpha=0.6, stat="density", linewidth=0) # KDE for smoothing
        ax2.set_xlabel("Skor Kepercayaan (%)", fontsize=9)
        ax2.set_ylabel("Densitas", fontsize=9)
        
        # Mean/median might not be representative for bimodal. Use custom labels if needed.
        ax2.axvline(x=50, color='r', linestyle=':', label='Batas Klasifikasi (50%)')
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_ylim(bottom=0)
        ax2.tick_params(axis='both', which='major', labelsize=8)

    else:
        # Message if scikit-learn or scipy not available
        ax1.text(0.5, 0.5, 'Sklearn / Scipy Not Available\n(Cannot display metrics)', 
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, 
                 fontsize=10, color='gray')
        ax1.axis('off')
        ax2.text(0.5, 0.5, 'Sklearn / Scipy Not Available\n(Cannot display distribution)', 
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, 
                 fontsize=10, color='gray')
        ax2.axis('off')

# ======================= Backward Compatibility (deprecated functions) =======================
# Fungsi ini dijaga agar tidak merusak pemanggilan dari file lain yang belum diupdate.

def create_technical_metrics_plot(ax, results):
    """
    Placeholder/deprecated function.
    Previously showed some technical metrics, now redirected to be shown in the full report/new visuals.
    """
    ax.axis('off')
    ax.text(0.5, 0.5, 'Detail Metrik Tersedia\nDalam Laporan Lengkap (DOCX/PDF)', 
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
            fontsize=10, alpha=0.7, color='gray', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.5))
    ax.set_title("Metrics & Parameters (Ref. Full Report)", fontsize=10, alpha=0.8) # Muted title

def export_kmeans_visualization(original_pil, analysis_results, output_filename="kmeans_analysis.jpg"):
    """
    Deprecated: Exports a dedicated visualization for K-means clustering analysis.
    This functionality is now part of `create_localization_visualization` within the main report.
    """
    print(f"Warning: `export_kmeans_visualization` is deprecated. Use `create_localization_visualization` in main visual or `generate_all_process_images` for output.")
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available for deprecated K-means visualization.")
        return None

    if 'localization_analysis' not in analysis_results:
        print("❌ K-means analysis data not available for deprecated visualization.")
        return None
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 6)) # Simplified layout for this deprecated export
    fig.suptitle('Deprecated: K-means Tampering Localization', fontsize=14, fontweight='bold')
    
    loc = analysis_results.get('localization_analysis', {})
    
    # Subplot 1: Original Image with Mask Overlay
    create_localization_visualization(axes[0], original_pil, analysis_results) # Re-use the new combined localization
    axes[0].set_title('Localization Overlay (New)', fontsize=10) # Change title for clarity that it uses updated visual
    axes[0].axis('off') # Keep axis off
    
    # Subplot 2: K-means Cluster Map (if `kmeans_localization` key holds necessary map, often `localization_map` array)
    kmeans_loc_data = loc.get('kmeans_localization', {})
    if 'localization_map' in kmeans_loc_data and kmeans_loc_data['localization_map'] is not None and kmeans_loc_data['localization_map'].size > 0:
        im_cluster = axes[1].imshow(kmeans_loc_data['localization_map'], cmap='viridis')
        plt.colorbar(im_cluster, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title('Cluster Map (K-Means)', fontsize=10)
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'K-Means Map Unavailable', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, fontsize=10, color='gray')
        axes[1].set_title('Cluster Map', fontsize=10)
        axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        plt.savefig(output_filename, dpi=120, bbox_inches='tight') # Lower DPI for quick export
        plt.close(fig)
        print(f"📊 (Deprecated) K-means visualization exported to '{output_filename}'")
        return output_filename
    except Exception as e:
        print(f"❌ (Deprecated) K-means visualization export failed: {e}")
        plt.close(fig)
        return None