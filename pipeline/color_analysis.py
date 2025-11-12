import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import io
import os
import pandas as pd
import colorsys
from collections import defaultdict

# -----------------------------------------------------------------
# --- NEW: Bar Graph Plotting Function ---
# -----------------------------------------------------------------
def plot_color_bar_graph(colors: np.ndarray, counts: np.ndarray, title: str) -> np.ndarray:
    """
    Creates a bar graph from color clusters and returns it as an image.
    """
    # --- Create the Matplotlib Bar Plot ---
    fig, ax = plt.subplots(figsize=(6, 4)) # 6-inch width, 4-inch height
    
    # Convert RGB colors (0-255) to Matplotlib format (0-1)
    bar_colors = colors / 255.0
    
    # Create labels for the x-axis (e.g., "#RRGGBB")
    labels = [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors]
    
    ax.bar(labels, counts, color=bar_colors, edgecolor='black')
    
    ax.set_ylabel("Relative Pixel Count")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    # --- Convert Matplotlib plot to a NumPy array ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig) # Close the plot to save memory
    
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for Streamlit
    
    return img_rgb

# -----------------------------------------------------------------
# --- Core Image Processing Functions (Refactored) ---
# -----------------------------------------------------------------

def leaf_vein_skeleton(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Input image is None")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    edges = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    _, binary = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize((binary // 255).astype(bool)).astype(np.uint8) * 255
    return skeleton

def leaf_boundary_dilation(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Input image is None")
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    _, binary = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    boundary = cv2.subtract(dilated, binary)
    return boundary

def extract_colors_around_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    buffer_ratio: float = 0.15,
    num_colors: int = 8,
    color_type: str = "general"
) -> List[Tuple[int, int, int]]:
    """
    Extracts colors around a mask, runs KMeans to find dominant clusters,
    and returns a list of colors (RGB tuples) proportional to their cluster size.
    
    Args:
        image_bgr: The image in BGR format.
        mask: The single-channel mask where foreground pixels are > 0.
        buffer_ratio: Ratio for dilation buffer around the mask.
        num_colors: Number of clusters (dominant colors) to find.
        color_type: Not used internally, but helpful for debugging.

    Returns:
        List[Tuple[int, int, int]]: A list of RGB color tuples, one for each pixel
        in the cluster, effectively weighting the color by its count.
    """
    if image_bgr is None:
        raise ValueError("Input image is None")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255

    h, w = mask.shape[:2]
    diag = int(np.sqrt(h ** 2 + w ** 2))
    buffer_pixels = max(2, int(diag * buffer_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_pixels, buffer_pixels))
    region = cv2.dilate(mask, kernel, iterations=1)

    masked_pixels = image_rgb[region == 255].reshape(-1, 3)
    if masked_pixels.size == 0:
        return []

    # Filter out near black/white pixels (threshold 20 instead of 5/250 for less aggressive cleaning)
    valid = ~(((masked_pixels <= 20).all(axis=1)) | ((masked_pixels >= 235).all(axis=1)))
    filtered_pixels = masked_pixels[valid]
    if filtered_pixels.shape[0] == 0:
        return []

    k = min(num_colors, filtered_pixels.shape[0])
    if k < 1:
        return []

    colors_array = filtered_pixels.astype(np.float32)
    k_unique = min(k, len(np.unique(colors_array, axis=0)))
    if k_unique < 1: return []

    kmeans = KMeans(n_clusters=k_unique, random_state=42, n_init="auto")
    klabels = kmeans.fit_predict(colors_array)
    
    # --- NEW RETURN STRUCTURE: List of weighted RGB tuples ---
    weighted_colors = []
    
    for i in range(kmeans.n_clusters):
        idx = np.where(klabels == i)[0]
        if idx.size == 0:
            continue

        cluster_colors = filtered_pixels[idx]
        mean_color = np.mean(cluster_colors, axis=0).astype(int)
        r, g, b = map(int, mean_color.tolist())
        
        # Add the mean color (R, G, B) to the list, repeated by its count
        # This makes the aggregated clustering later more accurate
        weighted_colors.extend([tuple(mean_color)] * idx.size)
            
    return weighted_colors

# -----------------------------------------------------------------
# --- NEW HELPER: Gets colors from a single BGR image ---
# -----------------------------------------------------------------
def _get_colors_from_image(image_bgr: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
    """Extracts vein and boundary colors from a single BGR image."""
    all_vein_colors = []
    all_boundary_colors = []
    try:
        # 1. Get Vein Mask and Colors
        vein_mask = leaf_vein_skeleton(image_bgr)
        # extract_colors_around_mask now returns a list of weighted RGB tuples
        all_vein_colors = extract_colors_around_mask(
            image_bgr, vein_mask, buffer_ratio=0.1, num_colors=4, color_type="vein"
        )
        
        # 2. Get Boundary Mask and Colors
        boundary_mask = leaf_boundary_dilation(image_bgr)
        # extract_colors_around_mask now returns a list of weighted RGB tuples
        all_boundary_colors = extract_colors_around_mask(
            image_bgr, boundary_mask, buffer_ratio=0.1, num_colors=4, color_type="boundary"
        )
        
    except Exception as e:
        print(f"Warning: Color analysis failed for one image: {e}")
        # Return empty lists if this image fails
        return [], []
        
    return all_vein_colors, all_boundary_colors

# -----------------------------------------------------------------
# --- NEW FUNCTION: Generates a bar graph for ONE image ---
# -----------------------------------------------------------------
def generate_individual_color_graph(image_bgr: np.ndarray, num_clusters: int = 5) -> np.ndarray:
    """
    Runs color analysis on a single BGR image and returns its color bar graph.
    """
    all_vein_colors, all_boundary_colors = _get_colors_from_image(image_bgr)
    
    all_colors = all_vein_colors + all_boundary_colors
    
    if len(all_colors) < num_clusters:
        # Failsafe for blank images or errors
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No color data found", ha='center', va='center', color='gray')
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)
        img_rgb = cv2.cvtColor(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img_rgb

    # Run KMeans
    colors_array = np.array(all_colors, dtype=np.float32)
    k = min(num_clusters, len(np.unique(colors_array, axis=0)))
    if k < 1: k = 1

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(colors_array)
    
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    centers = centers[sorted_indices]
    counts = counts[sorted_indices]

    # Generate the bar graph image
    palette_image_rgb = plot_color_bar_graph(centers, counts, "Dominant Colors (This Image)")
    
    return palette_image_rgb

# -----------------------------------------------------------------
# --- UPDATED BATCH FUNCTION ---
# -----------------------------------------------------------------
def run_batch_color_analysis(image_group: List[Tuple[str, np.ndarray]], num_clusters: int = 8) -> np.ndarray:
    """
    Runs color analysis on a BATCH of BGR images and returns one AGGREGATED bar graph.
    
    Args:
        image_group: A list of tuples, e.g., [('fname1', bgr_array_1), ...]
    """
    all_vein_colors = []
    all_boundary_colors = []

    for fname, image_bgr in image_group:
        # Use the helper to get colors for each image
        veins, boundaries = _get_colors_from_image(image_bgr)
        all_vein_colors.extend(veins)
        all_boundary_colors.extend(boundaries)

    # 3. Combine all colors and run final clustering
    all_colors = all_vein_colors + all_boundary_colors
    
    if len(all_colors) < num_clusters:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough color data to plot", ha='center', va='center', color='gray')
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)
        img_rgb = cv2.cvtColor(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img_rgb

    colors_array = np.array(all_colors, dtype=np.float32)
    k = min(num_clusters, len(np.unique(colors_array, axis=0)))
    if k < 1: k = 1 

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(colors_array)
    
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    
    sorted_indices = np.argsort(counts)[::-1]
    centers = centers[sorted_indices]
    counts = counts[sorted_indices]

    # 4. Generate the bar graph image
    palette_image_rgb = plot_color_bar_graph(centers, counts, "Aggregated Dominant Colors (All Images)")
    
    return palette_image_rgb
