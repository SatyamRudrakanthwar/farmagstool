import cv2
import numpy as np
import os
from PIL import Image # Added for the wrapper

# --- CONFIGURATION PARAMETERS ---
CONFIG = {
    "RESIZE_DIM": (600, 600),
    # --- General ---
    "GRABCUT_ITERATIONS": 5,
    "FEATHER_AMOUNT": 15,
    # --- Segmentation Strategy Logic ---
    "SEGMENTATION_SAT_THRESHOLD": 85, # Saturation value to decide which segmentation method to use
    # --- Healthy Leaf Segmentation ---
    "HEALTHY_MIN_CONTOUR_AREA_RATIO": 0.015,
    "HEALTHY_SOLIDITY_THRESHOLD": 0.5,
    # --- Pale/Diseased Leaf Segmentation ---
    "PALE_HSV_LOWER": np.array([15, 20, 30]),
    "PALE_HSV_UPPER": np.array([95, 255, 255]),
    # --- Sky Exclusion CONFIG ---
    "SKY_EXCLUSION_THRESHOLD": 0.3, # Percentage of image height from top to check for sky
    "SKY_HSV_LOWER": np.array([90, 50, 50]), # Typical blue range for sky
    "SKY_HSV_UPPER": np.array([130, 255, 255]),
}

#
# --- (Your provided functions: segment_healthy_leaf, segment_pale_leaf, ---
# ---  exclude_sky, _fill_holes, remove_background_hybrid) ---
# --- (Pasting them here without any changes) ---
#

def segment_healthy_leaf(img):
    h, w = img.shape[:2]
    img_16bit = img.astype(np.int16)
    b, g, r = cv2.split(img_16bit)
    g_minus_r = cv2.subtract(g, r)
    g_minus_b = cv2.subtract(g, b)
    green_dominant_img = cv2.add(g_minus_r, g_minus_b)
    green_dominant_img = cv2.normalize(green_dominant_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, mask = cv2.threshold(green_dominant_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, "No green objects found"
    
    min_area = h * w * CONFIG["HEALTHY_MIN_CONTOUR_AREA_RATIO"]
    candidate_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area: continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity > CONFIG["HEALTHY_SOLIDITY_THRESHOLD"]:
            candidate_contours.append(contour)
    
    if not candidate_contours:
        main_contour = max(contours, key=cv2.contourArea)
    else:
        main_contour = max(candidate_contours, key=cv2.contourArea)
        
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [main_contour], -1, 255, -1)
    return final_mask, "Healthy Leaf Method"


def segment_pale_leaf(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    _, lab_mask = cv2.threshold(a_channel, 125, 255, cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, CONFIG["PALE_HSV_LOWER"], CONFIG["PALE_HSV_UPPER"])
    combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No pale leaf objects found"
        
    main_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [main_contour], -1, 255, -1)
    return final_mask, "Pale Leaf Method"


def exclude_sky(img, current_mask):
    h, w = img.shape[:2]
    
    sky_roi_height = int(h * CONFIG["SKY_EXCLUSION_THRESHOLD"])
    sky_roi = img[0:sky_roi_height, 0:w]
    
    hsv_sky_roi = cv2.cvtColor(sky_roi, cv2.COLOR_BGR2HSV)
    
    sky_color_mask_roi = cv2.inRange(hsv_sky_roi, CONFIG["SKY_HSV_LOWER"], CONFIG["SKY_HSV_UPPER"])
    
    full_sky_mask = np.zeros((h, w), dtype=np.uint8)
    full_sky_mask[0:sky_roi_height, 0:w] = sky_color_mask_roi
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    full_sky_mask = cv2.morphologyEx(full_sky_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    modified_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(full_sky_mask))
    
    return modified_mask


def _fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask 
    main_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, [main_contour], -1, 255, cv2.FILLED)
    return filled_mask


def remove_background_hybrid(img):
    orig_h, orig_w = img.shape[:2]
    small = cv2.resize(img, CONFIG["RESIZE_DIM"])
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([100, 255, 255]))
    
    avg_saturation = 0
    if np.any(green_mask):
        avg_saturation = cv2.mean(hsv, mask=green_mask)[1]
    
    initial_mask = None
    method_used = ""

    if avg_saturation < CONFIG["SEGMENTATION_SAT_THRESHOLD"]:
        initial_mask, method_used = segment_pale_leaf(small)
    else:
        initial_mask, method_used = segment_healthy_leaf(small)
        
    if initial_mask is None:
        return None, "Segmentation Failed", 100.0, "Unknown"

    initial_mask = exclude_sky(small, initial_mask)
    method_used += " + Sky Exclusion"

    try:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (0, 0, CONFIG["RESIZE_DIM"][0] - 1, CONFIG["RESIZE_DIM"][1] - 1)
        mask_gc = np.where(initial_mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype("uint8")
        
        cv2.grabCut(small, mask_gc, rect, bgdModel, fgdModel, CONFIG["GRABCUT_ITERATIONS"], cv2.GC_INIT_WITH_MASK)
        
        grabcut_mask = np.where((mask_gc == cv2.GC_PR_FGD) | (mask_gc == cv2.GC_FGD), 255, 0).astype("uint8")
        
        if np.count_nonzero(grabcut_mask) > np.count_nonzero(initial_mask) * 0.5:
            final_mask = grabcut_mask
            method_used += " + GrabCut"
        else:
            final_mask = initial_mask 
            
    except Exception:
        final_mask = initial_mask 
        pass 

    final_mask = _fill_holes(final_mask)
    if "GrabCut" in method_used: 
        method_used += " + Hole Fill"

    mask_orig = cv2.resize(final_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    blur_amount = CONFIG["FEATHER_AMOUNT"]
    if blur_amount > 0:
        if blur_amount % 2 == 0: blur_amount += 1
        mask_orig = cv2.GaussianBlur(mask_orig, (blur_amount, blur_amount), 0)
        
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask_orig])
    erased_pixels = np.sum(mask_orig == 0)
    data_erased_pct = 100 * erased_pixels / (orig_h * orig_w)
    result = "Success" if data_erased_pct < 99 else "Segmentation Failed"
    
    return rgba, result, data_erased_pct, method_used


# -----------------------------------------------------------------
# --- NEW WRAPPER FUNCTION ---
# -----------------------------------------------------------------
def remove_bg_from_pil_and_get_bgr(pil_image: Image.Image) -> np.ndarray:
    """
    Wrapper function to integrate with the Streamlit app.
    Takes a PIL (RGB) image, runs BG removal, and returns a 3-channel
    BGR image (on a white background) ready for color analysis.
    """
    # 1. Convert PIL (RGB) to BGR for processing
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    try:
        # 2. Run your hybrid background removal
        rgba, result, _, _ = remove_background_hybrid(bgr_image)

        if result != "Success" or rgba is None:
            # If BG removal fails, return the original BGR image
            return bgr_image

        # 3. Convert 4-channel RGBA to 3-channel BGR
        # Create a white background
        bg = np.full((rgba.shape[0], rgba.shape[1], 3), 255, dtype=np.uint8)
        
        # Get the alpha mask and convert to 3 channels
        alpha = rgba[:, :, 3] / 255.0
        alpha_mask = cv2.merge([alpha, alpha, alpha])
        
        # Get the RGB channels
        rgb = rgba[:, :, :3]
        
        # Alpha-blend the foreground (rgb) onto the background (bg)
        blended_bgr = (rgb * alpha_mask + bg * (1 - alpha_mask)).astype(np.uint8)
        
        return blended_bgr

    except Exception as e:
        print(f"Warning: BG removal failed with error: {e}. Returning original image.")
        # If any other error occurs, return the original
        return bgr_image