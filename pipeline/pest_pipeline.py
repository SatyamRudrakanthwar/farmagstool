import streamlit as st
from collections import defaultdict
import torch
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
from PIL import Image
import cv2
import supervision as sv
import numpy as np
from collections import defaultdict

# PEST DETECTION (BRANCH 1 - PART 1)-

@st.cache_resource
def load_pest_model():
    """
    Loads the YOLOv5 pest model from disk. 
    Caches it in Streamlit's resource cache.
    """
    try:
        model = YOLO("models/best.pt") # Assumes 'models/best.pt'
        return model
    except Exception as e:
        st.error(f"Error loading pest model: {e}")
        return None

# --- NEW HELPER FUNCTION 1 ---
def get_pests_from_results(results):
    """
    Extracts a list of detected pest class names from YOLO results.
    """
    # Get the class name map (e.g., {0: 'aphid', 1: 'thrip'})
    class_names = results[0].names 
    
    # Get the detected class indices as a numpy array
    detected_classes = results[0].boxes.cls.cpu().numpy() 
    
    pests_found = []
    for cls_id in detected_classes:
        pests_found.append(class_names[int(cls_id)])
    
    # Returns a list like ['aphid', 'aphid', 'thrip']
    return pests_found 

# --- NEW HELPER FUNCTION 2 ---
def draw_boxes(pil_image, results):
    """
    Draws bounding boxes on an image using Supervision.
    """
    # 1. Convert PIL (RGB) to CV2 (BGR) format
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 2. Convert YOLO results to a standard Supervision Detections object
    detections = sv.Detections.from_ultralytics(results[0])
    
    # 3. Create annotators
    # 3. Create annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_CENTER,
        text_scale=0.5,
        text_thickness=1,
    )

    # 4. Create labels for the detections
    # 4. Create labels for the detections
    class_names = results[0].names
    labels = [
        f"{class_names[class_id]} {confidence:0.2f}"
        # (xyxy, mask, confidence, class_id, tracker_id, data)
        for _, _, confidence, class_id, _, _ in detections # <-- THIS IS THE FIX
    ]
    
    # 5. Annotate the image
    annotated_frame = box_annotator.annotate(
        scene=cv2_image.copy(), 
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections, 
        labels=labels
    )
    
    # 6. Return the annotated image (in CV2 format)
    return annotated_frame


def run_pest_detection_batch(image_batch_with_names, pest_model):
    """
    Runs pest detection and returns counts AND a dictionary of images grouped by pest.
    """
    total_pest_counts = {}       # e.g., {'aphid': 5, 'thrip': 2}
    images_by_pest = defaultdict(list) # e.g., {'aphid': [('img1', img1_annotated), ...]}
    
    # Cache to store annotated images (so we only draw boxes on each image once)
    annotated_image_cache = {}

    for filename, pil_image in image_batch_with_names:
        
        # --- THIS IS THE FIX ---
        # Create a 3-channel RGB image on a white background if it's RGBA
        if pil_image.mode == "RGBA":
            # Create a new white background image
            white_bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            # Paste the RGBA image onto the white background, using the alpha channel
            white_bg.paste(pil_image, (0, 0), pil_image)
            pil_rgb = white_bg
        else:
            # It's already RGB (or some other mode), just convert
            pil_rgb = pil_image.convert("RGB")
        # --- END OF FIX ---

        # 'pil_rgb' is now guaranteed to be a 3-channel RGB image
        # (composited on white if it was transparent)
        
        # 1. Run detection on the 3-channel image
        results = pest_model(pil_rgb) 
        
        # This helper should return a list of all pests found, e.g., ['aphid', 'aphid', 'thrip']
        pests_found_in_image = get_pests_from_results(results) 

        if not pests_found_in_image:
            continue # Skip this image, no pests found

        # 2. Update total counts (counts all instances)
        for pest in pests_found_in_image:
            total_pest_counts[pest] = total_pest_counts.get(pest, 0) + 1
        
        # 3. Get the single annotated image
        # Check cache first
        if filename in annotated_image_cache:
            annotated_cv2_img = annotated_image_cache[filename]
        else:
            # If not in cache, create, store, and use
            # Pass the 3-channel (on-white) image to be drawn on
            annotated_cv2_img = draw_boxes(pil_rgb, results)
            annotated_image_cache[filename] = annotated_cv2_img
        
        # 4. Add this one image to the list for each *unique* pest found
        for pest_name in set(pests_found_in_image):
            images_by_pest[pest_name].append((filename, annotated_cv2_img))
    
    # Return the counts and the new dictionary
    return total_pest_counts, images_by_pest

# ETL CALCULATION (BRANCH 1 - PART 2)

def calculate_value_loss(I, market_cost_per_kg):
    """Helper function for ETL calculation."""
    yield_lost = I * 1000
    return yield_lost, market_cost_per_kg * yield_lost

def predict_etl_days(data):
    """
    Core ETL prediction engine.
    (This is your original function, unchanged)
    """
    etl_days_list = []
    full_progress_data = []

    for row in data:
        # This function expects a very specific row structure
        pest_name, N_current, I_old, _, C, market_cost_per_kg, _, _, fev_con = row
        days = 0
        while days <= 28:
            if 19 <= fev_con <= 21:
                use = 1.2
            elif 22 <= fev_con <= 40:
                use = 1.4
            elif 14.25 <= fev_con <= 15.75:
                use = 0.8
            else:
                use = 1.0

            N_new = N_current * use
            I_new = (I_old / N_current) * N_new if N_current != 0 else 0
            yield_lost_new = I_new * 1000
            value_loss_new = yield_lost_new * market_cost_per_kg
            result = value_loss_new / C if C != 0 else 0

            full_progress_data.append([
                pest_name, days, round(N_current, 3), round(I_old, 3),
                round(yield_lost_new, 3), round(value_loss_new, 3), round(result, 3)
            ])

            if result > 0.85:
                etl_days_list.append({"Pest Name": pest_name, "Days to ETL": days})
                break

            days += 7
            N_current = N_new
            I_old = I_new

    for item in etl_days_list:
        pest = item["Pest Name"]
        days = item.get("Days to ETL")
        if days is not None:
            delta = max(1, int(round(days * 0.1)))
            item["ETL Range (Days)"] = f"{max(0, days - delta)} – {days + delta} days"
        else:
            item["ETL Range (Days)"] = "Not reached within 28 days"

    df_etl = pd.DataFrame(etl_days_list)
    df_progress = pd.DataFrame(full_progress_data, columns=[
        "Pest Name", "Day", "Pest Count (N)", "Damage Index (I)",
        "Yield Loss (kg)", "Value Loss", "Value Loss / Cost"
    ])
    
    if not df_progress.empty:
        df_progress["Pest Severity (%)"] = (df_progress["Value Loss / Cost"] * 100).round(2)
        df_progress.drop("Value Loss / Cost", axis=1, inplace=True)

    return df_etl, df_progress

def run_etl_calculation(etl_input_data: list):
    """
    Runs the full ETL workflow.
    Instead of printing to Streamlit, it returns the objects for app.py to display.
    
    Args:
        etl_input_data (list): A list of tuples, where each tuple contains
                               the required data for predict_etl_days.
                               e.g., [(pest, N, I, ...), (pest2, N2, I2, ...)]
    """
    
    # 1. Get the raw dataframes from the prediction engine
    df_etl, df_progress = predict_etl_days(etl_input_data)

    fig = None
    if not df_progress.empty:
        # 2. Create the Plotly figure
        fig = px.line(
            df_progress,
            x="Day",
            y="Pest Severity (%)",
            color="Pest Name",
            markers=True,
            line_shape="spline",
            labels={
                "Day": "Days",
                "Pest Severity (%)": "Pest Severity (%)",
                "Pest Name": "Pest"
            }
        )
        fig.update_layout(template="plotly_white", hovermode="x unified", height=500)

    # 3. Return the three objects for app.py to display
    return df_etl, df_progress, fig
def run_pest_pipeline_by_crop(crop_groups, pest_model):
    """
    Runs pest detection for each crop group.
    
    Args:
        crop_groups (dict): A dictionary from run_crop_classification, e.g.,
                            {'Crop 1': [('img1.jpg', img1_pil), ...]}
        pest_model: The loaded YOLO pest model.

    Returns:
        A tuple: (pest_results_by_crop, total_pest_counts_all_crops)
    """
    try:
        pest_results_by_crop = {}
        total_pest_counts_all_crops = defaultdict(int)

        for crop_name, image_group in crop_groups.items():
            if not image_group:
                pest_results_by_crop[crop_name] = {
                    "pest_counts": {},
                    "images_by_pest": {}
                }
                continue
            
            # Run your existing batch function on this smaller image group
            crop_pest_counts, crop_images_by_pest = run_pest_detection_batch(image_group, pest_model)
            
            # Store results for this crop
            pest_results_by_crop[crop_name] = {
                "pest_counts": crop_pest_counts,
                "images_by_pest": crop_images_by_pest
            }

            # Aggregate counts for the ETL form
            for pest, count in crop_pest_counts.items():
                total_pest_counts_all_crops[pest] += count
        
        return pest_results_by_crop, dict(total_pest_counts_all_crops)

    except Exception as e:
        # Return the error as a tuple, matching the structure
        return None, f"Error in Pest Pipeline: {e}"