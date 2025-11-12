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
import boto3      # ADDED: AWS SDK
import os         # ADDED: For file path handling
import tempfile   # ADDED: For creating temporary files

# --- S3 CONFIGURATION (MUST BE UPDATED IN DEPLOYMENT ENVIRONMENT) ---
S3_BUCKET_NAME = "srrudra-agrisavant-models" # YOUR ACTUAL BUCKET NAME
S3_MODEL_KEY = "models/best.pt" # Key must match the path in your S3 bucket

# PEST DETECTION (BRANCH 1 - PART 1)-

@st.cache_resource
def load_pest_model():
    """
    Downloads the YOLO model from S3 to a temporary file and loads it.
    This resolves the PyTorch security error by reading from a clean local path.
    """
    # 1. Create a secure temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        local_model_path = tmp_file.name

    try:
        st.info(f"Downloading YOLO model from S3...")
        
        # 2. Initialize the S3 client (Boto3 uses the attached IAM role automatically)
        s3 = boto3.client("s3") 
        
        # 3. Download the file
        s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, local_model_path)
        
        # 4. Load the model from the local temporary file
        model = YOLO(local_model_path)
        
        print(f"[INFO] Pest model loaded successfully from S3 key: {S3_MODEL_KEY}")
        return model
        
    except Exception as e:
        # Crucial error display for S3/IAM role problems
        st.error(f"Error loading Pest Model from S3: {e}. Check IAM role, bucket name, and key.")
        return None
        
    finally:
        # 5. Clean up the temporary file path immediately
        if os.path.exists(local_model_path):
            os.remove(local_model_path)


# --- NEW HELPER FUNCTION 1 (remains unchanged) ---
def get_pests_from_results(results):
    """
    Extracts a list of detected pest class names from YOLO results.
    """
    class_names = results[0].names 
    detected_classes = results[0].boxes.cls.cpu().numpy() 
    
    pests_found = []
    for cls_id in detected_classes:
        pests_found.append(class_names[int(cls_id)])
    
    return pests_found 

# --- NEW HELPER FUNCTION 2 (remains unchanged) ---
def draw_boxes(pil_image, results):
    """
    Draws bounding boxes on an image using Supervision.
    """
    # 1. Convert PIL (RGB) to CV2 (BGR) format
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 2. Convert YOLO results to a standard Supervision Detections object
    detections = sv.Detections.from_ultralytics(results[0])
    
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
    class_names = results[0].names
    labels = [
        f"{class_names[class_id]} {confidence:0.2f}"
        for confidence, class_id in zip(detections.confidence, detections.class_id)
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
    total_pest_counts = {} 
    images_by_pest = defaultdict(list)
    
    annotated_image_cache = {}

    for filename, pil_image in image_batch_with_names:
        
        # --- FIX: Ensure pil_rgb is created properly if RGBA ---
        if pil_image.mode == "RGBA":
            white_bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            white_bg.paste(pil_image, (0, 0), pil_image)
            pil_rgb = white_bg
        else:
            pil_rgb = pil_image.convert("RGB")
        # --- END OF FIX ---

        # 1. Run detection on the 3-channel image
        results = pest_model(pil_rgb) 
        
        pests_found_in_image = get_pests_from_results(results) 

        if not pests_found_in_image:
            continue # Skip this image, no pests found

        # 2. Update total counts (counts all instances)
        for pest in pests_found_in_image:
            total_pest_counts[pest] = total_pest_counts.get(pest, 0) + 1
        
        # 3. Get the single annotated image
        if filename in annotated_image_cache:
            annotated_cv2_img = annotated_image_cache[filename]
        else:
            annotated_cv2_img = draw_boxes(pil_rgb, results)
            annotated_image_cache[filename] = annotated_cv2_img
        
        # 4. Add this one image to the list for each *unique* pest found
        for pest_name in set(pests_found_in_image):
            images_by_pest[pest_name].append((filename, annotated_cv2_img))
    
    return total_pest_counts, images_by_pest

# ETL CALCULATION (BRANCH 1 - PART 2)
# ... (All ETL and pipeline functions remain unchanged) ...

def calculate_value_loss(I, market_cost_per_kg):
    """Helper function for ETL calculation."""
    yield_lost = I * 1000
    return yield_lost, market_cost_per_kg * yield_lost

def predict_etl_days(data):
    """
    Core ETL prediction engine.
    """
    etl_days_list = []
    full_progress_data = []

    for row in data:
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
