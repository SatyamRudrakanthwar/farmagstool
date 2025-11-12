import streamlit as st
from PIL import Image
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModel, pipeline # Ensure 'pipeline' is imported
from sklearn.cluster import KMeans
from collections import defaultdict
import torch
import boto3      # ADDED: For S3 access
import os
import tempfile   # ADDED: For creating temporary directories
import shutil     # ADDED: For copying/moving directories

# --- S3 CONFIGURATION ---
S3_BUCKET_NAME = "srrudra-agrisavant-models" 
# Key must point to the FOLDER containing the CLIP model files (e.g., config.json, pytorch_model.bin)
S3_CLIP_MODEL_KEY = "models/clip-model" 

# --- S3 Helper Function: Recursive Download ---
def download_s3_folder(bucket_name, s3_folder, local_dir):
    """Downloads all files from an S3 folder (prefix) to a local directory."""
    s3 = boto3.client('s3')
    bucket = bucket_name
    prefix = s3_folder.rstrip('/') + '/'
    
    # 1. Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        
    # 2. List objects with the specified prefix
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in objects:
        raise FileNotFoundError(f"No objects found in s3://{bucket}/{prefix}")

    # 3. Download each file
    for obj in objects['Contents']:
        # Construct the local file path
        # Removes the S3 folder prefix to maintain the local folder structure
        relative_path = obj['Key'][len(prefix):]
        if not relative_path: # Skip the folder itself if listed
            continue
            
        local_file_path = os.path.join(local_dir, relative_path)
        
        # Create any necessary subdirectories locally
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        s3.download_file(bucket, obj['Key'], local_file_path)
    return local_dir


# --- load_dino_model and run_crop_classification remain unchanged ---
@st.cache_resource
def load_dino_model():
    # ... (DINO model loading logic) ...
    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"[INFO] Loading DINOv2 (facebook/dinov2-small) on {device}...")
        return model, processor, device
    except Exception as e:
        print(f"Error loading DINOv2 model: {e}")
        return None, None, None

def run_crop_classification(image_batch, model, processor, device, num_clusters=3):
    # ... (Crop classification logic) ...
    if not image_batch or model is None or processor is None: return {}
    features = []
    with torch.no_grad():
        for _, pil_image in image_batch:
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            features.append(feature)
    # ... (KMeans clustering and grouping logic) ...
    features_array = np.array(features)
    if len(features_array) < num_clusters:
        final_groups = {"Crop 1": image_batch}
        for i in range(1, num_clusters): final_groups[f"Crop {i + 1}"] = []
        return final_groups
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(features_array)
    labels = kmeans.labels_
    crop_groups_by_label = defaultdict(list)
    for i, label in enumerate(labels):
        crop_groups_by_label[label].append(image_batch[i])
    final_groups = {}
    for i in range(num_clusters):
        key_name = f"Crop {i + 1}"
        final_groups[key_name] = crop_groups_by_label.get(i, [])
    return final_groups


# --- load_clip_classifier (FIXED: Loads from S3) ---
@st.cache_resource
def load_clip_classifier():
    """
    Loads the secondary CLIP classifier (for health check) from Hugging Face Hub.
    This resolves the 'models/clip-model is not a local folder' error.
    """
    try:
        from transformers import pipeline
        
        # This replaces the failing local path ("models/clip-model")
        # and bypasses the complex S3 folder download.
        model_id = "openai/clip-vit-base-patch16" 
        
        print(f"[INFO] Loading secondary CLIP classifier from {model_id}...")
        classifier = pipeline(task="zero-shot-image-classification", model=model_id)
        return classifier
        
    except Exception as e:
        # This handles general download/installation errors
        st.error(f"Error loading secondary CLIP model: {e}")
        return None
        
    finally:
        # 4. Clean up the entire temporary directory structure
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# --- run_health_classification, run_disease_classification, and run_disease_pipeline_by_crop remain unchanged ---

def run_health_classification(crop_name, image_group, classifier):
    if not image_group or classifier is None: return {"healthy": [], "unhealthy": []}
    candidate_labels = ["healthy leaf", "unhealthy leaf with disease, spots, or pests"]
    health_results = {"healthy": [], "unhealthy": []}
    pil_images = [img for _, img in image_group]
    classifications = classifier(pil_images, candidate_labels=candidate_labels)
    for i, classification in enumerate(classifications):
        best_score = classification[0]
        if best_score['label'] == 'healthy leaf':
            health_results["healthy"].append(image_group[i])
        else:
            health_results["unhealthy"].append(image_group[i])
    return health_results

def run_disease_classification(crop_name, unhealthy_images, model, processor, device, num_clusters=3):
    if not unhealthy_images or model is None or processor is None: return {}
    features = []
    with torch.no_grad():
        for _, pil_image in unhealthy_images:
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            features.append(feature)
    grouped_by_id = defaultdict(list)
    if not features: pass
    else:
        features_array = np.array(features)
        if len(features_array) < num_clusters:
            for i in range(len(unhealthy_images)): grouped_by_id[0].append(unhealthy_images[i])
        else:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(features_array)
            labels = kmeans.labels_
            for i, label in enumerate(labels): grouped_by_id[label].append(unhealthy_images[i])
    final_disease_groups = {
        f"{crop_name} Disease A": grouped_by_id.get(0, []),
        f"{crop_name} Disease B": grouped_by_id.get(1, []),
        f"{crop_name} Disease C": grouped_by_id.get(2, [])
    }
    print(f"Disease Results: A={len(final_disease_groups[f'{crop_name} Disease A'])}, B={len(final_disease_groups[f'{crop_name} Disease B'])}, C={len(final_disease_groups[f'{crop_name} Disease C'])}")
    return final_disease_groups


def run_disease_pipeline_by_crop(crop_groups, dino_model, dino_processor, dino_device, clip_classifier, global_bg_removed: bool = False):
    # ... (rest of the run_disease_pipeline_by_crop function remains unchanged) ...
    RESIZE_DIM = (600, 600)
    
    try:
        final_sorting_results = {}
        
        for crop_name, image_group in crop_groups.items():
            if not image_group: 
                final_sorting_results[crop_name] = {
                    "healthy": ([], None),
                    "unhealthy_by_disease": {}
                }
                continue
            
            # 1. Health/Disease Sorting
            health_results = run_health_classification(crop_name, image_group, clip_classifier)
            healthy_images = health_results["healthy"]
            unhealthy_images = health_results["unhealthy"]

            # 2. Process HEALTHY images
            healthy_aggregated_palette = None
            healthy_image_data = []
            all_healthy_bg_removed_bgr = []

            if healthy_images:
                for fname, img_pil in healthy_images:
                    
                    # Resizing Check (Block 1)
                    if img_pil.size[0] > RESIZE_DIM[0] or img_pil.size[1] > RESIZE_DIM[1]:
                        img_pil = img_pil.resize(RESIZE_DIM, Image.Resampling.LANCZOS)
                    
                    # --- FIX: Block 1 (Healthy Images) - Robust Blending ---
                    if global_bg_removed:
                        rgba_np = np.array(img_pil.convert("RGBA"))
                        alpha = (rgba_np[:, :, 3] / 255.0).astype(np.float32) 
                        bgr_float = cv2.cvtColor(rgba_np[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32) 
                        bg_float = np.full(bgr_float.shape, 255.0, dtype=np.float32)
                        alpha_mask_3ch = cv2.merge([alpha, alpha, alpha])
                        bgr_img = (bgr_float * alpha_mask_3ch + bg_float * (1.0 - alpha_mask_3ch)).astype(np.uint8)
                    else:
                        bgr_img = remove_bg_from_pil_and_get_bgr(img_pil)

                    all_healthy_bg_removed_bgr.append((fname, bgr_img))
                    
                    individual_palette = generate_individual_color_graph(bgr_img)
                    
                    healthy_image_data.append({
                        "fname": fname, "original": img_pil, "bg_removed": bgr_img, "individual_palette": individual_palette
                    })
                
                healthy_aggregated_palette = run_batch_color_analysis(all_healthy_bg_removed_bgr)

            # 3. Process UNHEALTHY images
            disease_groups_with_palettes = {}
            if unhealthy_images:
                disease_groups_raw = run_disease_classification(
                    crop_name, unhealthy_images, dino_model, dino_processor, dino_device
                )
                
                for disease_name, disease_image_group in disease_groups_raw.items():
                    aggregated_palette = None
                    image_data_list = [] 
                    all_disease_bg_removed_bgr = []

                    if disease_image_group:
                        for fname, img_pil in disease_image_group:
                            
                            if img_pil.size[0] > RESIZE_DIM[0] or img_pil.size[1] > RESIZE_DIM[1]:
                                img_pil = img_pil.resize(RESIZE_DIM, Image.Resampling.LANCZOS)
                            
                            # --- FIX: Block 2 (Disease Images) - Robust Blending ---
                            if global_bg_removed:
                                rgba_np = np.array(img_pil.convert("RGBA"))
                                alpha = (rgba_np[:, :, 3] / 255.0).astype(np.float32)
                                bgr_float = cv2.cvtColor(rgba_np[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
                                bg_float = np.full(bgr_float.shape, 255.0, dtype=np.float32)
                                alpha_mask_3ch = cv2.merge([alpha, alpha, alpha])
                                bgr_img = (bgr_float * alpha_mask_3ch + bg_float * (1.0 - alpha_mask_3ch)).astype(np.uint8)
                            else:
                                bgr_img = remove_bg_from_pil_and_get_bgr(img_pil)

                            all_disease_bg_removed_bgr.append((fname, bgr_img))
                            
                            individual_palette = generate_individual_color_graph(bgr_img)
                            
                            image_data_list.append({
                                "fname": fname, "original": img_pil, "bg_removed": bgr_img, "individual_palette": individual_palette
                            })
                        
                        aggregated_palette = run_batch_color_analysis(all_disease_bg_removed_bgr)
                    
                    disease_groups_with_palettes[disease_name] = (image_data_list, aggregated_palette)
            
            final_sorting_results[crop_name] = {
                "healthy": (healthy_image_data, healthy_aggregated_palette),
                "unhealthy_by_disease": disease_groups_with_palettes
            }
        return final_sorting_results
    except Exception as e:
        print(f"Error in Disease Pipeline: {e}")
        return None, f"Error in Disease Pipeline: {e}"
