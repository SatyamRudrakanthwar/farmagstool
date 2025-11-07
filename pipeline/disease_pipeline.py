import streamlit as st
from PIL import Image
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModel
# --- UPDATED IMPORTS ---
from sklearn.cluster import KMeans  # Replaced DBSCAN
from collections import defaultdict
import torch

# --- Import your new BG remover function ---
try:
    from pipeline.bg_remover import remove_bg_from_pil_and_get_bgr
except ImportError:
    st.error("Could not import bg_remover.py.")
    # Dummy function
    def remove_bg_from_pil_and_get_bgr(pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# --- Import your new color analysis function ---
try:
    from pipeline.color_analysis import run_batch_color_analysis, generate_individual_color_graph
except ImportError:
    st.error("Could not import color_analysis.py.")
    # Create a dummy function to prevent crashes
    def run_batch_color_analysis(image_group):
        palette = np.zeros((256, 512, 3), dtype=np.uint8)
        cv2.putText(palette, "Batch Color Analysis Failed", (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return cv2.cvtColor(palette, cv2.COLOR_BGR2RGB)
    def generate_individual_color_graph(image_bgr):
        palette = np.zeros((256, 512, 3), dtype=np.uint8)
        cv2.putText(palette, "Individual Color Analysis Failed", (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return cv2.cvtColor(palette, cv2.COLOR_BGR2RGB)


@st.cache_resource
def load_dino_model():
    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"[INFO] Loading DINOv2 (facebook/dinov2-small) on {device}...")
        print("[INFO] DINOv2 model loaded successfully.")
        return model, processor, device
    except Exception as e:
        print(f"Error loading DINOv2 model: {e}")
        return None, None, None

# ---
# --- MODIFIED: run_crop_classification (Using KMeans)
# ---
def run_crop_classification(image_batch, model, processor, device, num_clusters=3):
    """
    Classifies images into a fixed number of crop clusters using KMeans.
    """
    if not image_batch or model is None or processor is None:
        return {}
    
    features = []
    with torch.no_grad():
        for _, pil_image in image_batch:
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            features.append(feature)
    
    if not features:
        return {}

    features_array = np.array(features)
    
    # Handle edge case: less images than clusters
    if len(features_array) < num_clusters:
        print(f"Warning: Only {len(features_array)} images, but {num_clusters} clusters requested. Placing all in 'Crop 1'.")
        # Fallback: just put all images in "Crop 1" and ensure other keys exist
        final_groups = {"Crop 1": image_batch}
        for i in range(1, num_clusters):
            final_groups[f"Crop {i + 1}"] = []
        return final_groups

    # Run KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(features_array)
    labels = kmeans.labels_
    
    # Group images based on labels
    crop_groups_by_label = defaultdict(list)
    for i, label in enumerate(labels):
        crop_groups_by_label[label].append(image_batch[i])
        
    # Map labels (0, 1, 2) to names ("Crop 1", "Crop 2", "Crop 3")
    final_groups = {}
    for i in range(num_clusters):
        key_name = f"Crop {i + 1}"
        final_groups[key_name] = crop_groups_by_label.get(i, []) # Get images for label i, or empty list
            
    print(f"KMeans Crop Results: " + ", ".join([f"C{i+1}={len(final_groups[f'Crop {i+1}'])}" for i in range(num_clusters)]))
    return final_groups

@st.cache_resource
def load_clip_classifier():
    """
    Loads the CLIP model from Hugging Face Hub (fallbacks to local if needed).
    """
    try:
        from transformers import pipeline
        local_model_path = "srrudra78/agrisavant-clip-model"  # Hugging Face repo
        print(f"Loading CLIP model from Hugging Face Hub: {local_model_path}")
        classifier = pipeline(
            task="zero-shot-image-classification",
            model=local_model_path
        )
        print("✅ CLIP model loaded successfully from Hugging Face.")
        return classifier
    except Exception as e:
        import streamlit as st
        st.error(f"Error loading CLIP model from Hugging Face: {e}")
        return None


def run_health_classification(crop_name, image_group, classifier):
    if not image_group or classifier is None:
        return {"healthy": [], "unhealthy": []}
    
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

# ---
# --- THIS IS THE UPDATED FUNCTION ---
# ---
def run_disease_classification(crop_name, unhealthy_images, model, processor, device, num_clusters=3):
    """
    Classifies unhealthy images into a fixed number of disease clusters using KMeans.
    ALWAYS returns keys for 'Disease A', 'Disease B', and 'Disease C'.
    """
    if not unhealthy_images or model is None or processor is None:
        return {}

    features = []
    with torch.no_grad():
        for _, pil_image in unhealthy_images:
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            features.append(feature)
    
    # This dictionary will hold the images grouped by cluster ID (0, 1, 2)
    grouped_by_id = defaultdict(list)

    if not features:
        # No features, but we still return the empty structure
        pass # grouped_by_id is already an empty defaultdict

    else:
        features_array = np.array(features)
        
        # Handle edge case: less images than clusters
        if len(features_array) < num_clusters:
            print(f"Warning: Only {len(features_array)} unhealthy images, but {num_clusters} clusters requested.")
            # Put all available images into the first cluster (label 0)
            for i in range(len(unhealthy_images)):
                grouped_by_id[0].append(unhealthy_images[i])
        else:
            # We have enough images, so run KMeans
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(features_array)
            labels = kmeans.labels_
            
            # Group images based on their assigned label
            for i, label in enumerate(labels):
                grouped_by_id[label].append(unhealthy_images[i])

    # --- This is the key change ---
    # ALWAYS create all three keys, pulling from the grouped_by_id dict.
    # If a key (e.g., 1) doesn't exist in grouped_by_id, .get() will return 
    # the default value (an empty list).
    final_disease_groups = {
        f"{crop_name} Disease A": grouped_by_id.get(0, []),
        f"{crop_name} Disease B": grouped_by_id.get(1, []),
        f"{crop_name} Disease C": grouped_by_id.get(2, [])
    }
    
    print(f"Disease Results: A={len(final_disease_groups[f'{crop_name} Disease A'])}, B={len(final_disease_groups[f'{crop_name} Disease B'])}, C={len(final_disease_groups[f'{crop_name} Disease C'])}")
    return final_disease_groups


def run_disease_pipeline_by_crop(crop_groups, dino_model, dino_processor, dino_device, clip_classifier, global_bg_removed: bool = False):
    """
    Runs the full pipeline and returns a new data structure:
    (image_data_list, aggregated_palette)
    """
    try:
        final_sorting_results = {}
        
        for crop_name, image_group in crop_groups.items():
            if not image_group: 
                final_sorting_results[crop_name] = {
                    "healthy": ([], None), # (image_data_list, aggregated_palette)
                    "unhealthy_by_disease": {}
                }
                continue
            
            # 1. Health/Disease Sorting
            health_results = run_health_classification(crop_name, image_group, clip_classifier)
            healthy_images = health_results["healthy"]
            unhealthy_images = health_results["unhealthy"]

            # 2. Process HEALTHY images
            healthy_aggregated_palette = None
            healthy_image_data = [] # <-- This will be the list of dicts
            all_healthy_bg_removed_bgr = [] # For aggregated palette

            if healthy_images:
                for fname, img_pil in healthy_images:
                    
                    # --- THIS IS THE FIX (Block 1) ---
                    if global_bg_removed:
                        # 1. Convert PIL RGBA to 4-channel numpy array
                        rgba_np = np.array(img_pil.convert("RGBA"))
                        
                        # 2. Get the alpha mask
                        alpha = rgba_np[:, :, 3] / 255.0
                        alpha_mask = cv2.merge([alpha, alpha, alpha])
                        
                        # 3. Get the RGB channels (and convert to BGR)
                        rgb = rgba_np[:, :, :3]
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # Convert from PIL's RGB to CV2's BGR
                        
                        # 4. Create white background
                        bg = np.full(bgr.shape, 255, dtype=np.uint8)
                        
                        # 5. Blend
                        bgr_img = (bgr * alpha_mask + bg * (1 - alpha_mask)).astype(np.uint8)
                    else:
                        # Not yet removed. Run the internal remover (this already returns BGR-on-white)
                        bgr_img = remove_bg_from_pil_and_get_bgr(img_pil)

                    all_healthy_bg_removed_bgr.append((fname, bgr_img))
                    
                    # 2b. Get individual color graph
                    individual_palette = generate_individual_color_graph(bgr_img)
                    
                    # 2c. Store all data for this image
                    healthy_image_data.append({
                        "fname": fname,
                        "original": img_pil,
                        "bg_removed": bgr_img,
                        "individual_palette": individual_palette
                    })
                
                # 2d. Run batch color analysis on ALL bg-removed images
                healthy_aggregated_palette = run_batch_color_analysis(all_healthy_bg_removed_bgr)

            # 3. Process UNHEALTHY images
            disease_groups_with_palettes = {}
            if unhealthy_images:
                # --- THIS CALL IS NOW UPDATED ---
                # It calls the new KMeans-based function.
                disease_groups_raw = run_disease_classification(
                    crop_name, unhealthy_images, dino_model, dino_processor, dino_device
                )
                
                for disease_name, disease_image_group in disease_groups_raw.items():
                    aggregated_palette = None
                    image_data_list = [] # <-- This will be the list of dicts
                    all_disease_bg_removed_bgr = [] # For aggregated palette

                    if disease_image_group:
                        for fname, img_pil in disease_image_group:
                            
                            # --- THIS IS THE FIX (Block 2) ---
                            if global_bg_removed:
                                # 1. Convert PIL RGBA to 4-channel numpy array
                                rgba_np = np.array(img_pil.convert("RGBA"))
                                
                                # 2. Get the alpha mask
                                alpha = rgba_np[:, :, 3] / 255.0
                                alpha_mask = cv2.merge([alpha, alpha, alpha])
                                
                                # 3. Get the RGB channels (and convert to BGR)
                                rgb = rgba_np[:, :, :3]
                                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # Convert from PIL's RGB to CV2's BGR
                                
                                # 4. Create white background
                                bg = np.full(bgr.shape, 255, dtype=np.uint8)
                                
                                # 5. Blend
                                bgr_img = (bgr * alpha_mask + bg * (1 - alpha_mask)).astype(np.uint8)
                            else:
                                # Not yet removed. Run the internal remover (this already returns BGR-on-white)
                                bgr_img = remove_bg_from_pil_and_get_bgr(img_pil)

                            all_disease_bg_removed_bgr.append((fname, bgr_img))
                            
                            # 3b. Get individual color graph
                            individual_palette = generate_individual_color_graph(bgr_img)
                            
                            # 3c. Store all data for this image
                            image_data_list.append({
                                "fname": fname,
                                "original": img_pil,
                                "bg_removed": bgr_img,
                                "individual_palette": individual_palette
                            })
                        
                        # 3d. Run batch color analysis
                        aggregated_palette = run_batch_color_analysis(all_disease_bg_removed_bgr)
                    
                    # Store as a tuple: (list_of_image_data_dicts, aggregated_palette_image)
                    disease_groups_with_palettes[disease_name] = (image_data_list, aggregated_palette)
            
            # 4. Store all results
            final_sorting_results[crop_name] = {
                # **THIS IS THE FIX**: Return the new data structure
                "healthy": (healthy_image_data, healthy_aggregated_palette),
                "unhealthy_by_disease": disease_groups_with_palettes
            }
        return final_sorting_results
    except Exception as e:
        # Return the error as a tuple, matching the structure
        return None, f"Error in Disease Pipeline: {e}"
