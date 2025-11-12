import streamlit as st
import os              # NEW: To read environment variables
import requests        # NEW: To make HTTP API calls
import base64          # NEW: To encode images for API transfer
import io              # NEW: To handle image bytes
from PIL import Image

# --- Configuration (LOADED FROM OS ENVIRONMENT) ---
HUGGINGFACE_API_URL = os.environ.get(
    "HF_INFERENCE_ENDPOINT_URL", 
    "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch16" # Fallback/Reference URL
)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# Setup Headers for API call with the token
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
} if HF_API_TOKEN else {}

# --- Prompt Definitions and Mapping ---
# These prompts were used to compute the features locally; they are now passed to the API.
PEST_PROMPTS = [
    "a leaf infested with aphids", "a leaf having aphids", "a leaf with whiteflies on it", 
    "a leaf infested by white flies", "a leaf attacked by leafminer insects", 
    "a leaf with leafminer damage", "a leaf infested by Aphis gossypii pests", 
    "a leaf attacked by pests or insects"
]
DISEASE_PROMPTS = [
    "a diseased leaf without any insects", "a leaf infected by fungus or bacteria but no visible pests", 
    "a leaf with curling or yellowing due to disease, not insects", 
    "a leaf showing leaf spot or mosaic disease", "a leaf damaged by nutrient deficiency or virus but not insects"
]
ALL_CANDIDATE_PROMPTS = PEST_PROMPTS + DISEASE_PROMPTS

# Map to convert the best matching detailed prompt back to the main class
PROMPT_TO_MAIN_CLASS_MAP = {prompt: "pest" for prompt in PEST_PROMPTS}
PROMPT_TO_MAIN_CLASS_MAP.update({prompt: "disease" for prompt in DISEASE_PROMPTS})

# --- Utility Function to Encode Image ---
def pil_to_base64(pil_image):
    """Converts a PIL Image to a Base64 string for API payload."""
    if pil_image.mode in ("RGBA", "P"):
        white_bg = Image.new("RGB", pil_image.size, (255, 255, 255))
        if pil_image.mode == "RGBA":
            white_bg.paste(pil_image, (0, 0), pil_image)
        else:
            pil_image = pil_image.convert("RGB")
            white_bg.paste(pil_image, (0, 0))
        img_to_encode = white_bg
    else:
        img_to_encode = pil_image.convert("RGB")

    buff = io.BytesIO()
    img_to_encode.save(buff, format="JPEG", quality=90) 
    return base64.b64encode(buff.getvalue()).decode("utf-8")


# --- Model Loading (Cached) - NOW CONFIGURATION CHECK ---

@st.cache_resource
def load_primary_clip_model():
    """
    Replaces local model loading: Checks API configuration and returns necessary config values.
    
    Returns: (api_url, api_status_flag, device_str)
    """
    if not HF_API_TOKEN or not HUGGINGFACE_API_URL:
        st.error("Error: HF_API_TOKEN or HUGGINGFACE_API_URL is missing.")
        # Returns None, None, None to signal failure, matching original logic
        return None, None, None 

    print(f"[INFO] Using Primary CLIP Model via remote endpoint: {HUGGINGFACE_API_URL}...")
    
    # We return the API_URL (model), a dummy status (preprocess), and a dummy device string
    return HUGGINGFACE_API_URL, True, "remote" 


# --- Feature Encoding (Cached) - NOW PROMPT CACHING ---

@st.cache_resource
def get_primary_clip_features(_model_url): 
    """
    Replaces feature encoding: Returns the necessary text prompts/map.
    
    Returns: (ALL_CANDIDATE_PROMPTS, PROMPT_TO_MAIN_CLASS_MAP)
    """
    if _model_url is None:
        st.warning("Cannot compute features because the API configuration failed.")
        # Returns two dummy objects to match the original structure/failure mode
        return [], {}
    
    print("Caching primary pest/disease text prompts for remote use...")
    # This function now returns the prompts list and the mapping dictionary
    return ALL_CANDIDATE_PROMPTS, PROMPT_TO_MAIN_CLASS_MAP


# --- Main Classification Function - NOW REMOTE API CALLER ---

def run_primary_classification(image_batch, api_url, _, prompts, mapping, __):
    """
    Classifies a batch of images by calling the remote Hugging Face API.
    
    Args: (Matching original signature)
        image_batch (list): List of (filename, PIL.Image) tuples.
        api_url (str): The HUGGINGFACE_API_URL (replaces 'model').
        _ (bool): Replaces 'preprocess', ignored.
        prompts (list): Replaces 'pest_text_features', holds ALL_CANDIDATE_PROMPTS.
        mapping (dict): Replaces 'disease_text_features', holds PROMPT_TO_MAIN_CLASS_MAP.
        __ (str): Replaces 'device', ignored.

    Returns:
        (list, list): A tuple containing (pest_batch, disease_batch).
    """
    if api_url is None or not HF_API_TOKEN:
        st.error("Classification skipped: API configuration or token is missing.")
        return [], []
        
    pest_batch = []
    disease_batch = []

    for filename, pil_image in image_batch:
        base64_image = pil_to_base64(pil_image)
        
        # Payload structure for HF Zero-Shot Image Classification
        payload = {
            "inputs": base64_image,
            "parameters": {
                "candidate_labels": prompts 
            }
        }

        try:
            response = requests.post(api_url, headers=HEADERS, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            
            if result and isinstance(result, list):
                # 1. Get the prompt with the highest score
                best_match_prompt = result[0].get("label")
                
                # 2. Map the best-match prompt back to the main class ("pest" or "disease")
                main_class = mapping.get(best_match_prompt, "unknown")

                if main_class == "pest":
                    pest_batch.append((filename, pil_image))
                elif main_class == "disease":
                    disease_batch.append((filename, pil_image))
            else:
                 st.warning(f"File **{filename}**: API returned an invalid response structure.")

        except requests.exceptions.RequestException as e:
            st.error(f"File **{filename}**: API request failed. Error: {e}")
            
    return pest_batch, disease_batch
