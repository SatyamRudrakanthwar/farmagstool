import streamlit as st
import requests  # NEW: For making API calls
import os        # NEW: For reading environment variables
import base64    # NEW: For encoding images
import io        # NEW: For handling image bytes
from PIL import Image

# --- Configuration and Environment Setup ---

# Load configuration from OS environment variables
HUGGINGFACE_API_URL = os.environ.get(
    "HF_INFERENCE_ENDPOINT_URL", 
    "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14" 
)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
} if HF_API_TOKEN else {"Content-Type": "application/json"}


# --- Prompt Definitions and Mapping ---
# These replace the torch/clip text feature encoding
PEST_PROMPTS = [
    "a leaf infested with aphids",
    "a leaf having aphids",
    "a leaf with whiteflies on it",
    "a leaf infested by white flies",
    "a leaf attacked by leafminer insects",
    "a leaf with leafminer damage",
    "a leaf infested by Aphis gossypii pests",
    "a leaf attacked by pests or insects"
]
DISEASE_PROMPTS = [
    "a diseased leaf without any insects",
    "a leaf infected by fungus or bacteria but no visible pests",
    "a leaf with curling or yellowing due to disease, not insects",
    "a leaf showing leaf spot or mosaic disease",
    "a leaf damaged by nutrient deficiency or virus but not insects"
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
    Simulates model loading by returning configuration/status.
    
    Returns: (api_status, api_url, device_str)
    """
    if not HF_API_TOKEN:
        st.error("Error: HF_API_TOKEN is not configured in the environment.")
        # Returning None, None, None to signal failure, matching original logic
        return None, None, None 

    print(f"Using Primary CLIP Model via remote endpoint: {HUGGINGFACE_API_URL}...")
    
    # We return the API_URL as the 'model', and True as the 'preprocess' for dependent code
    # The device string is irrelevant but maintained for signature compatibility
    return HUGGINGFACE_API_URL, True, "remote" 


# --- Feature Encoding (Cached) - NOW PROMPT CACHING ---

@st.cache_resource
def get_primary_clip_features(_model_url, _device): 
    """
    Simulates encoding by returning the necessary text prompts/map.
    
    Returns: (ALL_CANDIDATE_PROMPTS, PROMPT_TO_MAIN_CLASS_MAP)
    """
    # The original function returned two tensors. 
    # We must maintain this structure, so we return two objects needed for classification.
    if _model_url is None:
        st.warning("Cannot proceed because the API configuration failed.")
        # Returning two empty lists to match the tensor structure/failure mode
        return [], []
    
    print("Caching primary pest/disease text prompts for remote use...")
    # This function now returns the prompts list and the mapping dictionary
    return ALL_CANDIDATE_PROMPTS, PROMPT_TO_MAIN_CLASS_MAP


# --- Main Classification Function - NOW REMOTE API CALLER ---

def run_primary_classification(image_batch, api_url, _, prompts, mapping, __):
    """
    Classifies a batch of images by calling the remote Hugging Face API.
    
    Args:
        image_batch (list): List of (filename, PIL.Image) tuples.
        api_url (str): The HUGGINGFACE_API_URL (replaces 'model').
        _ (bool): Replaces 'preprocess', ignored.
        prompts (list): Replaces 'pest_text_features', holds ALL_CANDIDATE_PROMPTS.
        mapping (dict): Replaces 'disease_text_features', holds PROMPT_TO_MAIN_CLASS_MAP.
        __ (str): Replaces 'device_str', ignored.

    Returns:
        (list, list): A tuple containing (pest_batch, disease_batch).
    """
    # The first check from the original logic is maintained but adapted
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
                "candidate_labels": prompts # Use the prompts passed from get_primary_clip_features
            }
        }

        try:
            response = requests.post(api_url, headers=HEADERS, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            
            if result and isinstance(result, list):
                # The best match prompt is the first item in the result list
                best_match_prompt = result[0].get("label")
                
                # Map the prompt back to the main class ("pest" or "disease")
                main_class = mapping.get(best_match_prompt, "unknown")

                if main_class == "pest":
                    pest_batch.append((filename, pil_image))
                elif main_class == "disease":
                    disease_batch.append((filename, pil_image))
                else:
                    st.warning(f"File **{filename}**: API returned an unmatched prompt: `{best_match_prompt}`")
            else:
                 st.warning(f"File **{filename}**: API returned an invalid response structure.")


        except requests.exceptions.RequestException as e:
            st.error(f"File **{filename}**: API request failed. Error: {e}")
            
    return pest_batch, disease_batch
