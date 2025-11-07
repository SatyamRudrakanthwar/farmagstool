import streamlit as st
import torch
# We use transformers for the model, so the standard 'clip' library import is redundant but harmless.
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import clip # Kept for potential compatibility, though not used in feature extraction fix

# --- Model Loading (Cached) ---

@st.cache_resource
def load_primary_clip_model():
    """
    Loads the CLIP model and processor function directly from Hugging Face Hub.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Primary CLIP Model (Hugging Face) on {device}...")

    try:
        model_id = "srrudra78/agrisavant-clip-model"
        # The token is handled by the HF_TOKEN environment variable
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id) # Renamed preprocess to processor for clarity
        print("✅ Primary CLIP Model loaded successfully from Hugging Face.")
        return model, processor, device

    except Exception as e:
        st.error(f"❌ Failed to load CLIP model from Hugging Face: {e}")
        return None, None, None

# ----------------------------------------------------------------------
# --- Feature Extraction (Fixes Caching/AttributeError) ---
# ----------------------------------------------------------------------

@st.cache_resource
# FIX: Renamed 'model' to '_model' to prevent UnhashableParamError
def get_primary_clip_features(_model, processor): 
    """
    Encodes and caches the pest/disease text prompts using the Hugging Face processor.
    """
    
    # Define text prompts
    pest_prompts = [
        "a leaf infested with aphids",
        "a leaf having aphids",
        "a leaf with whiteflies on it",
        "a leaf attacked by white flies",
        "a leaf attacked by leafminer insects",
        "a leaf with leafminer damage",
        "a leaf infested by Aphis gossypii pests",
        "a leaf attacked by pests or insects"
    ]
    
    disease_prompts = [
        "a diseased leaf without any insects",
        "a leaf infected by fungus or bacteria but no visible pests",
        "a leaf with curling or yellowing due to disease, not insects",
        "a leaf showing leaf spot or mosaic disease",
        "a leaf damaged by nutrient deficiency or virus but not insects"
    ]
    
    print("Encoding primary pest/disease text prompts...")
    
    # 1. FIX: TOKENIZE using the Hugging Face processor (CLIPProcessor)
    # This prepares the tokens in the format the HF model expects.
    pest_tokens = processor(pest_prompts, padding=True, return_tensors="pt") 
    disease_tokens = processor(disease_prompts, padding=True, return_tensors="pt")
    
    # Determine the model's device
    device = _model.device
    
    # Move tokens to the model's device (CPU/CUDA)
    pest_tokens = {k: v.to(device) for k, v in pest_tokens.items()}
    disease_tokens = {k: v.to(device) for k, v in disease_tokens.items()}
    
    with torch.no_grad():
        # 2. FIX: Use .get_text_features(**tokens) to replace .encode_text()
        # The ** syntax unpacks the token dictionary (input_ids, attention_mask)
        pest_features = _model.get_text_features(**pest_tokens).mean(dim=0, keepdim=True)
        disease_features = _model.get_text_features(**disease_tokens).mean(dim=0, keepdim=True)
        
        # 3. Normalize features
        pest_features /= pest_features.norm(dim=-1, keepdim=True)
        disease_features /= disease_features.norm(dim=-1, keepdim=True)
        
    return pest_features, disease_features

# ----------------------------------------------------------------------
# --- Main Classification Function ---
# ----------------------------------------------------------------------

def run_primary_classification(image_batch, model, preprocess, pest_text_features, disease_text_features, device):
    """
    Classifies a batch of images into 'pest' or 'disease' lists.
    
    Args:
        image_batch (list): List of (filename, PIL.Image) tuples.
        model: The loaded CLIP model.
        preprocess: The CLIP preprocess function (CLIPProcessor).
        pest_text_features (torch.Tensor): Encoded pest prompts.
        disease_text_features (torch.Tensor): Encoded disease prompts.
        device (str): 'cuda' or 'cpu'.

    Returns:
        (list, list): A tuple containing (pest_batch, disease_batch).
    """
    pest_batch = []
    disease_batch = []

    for filename, pil_image in image_batch:
        
        # --- Image Preprocessing ---
        # Ensure image is 3-channel RGB for CLIP
        if pil_image.mode == "RGBA":
            white_bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            white_bg.paste(pil_image, (0, 0), pil_image)
            image_rgb = white_bg
        else:
            image_rgb = pil_image.convert("RGB")
        
        # Apply CLIP's preprocessing (assumes 'preprocess' is the CLIPProcessor object)
        processed_image = preprocess(images=image_rgb, return_tensors="pt").to(device)
        
        # --- Scoring ---
        with torch.no_grad():
            # NOTE: For HF transformers, use model.get_image_features() for clarity
            image_features = model.get_image_features(**processed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            pest_score = (image_features @ pest_text_features.T).item()
            disease_score = (image_features @ disease_text_features.T).item()
            
        # --- Classification ---
        if pest_score > disease_score:
            pest_batch.append((filename, pil_image)) 
        else:
            disease_batch.append((filename, pil_image))
            
    return pest_batch, disease_batch
