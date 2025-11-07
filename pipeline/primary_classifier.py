import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import clip # Kept for potential compatibility

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
        model = CLIPModel.from_pretrained(model_id).to(device)
        # processor is used as the argument name in get_primary_clip_features
        processor = CLIPProcessor.from_pretrained(model_id) 
        print("✅ Primary CLIP Model loaded successfully from Hugging Face.")
        return model, processor, device

    except Exception as e:
        st.error(f"❌ Failed to load CLIP model from Hugging Face: {e}")
        return None, None, None

# ----------------------------------------------------------------------
# --- Feature Extraction (FIXED) ---
# ----------------------------------------------------------------------

@st.cache_resource
# FIX: Use underscores for both unhashable arguments
def get_primary_clip_features(_model, _processor): 
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
    
    # FIX: Use _processor argument name for tokenization
    pest_tokens = _processor(pest_prompts, padding=True, return_tensors="pt") 
    disease_tokens = _processor(disease_prompts, padding=True, return_tensors="pt")
    
    # Determine the model's device
    device = _model.device
    
    # Move tokens to the model's device (CPU/CUDA)
    pest_tokens = {k: v.to(device) for k, v in pest_tokens.items()}
    disease_tokens = {k: v.to(device) for k, v in disease_tokens.items()}
    
    with torch.no_grad():
        # Use _model for feature extraction
        pest_features = _model.get_text_features(**pest_tokens).mean(dim=0, keepdim=True)
        disease_features = _model.get_text_features(**disease_tokens).mean(dim=0, keepdim=True)
        
        # Normalize features
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
        if pil_image.mode == "RGBA":
            white_bg = Image.new("RGB", pil_image.size, (255, 255, 255))
            white_bg.paste(pil_image, (0, 0), pil_image)
            image_rgb = white_bg
        else:
            image_rgb = pil_image.convert("RGB")
        
        # Apply CLIP's preprocessing (using the processor passed in)
        processed_image = preprocess(images=image_rgb, return_tensors="pt").to(device)
        
        # --- Scoring ---
        with torch.no_grad():
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
