import streamlit as st
import torch
import clip
from PIL import Image

# --- Model Loading (Cached) ---

@st.cache_resource
def load_primary_clip_model():
    """
    Loads the CLIP model and preprocess function directly from Hugging Face Hub.
    """
    import torch
    import streamlit as st
    from transformers import CLIPProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Primary CLIP Model (Hugging Face) on {device}...")

    try:
        # Load from Hugging Face Hub (your uploaded model)
        model_id = "srrudra78/agrisavant-clip-model"
        model = CLIPModel.from_pretrained(model_id).to(device)
        preprocess = CLIPProcessor.from_pretrained(model_id)
        print("✅ Primary CLIP Model loaded successfully from Hugging Face.")
        return model, preprocess, device

    except Exception as e:
        st.error(f"❌ Failed to load CLIP model from Hugging Face: {e}")
        return None, None, None

@st.cache_resource
def get_primary_clip_features(_model): # Pass model to link cache
    """
    Encodes and caches the pest/disease text prompts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pest_prompts = [
        "a leaf infested with aphids",
        "a leaf having aphids",
        "a leaf with whiteflies on it",
        "a leaf infested by white flies",
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
    pest_text_tokens = clip.tokenize(pest_prompts).to(device)
    disease_text_tokens = clip.tokenize(disease_prompts).to(device)
    
    with torch.no_grad():
        pest_features = _model.encode_text(pest_text_tokens).mean(dim=0, keepdim=True)
        disease_features = _model.encode_text(disease_text_tokens).mean(dim=0, keepdim=True)
        
        # Normalize features
        pest_features /= pest_features.norm(dim=-1, keepdim=True)
        disease_features /= disease_features.norm(dim=-1, keepdim=True)
        
    return pest_features, disease_features
# Assuming this function receives the model and the tokenizer/processor from load_primary_clip_model()

@st.cache_resource
def get_primary_clip_features(model, tokenizer): # <-- Ensure it accepts tokenizer/processor
    """
    Encodes and caches the pest/disease text prompts.
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
    
    # --- FIX 1: TOKENIZE using the tokenizer/processor (e.g., loaded by Hugging Face) ---
    # We must use the tokenizer/processor returned by load_primary_clip_model
    pest_tokens = tokenizer(pest_prompts, padding=True, return_tensors="pt") 
    disease_tokens = tokenizer(disease_prompts, padding=True, return_tensors="pt")
    
    # --- FIX 2: GET FEATURES using the model's correct method ---
    with torch.no_grad():
        # Get pest features (ensure model is on the correct device if needed, 
        # but often the features calculation handles it if the inputs are moved)
        # Note: We use .to('cpu') here for safety, assuming the model is loaded on CPU as per the log
        pest_features = model.get_text_features(**pest_tokens).to('cpu').mean(dim=0, keepdim=True)
        disease_features = model.get_text_features(**disease_tokens).to('cpu').mean(dim=0, keepdim=True)
    
    return pest_features, disease_features
# --- Main Classification Function ---

def run_primary_classification(image_batch, model, preprocess, pest_text_features, disease_text_features, device):
    """
    Classifies a batch of images into 'pest' or 'disease' lists.
    
    Args:
        image_batch (list): List of (filename, PIL.Image) tuples.
        model: The loaded CLIP model.
        preprocess: The CLIP preprocess function.
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
        
        # Apply CLIP's preprocessing
        processed_image = preprocess(image_rgb).unsqueeze(0).to(device)
        
        # --- Scoring ---
        with torch.no_grad():
            image_features = model.encode_image(processed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            pest_score = (image_features @ pest_text_features.T).item()
            disease_score = (image_features @ disease_text_features.T).item()
        
        # --- Classification ---
        if pest_score > disease_score:
            pest_batch.append((filename, pil_image)) # Append the *original* image
        else:
            disease_batch.append((filename, pil_image))
            
    return pest_batch, disease_batch
