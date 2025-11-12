import streamlit as st
from PIL import Image
import time
import pandas as pd
import cv2 
import numpy as np
import concurrent.futures
import io 
from collections import defaultdict 
import os # ESSENTIAL: For reading environment variables

# Import all the functions from backend pipeline files
from pipeline.pest_pipeline import (
    load_pest_model,
    run_pest_detection_batch,
    run_etl_calculation,
    run_pest_pipeline_by_crop
)
from pipeline.disease_pipeline import (
    load_dino_model,
    run_crop_classification,
    load_clip_classifier,
    run_health_classification,
    run_disease_classification,
    run_disease_pipeline_by_crop
)
# --- Import primary classifier functions (now using remote API logic) ---
try:
    from pipeline import color_analysis
    from pipeline import bg_remover 
    
    from pipeline.primary_classifier import (
        load_primary_clip_model,
        get_primary_clip_features,
        run_primary_classification
    )
except ImportError as e:
    st.error(f"Could not import pipeline modules. Ensure all pipeline files are present: {e}")
    # Define placeholder functions to prevent app crash if modules are missing
    def load_primary_clip_model(): return None, None, None
    def get_primary_clip_features(*args): return [], {}
    def run_primary_classification(*args): return [], []

# -----------------------------------------------------------------
# --- Worker Functions (Wrappers for Parallel Execution) ---
# -----------------------------------------------------------------

def run_pest_pipeline_wrapper(crop_groups, model):
    """Wrapper for the pest pipeline."""
    try:
        pest_results, etl_counts = run_pest_pipeline_by_crop(crop_groups, model)
        return pest_results, etl_counts
    except Exception as e:
        return None, f"Error in Pest Pipeline: {e}"

def run_disease_pipeline_wrapper(crop_groups, dino_model, dino_processor, dino_device, clip_classifier, global_bg_removed):
    """Wrapper for the disease pipeline."""
    try:
        disease_results = run_disease_pipeline_by_crop(
            crop_groups, 
            dino_model, 
            dino_processor, 
            dino_device, 
            clip_classifier, 
            global_bg_removed 
        )
        return disease_results
    except Exception as e:
        return None, f"Error in Disease Pipeline: {e}"


def reset_app():
    # Clear all result/data keys
    st.session_state.pest_results_by_crop = {}
    st.session_state.disease_results_by_crop = {}
    st.session_state.total_pest_counts_for_etl = {}
    st.session_state.etl_inputs_ready = False
    st.session_state.image_batch_with_names = []
    st.session_state.processed_filenames = []
    
    # Increment the key to force the file_uploader to re-render
    st.session_state.uploader_key += 1

# -----------------------------------------------------------------
# --- Initialize Session State (MUST BE AT THE TOP) ---
# -----------------------------------------------------------------
if 'pest_results_by_crop' not in st.session_state: st.session_state.pest_results_by_crop = {}
if 'disease_results_by_crop' not in st.session_state: st.session_state.disease_results_by_crop = {}
if 'total_pest_counts_for_etl' not in st.session_state: st.session_state.total_pest_counts_for_etl = {}
if 'etl_inputs_ready' not in st.session_state: st.session_state.etl_inputs_ready = False
if 'image_batch_with_names' not in st.session_state: st.session_state.image_batch_with_names = []
if 'processed_filenames' not in st.session_state: st.session_state.processed_filenames = []
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0 
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------
st.set_page_config(layout="wide", page_title="AGS Farmhealth analyser Tool")

col1, col2 = st.columns([1, 6]) 

with col1:
    st.image("assets/logo.jpeg")

with col2:
    st.title("🚜 AGS Farm Health AnalysCer")
    st.write("Upload a batch of images to run Pest/ETL and Crop/Disease analysis.")


# CUSTOM CSS STYLING 
st.markdown("""
    <style>
    /* Add your custom CSS here if needed */
    </style>
    """, unsafe_allow_html=True)

# File Uploader
uploaded_files = st.file_uploader(
    "Choose images...",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.uploader_key}"
)

# -----------------------------------------------------------------
# --- Sidebar Controls and Configuration Status ---
# -----------------------------------------------------------------
st.sidebar.button("Clear All & Reset App", on_click=reset_app, use_container_width=True, type="secondary")
st.sidebar.divider()
st.sidebar.subheader("Processing Options")
global_bg_remove_enabled = st.sidebar.toggle(
    "Remove Background (Global)", 
    value=False, 
    help="If ON, removes background from ALL images *after* Primary Classification."
)
st.sidebar.divider()
st.sidebar.subheader("API Configuration Status")
st.sidebar.markdown(f"**HF Endpoint:** `{os.environ.get('HF_INFERENCE_ENDPOINT_URL', 'NOT SET').split('//')[-1]}`")
st.sidebar.markdown(f"**HF Token:** `{'✅ SET' if os.environ.get('HF_API_TOKEN') else '❌ NOT SET'}`")

st.sidebar.info("Upload multiple images and click 'Run Analysis'.")


# -----------------------------------------------------------------
# Main Processing Logic
# -----------------------------------------------------------------

if uploaded_files:
    
    current_filenames = [f.name for f in uploaded_files]
    
    if st.session_state.processed_filenames != current_filenames:
        st.session_state.pest_results_by_crop = {}
        st.session_state.disease_results_by_crop = {}
        st.session_state.total_pest_counts_for_etl = {}
        st.session_state.etl_inputs_ready = False
        st.session_state.image_batch_with_names = []
        st.session_state.processed_filenames = []
        
        temp_image_batch = []
        with st.spinner(f"Loading {len(uploaded_files)} images..."):
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                temp_image_batch.append((uploaded_file.name, img.copy()))
        
        st.session_state.image_batch_with_names = temp_image_batch
        st.session_state.processed_filenames = current_filenames
    
    st.subheader(f"Uploaded {len(st.session_state.image_batch_with_names)} Image(s)")
    cols = st.columns(min(len(st.session_state.image_batch_with_names), 10))
    for i, (name, img) in enumerate(st.session_state.image_batch_with_names):
        with cols[i % 10]:
            st.image(img, caption=name, width=100)
    st.markdown("---")


    if st.button(f"Run Full Analysis on {len(uploaded_files)} Images", type="primary", use_container_width=True):
        
        st.session_state.pest_results_by_crop = {}
        st.session_state.disease_results_by_crop = {}
        st.session_state.total_pest_counts_for_etl = {}
        st.session_state.etl_inputs_ready = False
        
        with st.spinner("Loading analysis models... (first time might take a while)"):
            # Load all models
            pest_model = load_pest_model()
            dino_model, dino_processor, dino_device = load_dino_model()
            clip_classifier = load_clip_classifier() 
            
            # --- Primary Classification (Loads remote configuration) ---
            primary_clip_model, primary_preprocess, primary_device = load_primary_clip_model()
            pest_text_features, disease_text_features = get_primary_clip_features(primary_clip_model, primary_device)

        if not all([pest_model, dino_model, clip_classifier, primary_clip_model]):
            st.error("One or more models or API configurations failed to load. Cannot proceed.")
        else:
            
            original_image_batch = st.session_state.image_batch_with_names
            
            # --- Step 1/5: Primary Classification ---
            with st.spinner("Step 1/5: Classifying Pest vs. Disease (on raw images) via Hugging Face API..."):
                # The function signature remains identical to the local version
                pest_image_batch, disease_image_batch = run_primary_classification(
                    original_image_batch, 
                    primary_clip_model,
                    primary_preprocess,
                    pest_text_features,
                    disease_text_features,
                    primary_device
                )
            st.info(f"Primary classification: {len(pest_image_batch)} pest images, {len(disease_image_batch)} disease images found.")

            # --- Step 2/5: Global Background Removal (Logic is preserved) ---
            pest_batch_for_pipeline = pest_image_batch
            disease_batch_for_pipeline = disease_image_batch

            if global_bg_remove_enabled:
                st.info("Global background removal is ON. Pre-processing sorted images...")
                
                processed_pest_batch = []
                with st.spinner(f"Removing background from {len(pest_image_batch)} pest images..."):
                    for fname, img in pest_image_batch:
                        try:
                            pil_rgb = img.convert("RGB")
                            bgr_image = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
                            rgba_np, result, _, _ = bg_remover.remove_background_hybrid(bgr_image)
                            if result == "Success" and rgba_np is not None:
                                b, g, r, a = cv2.split(rgba_np)
                                rgba_pil_np = cv2.merge([r, g, b, a])
                                processed_img = Image.fromarray(rgba_pil_np, 'RGBA')
                                processed_pest_batch.append((fname, processed_img))
                            else:
                                st.warning(f"BG removal failed for {fname} (pest). Using original.")
                                processed_pest_batch.append((fname, img.convert("RGBA")))
                        except Exception as e:
                            st.error(f"Error during BG removal on {fname} (pest): {e}. Using original.")
                            processed_pest_batch.append((fname, img.convert("RGBA")))
                
                processed_disease_batch = []
                with st.spinner(f"Removing background from {len(disease_image_batch)} disease images..."):
                    for fname, img in disease_image_batch:
                        try:
                            pil_rgb = img.convert("RGB")
                            bgr_image = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
                            rgba_np, result, _, _ = bg_remover.remove_background_hybrid(bgr_image)
                            if result == "Success" and rgba_np is not None:
                                b, g, r, a = cv2.split(rgba_np)
                                rgba_pil_np = cv2.merge([r, g, b, a])
                                processed_img = Image.fromarray(rgba_pil_np, 'RGBA')
                                processed_disease_batch.append((fname, processed_img))
                            else:
                                st.warning(f"BG removal failed for {fname} (disease). Using original.")
                                processed_disease_batch.append((fname, img.convert("RGBA")))
                        except Exception as e:
                            st.error(f"Error during BG removal on {fname} (disease): {e}. Using original.")
                            processed_disease_batch.append((fname, img.convert("RGBA")))

                pest_batch_for_pipeline = processed_pest_batch
                disease_batch_for_pipeline = processed_disease_batch
            
            # --- Steps 3-5 (Crop Classification, Parallel Pipeline, Finalizing) ---
            all_images_for_pipeline = pest_batch_for_pipeline + disease_batch_for_pipeline
            
            with st.spinner("Step 3/5: Classifying crops..."):
                pest_crop_groups = run_crop_classification(pest_batch_for_pipeline, dino_model, dino_processor, dino_device)
                all_images_crop_groups = run_crop_classification(all_images_for_pipeline, dino_model, dino_processor, dino_device)

            with st.spinner("Step 4/5: Running Pest and Disease analysis..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    pest_future = executor.submit(run_pest_pipeline_wrapper, pest_crop_groups, pest_model)
                    disease_future = executor.submit(
                        run_disease_pipeline_wrapper, 
                        all_images_crop_groups, 
                        dino_model, 
                        dino_processor, 
                        dino_device, 
                        clip_classifier,
                        global_bg_remove_enabled 
                    )
                    
                    pest_result = pest_future.result()
                    disease_result = disease_future.result()
            
            with st.spinner("Step 5/5: Finalizing results..."):
                if pest_result[0] is not None:
                    pest_results_by_crop, total_pest_counts = pest_result
                    st.session_state.pest_results_by_crop = pest_results_by_crop
                    st.session_state.total_pest_counts_for_etl = total_pest_counts
                    if total_pest_counts: st.session_state.etl_inputs_ready = True
                else: st.error(pest_result[1]) 

                if isinstance(disease_result, dict):
                    st.session_state.disease_results_by_crop = disease_result
                else:
                    if isinstance(disease_result, tuple) and disease_result[0] is None:
                        st.error(disease_result[1])
                    elif disease_result is None:
                        st.error("Disease pipeline returned an empty result.")
                    else:
                        st.error(f"An unknown error occurred in the disease pipeline: {disease_result}")
            
            st.success("Analysis Complete!", icon="✅") 
    
elif not uploaded_files and st.session_state.processed_filenames:
    st.session_state.pest_results_by_crop = {}
    st.session_state.disease_results_by_crop = {}
    st.session_state.total_pest_counts_for_etl = {}
    st.session_state.etl_inputs_ready = False
    st.session_state.image_batch_with_names = []
    st.session_state.processed_filenames = []
    st.rerun()

# -----------------------------------------------------------------
# --- DISPLAY LOGIC (Remains fully unchanged) ---
# -----------------------------------------------------------------
if st.session_state.pest_results_by_crop or st.session_state.disease_results_by_crop:
    
    col1, col2 = st.columns(2)
    
    pest_crops = set(st.session_state.pest_results_by_crop.keys())
    disease_crops = set(st.session_state.disease_results_by_crop.keys())
    all_crop_names = sorted(list(pest_crops.union(disease_crops))) 

    # --- BRANCH 1: Display Pest & ETL ---
    with col1:
        st.header("🐜 Pest & ETL Analysis")
        pest_results_data = st.session_state.pest_results_by_crop
        
        if not all_crop_names:
            st.warning("Run analysis to see pest results.")
        else:
            pest_crop_tabs = st.tabs([f"{crop} (Pests)" for crop in all_crop_names])

            for i, crop_name in enumerate(all_crop_names):
                with pest_crop_tabs[i]:
                    data = pest_results_data.get(crop_name, {}) 
                    pest_counts = data.get('pest_counts', {})
                    images_by_pest = data.get('images_by_pest', {})

                    if not pest_counts and not images_by_pest:
                        st.info("No pest-related images were found for this crop.")
                        continue

                    st.subheader("Pest Counts Found:")
                    if not pest_counts:
                        st.warning("No pests detected for this crop.")
                    else:
                        st.dataframe(pd.DataFrame(pest_counts.items(), columns=['Pest', 'Total Count']), use_container_width=True)
                    
                    if images_by_pest:
                        st.subheader("Annotated Pest Images (Verification)")
                        pest_img_tab_names = [f"{name} ({len(imgs)} images)" for name, imgs in images_by_pest.items()]
                        pest_img_tabs = st.tabs(pest_img_tab_names)
                        
                        for j, (pest_name, images) in enumerate(images_by_pest.items()):
                            with pest_img_tabs[j]:
                                num_cols = 5
                                cols = st.columns(num_cols)
                                for idx, (filename, img) in enumerate(images):
                                    with cols[idx % num_cols]:
                                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img_rgb, caption=filename, use_container_width=True)
                                        _ , buf = cv2.imencode(".png", img)
                                        data_bytes = buf.tobytes()
                                        st.download_button(
                                            label=" Download",
                                            data=data_bytes,
                                            file_name=f"annotated_{filename}.png",
                                            mime="image/png",
                                            key=f"pest_dl_{crop_name}_{pest_name}_{idx}"
                                        )

    # --- BRANCH 2: Display Crop & Disease ---
    with col2:
        st.header("🌿 Crop & Disease Analysis")
        disease_results_data = st.session_state.disease_results_by_crop

        if not all_crop_names:
            st.warning("Run analysis to see disease results.")
        else:
            disease_crop_tabs = st.tabs([f"{crop} (Diseases)" for crop in all_crop_names])
            
            for i, crop_name in enumerate(all_crop_names):
                with disease_crop_tabs[i]:
                    data = disease_results_data.get(crop_name, {})
                    
                    if not data:
                        st.info("No disease-related images were found for this crop.")
                        continue

                    (healthy_image_data, healthy_aggregated_palette) = data.get('healthy', ([], None))
                    unhealthy_groups = data.get('unhealthy_by_disease', {})
                    
                    healthy_count = len(healthy_image_data)
                    unhealthy_count = sum(len(img_data_list) for img_data_list, agg_palette in unhealthy_groups.values())
                    
                    if healthy_count == 0 and unhealthy_count == 0:
                        st.info("No disease-related images were found for this crop.")
                        continue
                    
                    health_tabs = st.tabs([f"HEALTHY ({healthy_count})", f"UNHEALTHY ({unhealthy_count})"])
                    
                    # --- HEALTHY TAB (GRID + EXPANDER) ---
                    with health_tabs[0]: 
                        if not healthy_image_data:
                            st.write("No healthy images found for this crop.")
                        else:
                            num_cols = 4
                            cols = st.columns(num_cols)
                            for idx, item_data in enumerate(healthy_image_data):
                                with cols[idx % num_cols]:
                                    st.image(item_data['original'], caption=item_data['fname'], use_container_width=True)
                                    
                                    buf = io.BytesIO()
                                    item_data['original'].save(buf, format="PNG")
                                    st.download_button(
                                        label=" Download",
                                        data=buf.getvalue(),
                                        file_name=item_data['fname'],
                                        mime="image/png",
                                        key=f"healthy_thumb_dl_{crop_name}_{idx}"
                                    )

                                    with st.expander("Additional Info"):
                                        st.image(cv2.cvtColor(item_data['bg_removed'], cv2.COLOR_BGR2RGB), 
                                                caption="Background Removed", use_container_width=True)
                                        st.image(item_data['individual_palette'], 
                                                caption="Individual Color Graph", use_container_width=True)
                                        
                                        st.markdown("---")
                                        dl_col1, dl_col2, dl_col3 = st.columns(3)
                                        buf_orig = io.BytesIO()
                                        item_data['original'].save(buf_orig, format="PNG")
                                        dl_col1.download_button(" Orig", buf_orig.getvalue(), 
                                                                file_name=item_data['fname'], mime="image/png",
                                                                key=f"h_orig_{crop_name}_{idx}")
                                        _ , buf_bg = cv2.imencode(".png", item_data['bg_removed'])
                                        dl_col2.download_button(" BG-Off", buf_bg.tobytes(), 
                                                                file_name=f"bg_removed_{item_data['fname']}", mime="image/png",
                                                                key=f"h_bg_{crop_name}_{idx}")
                                        _ , buf_graph = cv2.imencode(".png", cv2.cvtColor(item_data['individual_palette'], cv2.COLOR_RGB2BGR))
                                        dl_col3.download_button(" Graph", buf_graph.tobytes(), 
                                                                file_name=f"graph_{item_data['fname']}", mime="image/png",
                                                                key=f"h_graph_{crop_name}_{idx}")
                            
                            if healthy_aggregated_palette is not None:
                                st.markdown("---")
                                st.subheader("Aggregated Color Palette (All Healthy)")
                                st.image(healthy_aggregated_palette, use_container_width=True)
                    
                    # --- UNHEALTHY TAB (GRID + EXPANDER) ---
                    with health_tabs[1]: 
                        if not unhealthy_groups:
                            st.write("No unhealthy images found for this crop.")
                        else:
                            st.subheader("Disease Classification:")
                            disease_tab_names = [f"{disease} ({len(img_data_list)} images)" for disease, (img_data_list, agg_palette) in unhealthy_groups.items()]
                            disease_tabs = st.tabs(disease_tab_names)
                            
                            for j, (disease_name, (image_data_list, aggregated_palette)) in enumerate(unhealthy_groups.items()):
                                with disease_tabs[j]:
                                    if image_data_list:
                                        num_cols = 4
                                        cols = st.columns(num_cols)
                                        for idx, item_data in enumerate(image_data_list):
                                            with cols[idx % num_cols]:
                                                st.image(item_data['original'], caption=item_data['fname'], use_container_width=True)
                                                
                                                buf = io.BytesIO()
                                                item_data['original'].save(buf, format="PNG")
                                                st.download_button(
                                                    label=" Download",
                                                    data=buf.getvalue(),
                                                    file_name=item_data['fname'],
                                                    mime="image/png",
                                                    key=f"unhealthy_thumb_dl_{crop_name}_{disease_name}_{idx}"
                                                )

                                                with st.expander("Additional Info"):
                                                    st.image(cv2.cvtColor(item_data['bg_removed'], cv2.COLOR_BGR2RGB), 
                                                            caption="Background Removed", use_container_width=True)
                                                    st.image(item_data['individual_palette'], 
                                                            caption="Individual Color Graph", use_container_width=True)
                                                    
                                                    st.markdown("---")
                                                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                                                    buf_orig = io.BytesIO()
                                                    item_data['original'].save(buf_orig, format="PNG")
                                                    dl_col1.download_button(" Orig", buf_orig.getvalue(), 
                                                                            file_name=item_data['fname'], mime="image/png",
                                                                            key=f"unh_orig_{crop_name}_{disease_name}_{idx}")
                                                    _ , buf_bg = cv2.imencode(".png", item_data['bg_removed'])
                                                    dl_col2.download_button(" BG-Off", buf_bg.tobytes(), 
                                                                            file_name=f"bg_removed_{item_data['fname']}", mime="image/png",
                                                                            key=f"unh_bg_{crop_name}_{disease_name}_{idx}")
                                                    _ , buf_graph = cv2.imencode(".png", cv2.cvtColor(item_data['individual_palette'], cv2.COLOR_RGB2BGR))
                                                    dl_col3.download_button(" Graph", buf_graph.tobytes(), 
                                                                            file_name=f"graph_{item_data['fname']}", mime="image/png",
                                                                            key=f"unh_graph_{crop_name}_{disease_name}_{idx}")

                                    
                                    if aggregated_palette is not None:
                                        st.markdown("---")
                                        st.subheader(f"Aggregated Color Palette ({disease_name})")
                                        st.image(aggregated_palette, use_container_width=True)
                                    else:
                                        st.write(f"No images found for {disease_name}.")

# --- ETL Input Form (Unchanged) ---
if st.session_state.get('etl_inputs_ready', False):
    st.markdown("---")
    st.header("⚙️ Enter ETL Calculation Parameters")
    st.warning("Provide initial damage index (I), control cost (C), market price, and environmental factor (fev_con) for each detected pest.")
    
    pest_counts = st.session_state.total_pest_counts_for_etl
    etl_input_rows = []
    
    with st.form("etl_input_form"):
        for pest_name, n_count in pest_counts.items():
            st.subheader(f"Parameters for: {pest_name} (Total Detected Count: {n_count})")
            cols = st.columns(4)
            i_old = cols[0].number_input(f"Initial Damage Index (I) for {pest_name}", min_value=0.0, value=0.1, step=0.01, format="%.3f", key=f"i_{pest_name}")
            c_cost = cols[1].number_input(f"Control Cost (C) for {pest_name}", min_value=0.0, value=10.0, step=0.5, format="%.2f", key=f"c_{pest_name}")
            market_price = cols[2].number_input(f"Market Price/kg for {pest_name}", min_value=0.0, value=5.0, step=0.1, format="%.2f", key=f"mkt_{pest_name}")
            fev_con = cols[3].number_input(f"Environmental Factor (fev_con) for {pest_name}", min_value=0.0, value=20.0, step=0.5, format="%.1f", key=f"fev_{pest_name}")
            etl_input_rows.append((pest_name, n_count, i_old, 0, c_cost, market_price, 0, 0, fev_con))
        
        submitted = st.form_submit_button("Calculate ETL", type="primary", use_container_width=True)
        
        if submitted:
            if not etl_input_rows:
                st.error("No pest data available to calculate ETL.")
            else:
                st.info("Calculating ETL based on provided parameters...")
                df_etl, df_progress, etl_fig = run_etl_calculation(etl_input_rows)
                st.header("📊 ETL Calculation Results")
                if df_etl.empty and df_progress.empty:
                    st.warning("ETL calculation did not produce results.")
                else:
                    if not df_etl.empty:
                        st.subheader("Estimated ETL Days (±10% Range)")
                        st.dataframe(df_etl[["Pest Name", "ETL Range (Days)"]], use_container_width=True)
                    if not df_progress.empty:
                        st.subheader("Full ETL Progression Data")
                        st.dataframe(df_progress, use_container_width=True)
                    if etl_fig:
                        st.subheader("Pest Severity Progression Over Time")
                        st.plotly_chart(etl_fig, use_container_width=True)
