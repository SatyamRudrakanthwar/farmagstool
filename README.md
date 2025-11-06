# AGS Internal Tool V1.0 🌿🐜📊

## Description

This Streamlit application provides an internal tool for analyzing batches of agricultural images. It performs two main analyses in parallel:
1.  **Pest Detection & ETL Prediction:** Detects pests using a YOLOv5 model, aggregates counts across the batch, and runs an Economic Threshold Level (ETL) prediction based on user-provided parameters.
2.  **Crop & Disease Sorting:** Classifies images into one of three crop types using DINOv2 clustering, then sorts each crop group into "Healthy" or "Unhealthy" using CLIP. Finally, it further classifies the "Unhealthy" images into three disease categories using DINOv2 clustering.

---

## Features ✨

* **Batch Image Processing:** Analyzes multiple images uploaded simultaneously.
* **Dual-Branch Analysis:** Runs pest/ETL and crop/disease pipelines concurrently.
* **Pest Detection:** Uses YOLOv5 to identify and count pests.
* **ETL Prediction:** Calculates potential economic impact based on pest counts and user inputs.
* **Crop Classification:** Unsupervised clustering (DINOv2 + K-Means) into 3 crop types.
* **Health Classification:** Uses CLIP with descriptive prompts to sort images as Healthy/Unhealthy.
* **Disease Classification:** Unsupervised clustering (DINOv2 + K-Means) to categorize unhealthy images.
* **Interactive GUI:** Built with Streamlit for easy image upload and results visualization.

---

## Workflow Overview  Lược đồ

1.  **Upload:** User uploads a batch of images (.jpg, .png, .jpeg).
2.  **Analysis Trigger:** User clicks the "Run Full Analysis" button.
3.  **Branch 1 (Pest/ETL):**
    * All images are fed into the YOLOv5 model.
    * Total pest counts are aggregated.
    * Pest counts are displayed.
    * User enters ETL parameters (I, C, Market Price, Fev Con) for detected pests.
    * ETL model runs and displays results (dataframes, graph).
4.  **Branch 2 (Crop/Disease):**
    * All images are fed into the DINOv2 Crop Classifier (K=3).
    * Images are grouped into Crop 1, Crop 2, Crop 3.
    * Each crop group is fed into the CLIP Health Classifier.
    * Images are sorted into Healthy/Unhealthy subgroups for each crop.
    * Unhealthy images for each crop are fed into the DINOv2 Disease Classifier (K=3).
    * Images are sorted into Disease A, B, C subgroups for each unhealthy crop.
5.  **Display:** Results from both branches are displayed side-by-side.

---

## Setup & Installation ⚙️

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-folder>
    ```
2.  **Set up Python Environment:** (Recommended: Use a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install PyTorch:** Follow the official instructions for your system/CUDA version at [pytorch.org](https://pytorch.org/).
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Download Models:**
    * Place your trained YOLOv5 model as `models/best.pt`.
    * Run the script to download DINOv2:
        ```bash
        python download_dino.py
        ```
        (This saves to `models/dinov2_small/`)
    * Run a script (or manually) to download the CLIP model (`ViT-B-32.pt`) and place it in the `models/` folder. (Refer to previous instructions).
    * Ensure all other necessary model files (e.g., `.pkl` files for health/disease if not using DINO/CLIP) are placed in the `models/` directory. **Update the paths in `pipeline/*.py` if needed.**

---

## Usage ▶️

1.  Activate your virtual environment.
2.  Run the Streamlit app from the project root directory:
    ```bash
    streamlit run app.py
    ```
3.  Open the URL provided by Streamlit in your web browser.
4.  Upload your batch of images using the file uploader.
5.  Click the "Run Full Analysis" button.
6.  Wait for the pest counts to appear.
7.  Fill in the required ETL parameters in the form and click "Calculate ETL".
8.  View the results displayed in the two main columns.

---

## Project Structure 📁