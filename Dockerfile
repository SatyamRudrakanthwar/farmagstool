# ==========================================================
# 🐳 Streamlit + PyTorch GPU-ready Dockerfile
# ==========================================================

# --- Base Image: NVIDIA CUDA (for GPU acceleration) ---
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- Set working directory ---
WORKDIR /app

# --- Prevent interactive prompts during install ---
ENV DEBIAN_FRONTEND=noninteractive

# --- Install Python & system dependencies ---
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git wget curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Copy project files ---
COPY . .

# --- Install Python packages ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Streamlit configuration ---
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

# --- Run Streamlit app ---
CMD ["streamlit", "run", "app.py"]
