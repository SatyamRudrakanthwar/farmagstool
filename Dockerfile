# ==========================================================
# 🐳 Streamlit + CPU-optimized Dockerfile
# ==========================================================

# --- Base Image: Standard Python 3.11 Slim (Lighter and CPU-friendly) ---
FROM python:3.11-slim

# --- Set working directory ---
WORKDIR /app

# --- Prevent interactive prompts during install ---
ENV DEBIAN_FRONTEND=noninteractive

# --- Install essential system dependencies ---
# We need Git, OpenCV dependencies (libgl1), and general tools.
RUN apt-get update && apt-get install -y \
    python3-venv \
    git \
    wget \
    curl \
    # OpenCV dependencies for running 'opencv-python-headless'
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Copy project files ---
COPY . .

# --- Install Python packages ---
# Use the CPU index URL for PyTorch and associated packages
# NOTE: This assumes your requirements.txt still contains PyTorch/Transformers 
# needed for DINOv2 and local health classifier.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# --- Streamlit configuration ---
# EXPOSE 8501 is implicitly handled by the CMD if --server.port is used.
# Added EXPOSE for clarity.
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
# IMPORTANT: EC2 firewall/security group must allow port 8501

# --- Run Streamlit app ---
# Use CMD or ENTRYPOINT to specify the default command. 
# CMD ["streamlit", "run", "app.py"] 
# Using the full command with address=0.0.0.0 for external access:
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
