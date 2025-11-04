# --------------------------------------------------
# 🌿 AgriVision - Production Dockerfile (CPU-only, optimized)
# --------------------------------------------------

# 1️⃣ Base image: slim Python 3.10
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Install system dependencies (for OpenCV, Torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 4️⃣ Copy only requirements first (for caching)
COPY requirements.txt .

# 5️⃣ Install Python dependencies
# Install CPU-only Torch and torchvision from official PyTorch wheel index
RUN pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy the rest of your project (app code, model, etc.)
COPY . .

# 7️⃣ Expose Streamlit port
EXPOSE 8501

# 8️⃣ Environment variables (Streamlit config)
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

# 9️⃣ Default command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]