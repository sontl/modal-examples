# RunPod ComfyUI Video Upscaler Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    fastapi \
    uvicorn \
    comfy-cli==1.4.1 \
    sageattention \
    huggingface_hub[hf_transfer]==0.30.0

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV RUNPOD_DEBUG_LEVEL=INFO

# Install ComfyUI and dependencies
RUN comfy --skip-prompt install --fast-deps --nvidia

# Install custom nodes
RUN comfy node registry-install comfyui-kjnodes && \
    comfy node registry-install ComfyUI-GGUF && \
    comfy node registry-install rgthree-comfy && \
    comfy node registry-install comfyui-videohelpersuite && \
    comfy node registry-install comfyui-logicutils && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials

# Copy the workflow JSON file
COPY VideoUpscalerAPI.json /app/VideoUpscalerAPI.json

# Copy the handler script
COPY handler.py /app/handler.py

# Copy model download script
COPY download_models.py /app/download_models.py

# Download models during build (optional - can be done at runtime)
# RUN python /app/download_models.py

# Expose port for development/testing
EXPOSE 8000

# Start the RunPod serverless worker
CMD ["python", "-u", "/app/handler.py"]