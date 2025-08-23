FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /root/comfy

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi[standard]==0.115.4 \
    comfy-cli==1.4.1 \
    sageattention \
    huggingface_hub[hf_transfer]==0.30.0

# Install ComfyUI dependencies
RUN comfy --skip-prompt install --fast-deps --nvidia

# Install ComfyUI custom node packs
RUN comfy node registry-install comfyui-kjnodes && \
    comfy node registry-install ComfyUI-GGUF && \
    comfy node registry-install rgthree-comfy && \
    comfy node registry-install comfyui-videohelpersuite && \
    comfy node registry-install comfyui-logicutils && \
    comfy node registry-install audio-separation-nodes-comfyui

# Clone and install WanVideoWrapper
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper \
    && cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper \
    && pip install -r requirements.txt

# Create model directories
RUN mkdir -p /root/comfy/ComfyUI/models/diffusion_models/fusion \
    /root/comfy/ComfyUI/models/vae \
    /root/comfy/ComfyUI/models/text_encoders \
    /root/comfy/ComfyUI/models/clip_vision \
    /root/comfy/ComfyUI/models/loras

# Set environment variables for HuggingFace
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create cache directory for HuggingFace
RUN mkdir -p /cache

# Copy the workflow file
COPY multitalk_workflow_api.json /root/workflow_api.json

# Expose the port
EXPOSE 8000

# Set the default command
CMD ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000"] 