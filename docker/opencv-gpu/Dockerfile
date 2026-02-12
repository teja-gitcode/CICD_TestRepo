FROM dustynv/l4t-pytorch:r36.4.0

# Set working directory
WORKDIR /workspace

# Install GStreamer and Python bindings for CV pipeline
# Required for RTSP streaming, hardware video decode/encode, and DeepStream integration
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-rtsp \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-gi \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and wheel file
COPY docker/opencv-gpu/requirements.txt /tmp/requirements.txt
COPY wheels/pupil_apriltags-1.0.4.post11-cp310-cp310-linux_aarch64.whl /tmp/

# Install Python dependencies
RUN pip install --extra-index-url https://pypi.org/simple/ --ignore-installed -r /tmp/requirements.txt

# Clean up
RUN rm -rf /tmp/requirements.txt /tmp/*.whl

# Set Python path
ENV PYTHONPATH=/workspace

# Default command (can be overridden in docker-compose.yml)
CMD ["python3", "/workspace/app/api/server.py"]
