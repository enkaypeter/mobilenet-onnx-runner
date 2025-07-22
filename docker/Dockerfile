FROM python:3.9-slim

WORKDIR /app

# install only what we need (onnxruntime, numpy, pillow) then clean up
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-pip python3-pil \
 && pip3 install --no-cache-dir onnxruntime numpy pillow \
 && apt-get purge -y python3-pip \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Copy in mobilenet model, runner script, and imagenet labels
COPY models/mobilenet.onnx            /models/mobilenet.onnx
COPY docker/onnxruntime_runner.py            /app/onnxruntime_runner.py
COPY docker/imagenet_class_index.json        /app/imagenet_class_index.json

WORKDIR /app
ENTRYPOINT ["python3", "onnxruntime_runner.py", "/models/mobilenet.onnx"]

