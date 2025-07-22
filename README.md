# MobileNet ONNX Runner

A lightweight, single-inference ONNX Runtime Docker image for MobileNet. Wraps a MobileNet `.onnx` model in a CLI container that takes an image path, runs inference, and prints the top-1 label (e.g. `284: Siamese cat`).

---

## 🚀 Quickstart

### Clone the repo

```bash
git clone https://github.com/enkaypeter/mobilenet-onnx-runner.git
cd mobilenet-onnx-runner/docker
```

### Build the image locally
```bash
docker build -f docker/Dockerfile -t yourname/mobilenet-onnx-runner:latest .
  ```

### Run the local image
```bash
docker run --rm \
  -v "$(pwd)/images:/app/images" \
  yourname/mobilenet-onnx-runner:latest \
  images/cat.jpg
```

You should see output like:
```bash
284: Siamese_cat
```

### Run the public Docker Hub image
We publish this image on Docker Hub as `enkaypeter/mobilenet-onnx-runner:v0.1.0`. No local build required:
```bash
docker run --rm \
  -v "$(pwd)/images:/app/images" \
  enkaypeter/mobilenet-onnx-runner:v0.1.0 \
  images/cat.jpg
```

### Additional Info

- Based on MobileNetV2, exported with ONNX opset 10 using PyTorch 1.8
- Predictions use the `imagenet_class_index.json` mapping from the [official Keras reference](https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py).