#!/usr/bin/env python3
import sys
import json
import onnxruntime as ort
import numpy as np
from PIL import Image


def load_labels(path="/app/imagenet_class_index.json"):
    with open(path, "r") as f:
        # The JSON is of the form: { "0": ["n01440764","tench"], "1": [...], ... }
        mapping = json.load(f)
        # Convert to a simple list where idx i maps to mapping[str(i)][1]
        labels = [mapping[str(i)][1] for i in range(len(mapping))]
        return labels


def preprocess(image_path):
    # load image, resize to 224×224, normalize to [0,1], reshape to NCHW
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    
    # transpose to [N,C,H,W]
    return arr.transpose(2, 0, 1)[None, ...]


def main():
    if len(sys.argv) != 3:
        print("Usage: onnxruntime_runner.py <model.onnx> <image.jpg>")
        sys.exit(1)

    model_path, img_path = sys.argv[1], sys.argv[2]
    # 1) Load labels once
    labels = load_labels()

    # 2) Run inference
    sess = ort.InferenceSession(model_path)
    inp_name = sess.get_inputs()[0].name
    data = preprocess(img_path)
    outputs = sess.run(None, { inp_name: data })
    scores = outputs[0].flatten()

    # 3) Pick top-1
    idx = int(np.argmax(scores))
    label = labels[idx]

    # 4) Print both index and label
    print(f"{idx}: {label}")


if __name__ == "__main__":
    main()
