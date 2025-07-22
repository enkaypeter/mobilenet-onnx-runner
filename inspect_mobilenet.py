import onnx
import numpy as np

MODEL_PATH = "models/mobilenet.onnx"

# 1. Load the model
model = onnx.load(MODEL_PATH)

# 2. Print producer info & metadata props
print("Producer name:   ", model.producer_name)
print("Producer version:", model.producer_version)
for prop in model.metadata_props:
    print(f"  {prop.key}: {prop.value}")

# 3. Print Opset imports
print("\nOpset imports:")
for imp in model.opset_import:
    domain = imp.domain or "ai.onnx"
    print(f"  domain='{domain}', version={imp.version}")

# 4. Print first few graph nodes
print("\nFirst 5 graph nodes:")
for node in model.graph.node[:5]:
    print(" ", node.op_type, node.name)

# 5. Count total parameters
total_params = 0
for init in model.graph.initializer:
    total_params += np.prod(init.dims)
print(f"\nTotal parameters: {total_params:,}")
