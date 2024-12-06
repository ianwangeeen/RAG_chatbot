from safetensors.torch import load_file
import torch

# Load weights from safetensors
weights = load_file(".\\results\\model.safetensors")

# Save as .bin
torch.save(weights, "pytorch_model.bin")