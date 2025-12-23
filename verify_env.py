import sys

import numpy as np
import torch

print(f"Python executable: {sys.executable}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Basic NumPy check
arr = np.array([1, 2, 3])
print(f"NumPy array: {arr}")

# Basic PyTorch check
tensor = torch.tensor([1, 2, 3])
print(f"PyTorch tensor: {tensor}")

# Check interaction/conversion
tensor_from_numpy = torch.from_numpy(arr)
print(f"Tensor from NumPy: {tensor_from_numpy}")

if torch.equal(tensor, tensor_from_numpy):
    print("SUCCESS: PyTorch and NumPy are working correctly together.")
else:
    print("FAILURE: Tensor values do not match.")

# Einops check
import einops

print(f"Einops version: {einops.__version__}")
x = torch.randn(2, 3, 4)
y = einops.rearrange(x, "b c h -> b (c h)")
print(f"Einops rearrange successful. Output shape: {y.shape}")

# Plotly check
import plotly
import plotly.graph_objects as go

print(f"Plotly version: {plotly.__version__}")
fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
print("Plotly figure created successfully.")
