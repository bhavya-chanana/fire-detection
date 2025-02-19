import torch
import time
# Check MPS availability
print("\n=== MPS Availability ===")
print(f"Is MPS available: {torch.backends.mps.is_available()}")
print(f"Is MPS built: {torch.backends.mps.is_built()}")

# Set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}")