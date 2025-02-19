import torch
print(torch.version.cuda)  # Check CUDA version used by PyTorch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be > 0 if CUDA is working
