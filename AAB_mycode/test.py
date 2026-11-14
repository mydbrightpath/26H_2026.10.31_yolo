import torch
import ultralytics

print("PyTorch 版本：", torch.__version__)
print("PyTorch 对应的 CUDA 版本：", torch.version.cuda)
print("GPU 是否可用：", torch.cuda.is_available())

print(ultralytics.__version__)
print(ultralytics.__file__)