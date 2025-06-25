import torch
print(torch.version.cuda)           # Versión de CUDA que soporta tu instalación de PyTorch
print(torch.backends.cudnn.version())  # Verifica también cuDNN
print(torch.cuda.is_available())    # ¿CUDA está disponible?
print(torch.cuda.device_count())    # ¿Cuántas GPUs detecta?
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
