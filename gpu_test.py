import multiprocessing

import torch

print("CUDA dostępne:", torch.cuda.is_available())
print("Ilość GPU:", torch.cuda.device_count())
print("Nazwa GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Brak GPU")
print(multiprocessing.cpu_count())