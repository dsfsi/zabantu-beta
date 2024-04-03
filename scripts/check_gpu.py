"""
    Sanity check to see if Pytorch is able to detect CUDA enabled GPU(s)
"""

import sys
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if device.type == 'cuda':
    for i in range(torch.cuda.device_count()):
        print('Device:', i)
        print(torch.cuda.get_device_name(i))
        print('*' * 50)
        print('Memory Usage:')
        print('\tAllocated:', round(torch.cuda.memory_allocated(i)/1024**3,1), 'GB')
        print('\tCached:   ', round(torch.cuda.memory_reserved(i)/1024**3,1), 'GB')
        print()
    sys.exit(0)
else:
    print('No GPU detected. Try updating the CUDA bin and lib paths or reinstalling Pytorch with CUDA support.')
    sys.exit(1)