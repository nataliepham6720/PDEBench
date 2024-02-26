#%load_ext autoreload
#%autoreload 2

import sys
import os
parentdir = os.path.dirname(os.getcwd())
sys.path.append(parentdir)

print(parentdir)

import dotenv
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv('/home/thanhngp/PDEBench/pdebench/data_gen/ns.env')
print(dotenv.load_dotenv('/home/thanhngp/PDEBench/pdebench/data_gen/ns.env'))

# images list return, save
import h5py
import matplotlib.pyplot as plt
from einops import rearrange
from phi.flow import *
import matplotlib.pyplot as plt
import numpy as np
from phi.vis import *
from src.utils import resolve_path

data_path = resolve_path('${WORKING_DIR}', idx=-1, unique=False)
print(data_path)

save_path = resolve_path('${ARTEFACT_DIR}/*/*/*/ns_incom_inhom_2d_512-0.h5', idx=-1, unique=False)
print(save_path)

data_f = h5py.File(save_path, 'r')
print(data_f.keys())

print(data_f['force'].shape) # (4, 512, 512, 1)
print(data_f['particles'].shape) # (4, 1000, 512, 512, 1)
print(data_f['velocity'].shape) # (4, 1000, 512, 512, 2)
# print(data_f['particles'][0,0:10,:,:,0])
columns = 5
fsize = 12

arr = data_f['velocity'][0, :columns, :, :, 1]

fig = plt.figure(figsize=(fsize, fsize/columns+1))
plt.imshow(rearrange(arr, 't x y -> x (t y)'))
plt.gca().set_axis_off()
plt.tight_layout(pad = 1)
plt.title('Velocity')
plt.savefig('ns_incom_velocity2.jpg')


import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

import jax
print(jax.devices())

