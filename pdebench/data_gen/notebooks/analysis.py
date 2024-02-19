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

data_path = resolve_path('${WORKING_DIR}/*/*/*/*.h5', idx=-1, unique=False)
print(data_path)

save_path = resolve_path('${ARTEFACT_DIR}', idx=-1, unique=False)
print(save_path)

data_f = h5py.File(data_path, 'r')
data_f.keys()


import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

import jax
print(jax.devices())

