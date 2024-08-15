import os
from pathlib import Path
import sys
import jax
import numpy as np
import scipy
from jax import jit
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import sharded_fft
import time, timeit
from pario import parprint

def host_subset(size: int):
    host_id = jax.process_index()
    start = host_id * size // num_gpus
    end = (host_id + 1) * size // num_gpus
    return rng.random((size, size//num_gpus, size), dtype=np.float32)

nr = 5
nt = 1
verbose = False
size = 1024
if len(sys.argv) > 1: 
    size = int(sys.argv[1])

jax.distributed.initialize()
jax.config.update("jax_enable_x64", True)

num_gpus = jax.device_count()

global_shape = (size, size, size)
rng = np.random.default_rng(12345)
x_np = host_subset(size)

if verbose:
    parprint("\n----------------------")
    parprint("jax version:          ", jax.__version__)
    parprint("CUDA_VISIBLE_DEVICES: ", os.environ.get("CUDA_VISIBLE_DEVICES"))
    parprint("devices:              ", jax.device_count(), jax.devices())
    parprint("local_devices:        ", jax.local_device_count(), jax.local_devices())
    parprint("process_index         ", jax.process_index())
    parprint("number of GPUs:       ", num_gpus)
    parprint("size of local array:  ", x_np.nbytes / 1024 / 1024 / 1024, "GB")
    parprint("----------------------")

# warmup 
x_npi = sharded_fft.fft(x_np ,'r2c')
x_np  = sharded_fft.fft(x_npi ,'c2r')

# measure
dt_r2c = np.asarray(timeit.Timer(lambda: sharded_fft.fft(x_np ,'r2c')).repeat(repeat=nr,number=nt))
dt_c2r = np.asarray(timeit.Timer(lambda: sharded_fft.fft(x_npi,'c2r')).repeat(repeat=nr,number=nt))

parprint()
parprint(f"dt_r2c times on {num_gpus} GPUs: ",dt_r2c)
parprint(f"dt_c2r times on {num_gpus} GPUs: ",dt_c2r)
parprint()