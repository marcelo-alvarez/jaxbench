import os
from pathlib import Path
import sys
import jax
import numpy as np
import jax.numpy as jnp
import scipy
from jax import jit
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import time, timeit
from pario import parprint

nr = 5
nt = 3
verbose = False
docpu = False
size2 = 1024
if len(sys.argv) > 1: 
    size2 = int(sys.argv[1])
size = size2**2

jax.config.update("jax_enable_x64", True)

num_gpus = jax.device_count()

global_shape = (size, size, size)
rng = np.random.default_rng(12345)
x_np_gpu_32 = jnp.asarray(rng.random(size, dtype=np.float32))
x_np_gpu_64 = jnp.asarray(rng.random(size, dtype=np.float64))
if docpu:
    x_np_cpu_32 = rng.random(size, dtype=np.float32)
    x_np_cpu_64 = rng.random(size, dtype=np.float64)

if verbose:
    parprint("\n----------------------")
    parprint("jax version:          ", jax.__version__)
    parprint("CUDA_VISIBLE_DEVICES: ", os.environ.get("CUDA_VISIBLE_DEVICES"))
    parprint("devices:              ", jax.device_count(), jax.devices())
    parprint("local_devices:        ", jax.local_device_count(), jax.local_devices())
    parprint("process_index         ", jax.process_index())
    parprint("number of GPUs:       ", num_gpus)
    parprint("----------------------")

polytest = lambda x: 2*x - 3*x**2 + 5*x**3 - 7*x**4

# warmup
y_np_gpu_32 = polytest(x_np_gpu_32)
y_np_gpu_64 = polytest(x_np_gpu_64)

# measure
dt_gpu_32 = np.asarray(timeit.Timer(lambda: polytest(x_np_gpu_32)).repeat(repeat=nr,number=nt)).mean()
dt_gpu_64 = np.asarray(timeit.Timer(lambda: polytest(x_np_gpu_64)).repeat(repeat=nr,number=nt)).mean()
if docpu:
    dt_cpu_32 = np.asarray(timeit.Timer(lambda: polytest(x_np_cpu_32)).repeat(repeat=nr,number=nt)).mean()
    dt_cpu_64 = np.asarray(timeit.Timer(lambda: polytest(x_np_cpu_64)).repeat(repeat=nr,number=nt)).mean()

parprint()
if docpu:
    parprint(f"32-bit CPU float time: {dt_cpu_32:.2f}")
    parprint(f"64-bit CPU float time: {dt_cpu_64:.2f}")
parprint(f"32-bit GPU float time: {dt_gpu_32:.2e}")
parprint(f"64-bit GPU float time: {dt_gpu_64:.2e}")
parprint(f"ratio: {dt_gpu_64/dt_gpu_32:.2f}")
parprint()