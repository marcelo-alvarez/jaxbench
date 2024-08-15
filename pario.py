import jax, sys
from jax.experimental.multihost_utils import sync_global_devices

def parprint(*args,**kwargs):
    pid = jax.process_index()
    if pid == 0:
        print("".join(map(str,args)),**kwargs);  sys.stdout.flush()
    sync_global_devices('parprint')
