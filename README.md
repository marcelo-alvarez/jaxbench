# jaxbench
Scripts for benchmarking jax performance on GPUs

## usage
```
# on an S3DF bastion node:
% git clone https://github.com/marcelo-alvarez/jaxbench.git
% cd jaxbench
% ./run-jax-test.sh

Running python fft-test.py 2048 on 4 nodes of S3DF-ada with 32 GPUs

srun: job 54160652 queued and waiting for resources
srun: job 54160652 has been allocated resources

dt_r2c times on 32 GPUs: [1.93205035 1.78596059 1.75069619 1.69788293 1.77042886]
dt_c2r times on 32 GPUs: [1.71834164 1.68575498 1.8715454  1.69725737 1.71897307]
```