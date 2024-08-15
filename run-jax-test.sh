#!/bin/bash

if [ ${HOSTNAME:0:3} == "sdf" ] ; then
    envbase=/sdf/group/kipac/users/malvarez/env/xgsmenv/20240813-0.0.0
    qospar="-p ada"
    n=32; N=4; gpn=8
    system="S3DF-ada"
elif [ ${HOSTNAME:0:5} == "login" ] ; then
    envbase=/global/cfs/cdirs/mp107d/exgal/env/xgsmenv/20240813-cuda12
    qospar="-q regular"
    n=32; N=8; gpn=4
    system="NERSC-perlmutter"
fi

runtest () {
    N=$1; n=$2; gpn=$3; com="$4"
    printf "\nRunning $com on $N nodes of $system with $n GPUs\n\n"
    srun $qospar -N $N --gpus $n --gpus-per-node $gpn --tasks-per-node $gpn --time=01:00:00 \
        $com
}

module use $envbase/modulefiles
module load xgsmenv
source $envbase/conda/bin/activate malvarez

com="python fft-test.py 2048"
runtest $N $n $gpn "$com"