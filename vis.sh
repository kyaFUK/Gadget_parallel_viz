#!/bin/bash
#PJM -L "rscgrp=large"
#PJM -L "node=386"
#PJM -L "elapse=00:20:00"
#PJM --mpi "proc=386"
#PJM -S
#PJM --rsc-list "retention_state=0"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
export XOS_MMM_L_HPAGE_TYPE=none
export OMP_NUM_THREADS=48
export OMP_NESTED=1
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_STACKSIZE=128M
export OMP_SCHEDULE="static"

. /vol0004/apps/oss/spack/share/spack/setup-env.sh
spack load /gd4btvn # py-h5py@3.10
spack load /tq5kd2o # py-mpi4py@3.10
spack load /zpd2xho # hdf5+mpi@3.10
spack load /nlg2w4c # py-matplotlib@3.10


export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/vol0004/apps/oss/spack-v0.19/opt/spack/linux-rhel8-a64fx/fj-4.8.1/hdf5-1.12.2-2ildmrbqztubzpo6zhzbnfw7vsbtr7im/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/vol0004/apps/oss/spack-v0.19/opt/spack/linux-rhel8-a64fx/fj-4.10.0/fujitsu-fftw-1.1.0-duhp47eaiyq3mt7rzcz7lm5tsgc5qfxh/lib/:$LD_LIBRARY_PATH


# 実行
mpirun -n 386 python yt_mpi_IO.py
