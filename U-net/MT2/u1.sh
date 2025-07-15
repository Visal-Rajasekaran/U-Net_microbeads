#!/bin/bash
#PBS -P personal-e1357167
#PBS -l walltime=5:00:00
#PBS -j oe
#PBS -k oed
#PBS -N trial
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1


cd $PBS_O_WORKDIR;

## define singularity container to use
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

singularity exec $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

python -c "import torch; print(torch.cuda.get_device_name())"
# or python script.py
python nanopics_v5_train_seg_only.py

EOF
