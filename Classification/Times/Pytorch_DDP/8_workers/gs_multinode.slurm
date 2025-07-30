#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1             # GS CHANGE 1(a)
#SBATCH --ntasks=2						# GS CHANGE 1(b)
#SBATCH --cpus-per-task=80     			# GS CHANGE 2
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=PytorchDDP
#SBATCH --account=bsc19
#SBATCH --qos=acc_bsccs
#SBATCH --error=DDP_mnode_%j.err
#SBATCH --output=DDP_mnode_%j.out

module purge
module load miniforge nccl
source activate torch-venv


# GS CHANGE 3 - the NCCL variables

export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$(( 29000 + RANDOM % 1000 ))
export PYTHONFAULTHANDLER=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# GS CHANGE 4 - changed to torchrun
srun torchrun \
	--nnodes=2 \
	--nproc_per_node=4 \
	--rdzv_id=$SLURM_JOB_ID \
	--rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
	gs_multinode.py 

