#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4             # 4 tareas (una por GPU)
#SBATCH --cpus-per-task=20     # 20 CPUs por tarea (4*20=80 CPUs en total)
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --job-name=PytorchDDP
#SBATCH --account=bsc19
#SBATCH --qos=acc_bsccs
#SBATCH --error=DDP_1_node_1_task.e
#SBATCH --output=DDP_1_node_1_task.o

module purge
module load miniforge
source activate torch-venv

export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1              # Deshabilita InfiniBand
#export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=ib0
#export MASTER_ADDR="127.0.0.1"
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$(( 29000 + RANDOM % 1000 ))
export PYTHONFAULTHANDLER=1

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

#ulimit -c unlimited

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

python3 eval_ddp.py
