#!/bin/bash
#SBATCH --job-name=sor_mpi_112          # Имя задачи
#SBATCH --output=sor_mpi_112_%j.out     # Файл вывода
#SBATCH --partition=tornado-k40         # Нужный partition
#SBATCH --nodes=4                       # 4 узла
#SBATCH --ntasks=112                    # 112 MPI-процессов всего
#SBATCH --cpus-per-task=1               # 1 ядро на MPI-процесс
#SBATCH --time=00:30:00                 # Лимит времени

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo ""

module load mpi/openmpi/4.1.6/gcc/11

make

# Размер сетки
GRID=1001

# Запуск: число процессов из SLURM_NTASKS, exe из Makefile
mpirun -np ${SLURM_NTASKS} ./sor_mpi_adaptive_omega ${GRID}

