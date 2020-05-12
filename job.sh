#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=cifar100
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=3

cd $SLURM_TMPDIR
cp -r ~/scratch/pytorch-cifar100 .
cd pytorch-cifar100

module load python/3.7
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


python train.py -net resnet18

