#!/bin/bash
#SBATCH --job-name=client-test
#SBATCH --output=client-%j.out
#SBATCH --error=client-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00



echo "Starting ssh tunner"
#python src/client_socket.py
python src/client_socket_act.py

echo "==== Job complete ===="
