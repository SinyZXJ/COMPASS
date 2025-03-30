# COMPASS: Cooperative Multi-Agent Persistent Surveillance using Spatio-Temporal Attention Network

This work is developed based on [Spatio-Temporal Attention Network for Persistent Monitoring of Multiple Mobile Targets](https://arxiv.org/abs/2303.06350) (accepted for presentation at IROS 2023).

## Run
## Running command under NSCC(Singapore):
qsub train_nscc.pbs
export PBS_JOBID=*
qstat -f
ssh *
module purge
module load miniforge3
conda activate <your-env-name>(xingjian)
module load cuda/12.2.2
nvidia-smi
cd scratch/<your-dir-name>(COMPASS)/
python3 driver.py

qstat - check the running tasks
qdel xxx.pbs - delete the target 

## Requirements
```bash
python >= 3.9
pytorch >= 1.11
ray >= 2.0
ortools
scikit-image
scikit-learn
scipy
imageio
tensorboard
```

## Training
1. Set appropriate parameters in `arguments.py -> Arguments`.
2. Run `python driver.py`.

## Evaluation
tensorboard --logdir=/home/users/nus/e1373512/scratch/win-STAMP-main/runs/run
1. Set appropriate parameters in `arguments.py -> ArgumentsEval`.
2. Run `python /evals/eval_driver.py`.

## Files
- `arguments.py`: Training and evaluation arguments.
- `driver.py`: Driver of training program, maintain and update the global network.
- `runner.py`: Wrapper of the local network.
- `worker.py`: Interact with environment and collect episode experience.
- `network.py`: Spatio-temporal network architecture.
- `env.py`: Persistent monitoring environment.
- `gaussian_process.py`: Gaussian processes (wrapper) for belief representation.
- `/evals/*`: Evaluation files.
- `/utils/*`: Utility files for graph, target motion, and TSP.
- `/model/*`: Trained model.
