# COMPASS

### Xingjian Zhang, Yizhuo Wang, Guillaume Sartoretti

This repository hosts the code for paper COMPASS https://arxiv.org/abs/2507.16306, which was accepted by MRS 2025. 

*This repository is still under construction.

## Training Tips

**If you are using NSCC of Singapore to train, you could use the following commands to ease your experience.**

```
qsub train_nscc.pbs
export PBS_JOBID=*
qstat -f
ssh *
module purge
module load miniforge3
conda activate xingjian
module load cuda/12.2.2
nvidia-smi
cd scratch/win-STAMP-main/
python3 driver.py

qstat - To see all the tasks
qdel -  To delete certain task
```

### Requirements
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

### Training
1. Set appropriate parameters in `arguments.py -> Arguments`.
2. Run `python driver.py`.

### Evaluation
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
