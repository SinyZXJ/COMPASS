# arguments.py
import math

class Arguments:
    def __init__(self):
        self.num_agents = 3
        self.use_gpu_runner = True
        self.use_gpu_driver = True
        self.cuda_devices = [0,1] # Make sure this matches your setup
        self.episode_steps = 256
        self.num_meta = 16 # Number of parallel Ray runners
        self.num_minibatch = 16 # Number of minibatches to split buffer into for PPO updates
        self.buffer_size = int(self.num_meta * self.episode_steps)
        self.minibatch_size = int(self.buffer_size // self.num_minibatch)
        self.update_epochs = 4 # Number of PPO epochs over the collected buffer
        self.curriculum = True  # curriculum learning for environment parameters

        self.lr = 1e-4 # Learning rate
        self.lr_decay_step = 64 # Episodes after which LR decays
        self.gamma = 0.99 # Discount factor for rewards
        self.gae_lambda = 0.95 # Lambda for Generalized Advantage Estimation
        self.clip_coef = 0.2 # PPO clipping coefficient

        self.embedding_dim = 128
        self.high_info_thre = math.exp(-0.5)
        self.adaptive_kernel = False

        self.budget_size = (20, 40) # Range for budget randomization
        self.graph_size = (100, 201)   # Range for graph size randomization
        self.history_size = (50, 101)  # Range for history sequence length randomization
        self.k_size = 10  # KNN - number of neighboring nodes
        self.target_size = (6, 10) # Range for number of targets randomization
        self.history_stride = 5
        self.prior_measurement = True

        self.summary_window = 1 # Episodes to average for TensorBoard/W&B logging
        self.run_name = 'final_1' # Changed run_name for clarity
        self.model_path = f'models/{self.run_name}'
        self.train_path = f'runs/{self.run_name}'
        self.gifs_path = f'gifs/{self.run_name}'
        self.load_model = True # Set to True to load a saved model
        self.use_wandb = False
        if self.use_wandb:
            self.project_name = 'STAMP_IPPO' # Changed project name
            self.wandb_notes = 'Independent PPO with shared parameters'
            self.wandb_id = None # Set to a specific ID to resume a W&B run, or None/'' for new
        self.save_img_gap = 100
        self.save_files = True


class ArgumentsEval(Arguments):
    def __init__(self):
        super().__init__()
        self.high_info_thre = 'change in arguments' # Placeholder
        self.prior_measurement = 'change in arguments' # Placeholder

        self.run_name = 'run_ippo_shared_0607' # Match training run_name if evaluating that model
        self.model_path = f'models/{self.run_name}'
        self.result_path = self.run_name
        self.cuda_devices = [0,1]
        self.num_meta = 2
        self.num_eval = 10

        self.budget_size = 30
        self.graph_size = 200
        self.history_size = 100
        self.k_size = 10
        self.target_size = 6
        self.target_speed = 1/20.0 # Ensure float division
        self.history_stride = 5

        self.save_results = True
        self.save_img_gap = 20


arg = Arguments()
arg_eval = ArgumentsEval()