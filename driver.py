# driver.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import wandb # Weights & Biases
from torch.amp import GradScaler # Keep amp for GradScaler
from torch import amp # For autocast
from network import AttentionNet 
from runner import Runner # Assuming runner.py is compatible or not used for IPPO specific parts
from arguments import arg
from tqdm import tqdm

class Logger:
    def __init__(self):
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        self.cuda_devices_str = str(arg.cuda_devices)[1:-1] # For os.environ
        self.writer = SummaryWriter(arg.train_path) if arg.save_files else None

        # Buffer keys description for IPPO with shared parameters:
        self.episode_buffer_keys = [
            'history_pooled', # Pooled history features for network
            'edge',           # Edge connections for network
            'dist',           # Distance features for network
            'dt_pooled',      # Pooled dt features for network
            'nodeidx',        # Current node index of each agent: [NumAgents, 1]
            'logp',           # Log prob of taken action for each agent: [NumAgents, 1]
            'action',         # Action taken by each agent: [NumAgents, 1] (index for k-neighbors)
            'value',          # Per-agent value estimate V(s_i): [NumAgents]
            'temporalmask_pooled', # Mask for pooled temporal features (scalar: valid pooled length)
            'spatiomask',     # Spatial mask for pointer (usually [NumNodes, K_size])
            'spatiope',       # Spatio-positional encoding for nodes [NumNodes, PosEncDim]
            'done',           # Episode done flag (scalar boolean)
            'reward',         # Team reward (scalar, used for each agent's GAE)
            'advantage',      # Per-agent advantage A_i: [NumAgents]
            'return'          # Per-agent target return G_i: [NumAgents]
        ]
        self.metric_names = [ # From original, ensure worker provides these
            'avgnvisit', 'stdnvisit', 'avggapvisit', 'stdgapvisit',
            'avgrmse', 'avgunc', 'avgjsd', 'avgkld',
            'stdunc', 'stdjsd', 'covtr', 'f1', 'mi', 'js', 'rmse', 'scalex', 'scalet'
        ]
        np.random.seed(arg.seed if hasattr(arg, 'seed') else 0) # Use a seed from args if available
        print(f'=== COMPASS (IPPO with Shared Parameters) ===\n'
              f'Run Name: {arg.run_name}\n'
              f'Num Agents: {arg.num_agents}\n'
              f'Minibatch size: {arg.minibatch_size}, Buffer size: {arg.buffer_size}')
        if self.cuda_devices_str:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_devices_str
            if torch.cuda.is_available():
                 print(f'Using CUDA devices: {self.cuda_devices_str} ({torch.cuda.get_device_name() if torch.cuda.device_count() > 0 else "N/A"})')
            else:
                 print(f'CUDA specified ({self.cuda_devices_str}) but not available. Using CPU.')

        if not ray.is_initialized():
            ray.init(num_cpus=arg.num_meta + 1) # +1 for driver, adjust as needed

        if arg.use_wandb:
            wandb.init(project=arg.project_name, 
                       name=arg.run_name, 
                       entity=getattr(arg, 'wandb_entity', None), # Optional entity
                       config=vars(arg),
                       notes=arg.wandb_notes, 
                       resume='allow', 
                       id=arg.wandb_id if arg.wandb_id else None) # Pass None if empty for new run
        if arg.save_files:
            os.makedirs(arg.model_path, exist_ok=True)
            os.makedirs(arg.gifs_path, exist_ok=True)

    def set(self, net, optimizer, lr_scheduler):
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def write_to_board(self, collected_training_data, curr_episode_count):
        if not collected_training_data: return
        
        data_np = np.array(collected_training_data)
        # Mean over the summary window (multiple batches of experience)
        if np.isnan(data_np).any():
            avg_data = list(np.nanmean(data_np, axis=0))
        else:
            avg_data = list(np.mean(data_np, axis=0))
        
        avg_data = [0 if np.isnan(x) else x for x in avg_data] # Replace any remaining NaNs with 0 for logging

        # PPO training stats (these are already averages over minibatches/epochs from one buffer update)
        # Here we average them over `summary_window` number of buffer updates.
        mean_reward, mean_value_est, mean_p_loss, mean_v_loss, mean_entropy, \
        last_grad_norm, mean_return_val, mean_clip_frac, mean_approx_kl = avg_data[:9]
        
        perf_metrics_avg = avg_data[9:]

        log_dict = {
            'Loss/Learning_Rate': self.lr_scheduler.get_last_lr()[0],
            'PPO/Value_Estimate_Mean': mean_value_est, # Avg of per-agent value estimates
            'PPO/Policy_Loss_Mean': mean_p_loss,
            'PPO/Value_Loss_Mean': mean_v_loss,
            'PPO/Entropy_Mean': mean_entropy,
            'PPO/Grad_Norm_Last': last_grad_norm, # Grad norm from the last update of the last buffer
            'PPO/Clip_Fraction_Mean': mean_clip_frac,
            'PPO/Approx_KL_Mean': mean_approx_kl,
            'Episode/Mean_Reward_Collected': mean_reward, # Avg of scalar team rewards collected
            'Episode/Mean_Return_GAE': mean_return_val,    # Avg of per-agent GAE returns
        }
        for i, name in enumerate(self.metric_names):
            if i < len(perf_metrics_avg):
                log_dict[f'Perf/{name}'] = perf_metrics_avg[i]
            else:
                log_dict[f'Perf/{name}'] = np.nan

        if self.writer:
            for k, v_val in log_dict.items():
                if not np.isnan(v_val):
                    self.writer.add_scalar(tag=k, scalar_value=v_val, global_step=curr_episode_count)
        if arg.use_wandb:
            valid_log_dict = {k: v_val for k, v_val in log_dict.items() if not np.isnan(v_val)}
            wandb.log(valid_log_dict, step=curr_episode_count)

    def load_saved_model(self):
        model_file = os.path.join(arg.model_path, 'checkpoint.pth')
        if not os.path.exists(model_file):
            print(f"No checkpoint found at {model_file}. Starting from scratch.")
            return 0
        
        print(f'Loading model: {arg.run_name} from {model_file}')
        try:
            # Ensure loading to the correct device, especially if model was saved on GPU and loading on CPU
            map_location = torch.device('cpu') if not torch.cuda.is_available() else None
            checkpoint = torch.load(model_file, map_location=map_location, weights_only=True)
            
            self.net.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_decay'])
            curr_episode = checkpoint.get('episode', 0) # Use .get for backward compatibility
            print(f"Resuming from episode: {curr_episode}")
            print(f'Optimizer LR: {self.optimizer.param_groups[0]["lr"]}')
            return curr_episode
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            return 0

    def save_model(self, curr_episode_count):
        print(f'Saving model at episode {curr_episode_count}')
        checkpoint = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode": curr_episode_count,
            "lr_decay": self.lr_scheduler.state_dict()
        }
        path_checkpoint = os.path.join(arg.model_path, "checkpoint.pth")
        try:
            torch.save(checkpoint, path_checkpoint)
            print(f"Model saved to {path_checkpoint}")
        except Exception as e:
            print(f"Error saving model: {e}")


def main():
    logger = Logger()
    # Determine device for the driver (global network)
    if arg.use_gpu_driver and torch.cuda.is_available():
        driver_device = torch.device('cuda')
    else:
        driver_device = torch.device('cpu')
    
    # Determine device for Ray runners (local network copies)
    # This is handled within the Runner class init based on arg.use_gpu_runner
    print(f"Driver using device: {driver_device}")

    global_network = AttentionNet(arg.embedding_dim).to(driver_device)
    global_optimizer = optim.Adam(global_network.parameters(), lr=arg.lr, eps=1e-5) # Added eps for Adam stability
    lr_scheduler = optim.lr_scheduler.StepLR(global_optimizer, step_size=arg.lr_decay_step, gamma=0.96)
    logger.set(global_network, global_optimizer, lr_scheduler)

    current_episode_count = 0 # Total episodes processed by runners
    if arg.load_model:
        current_episode_count = logger.load_saved_model()

    # DataParallel for multi-GPU training if multiple driver GPUs are specified and available
    # However, PPO typically updates the global_network on a single device or uses DDP.
    # For simplicity with Ray, global_network updates on driver_device.
    # dp_global_network = nn.DataParallel(global_network) if torch.cuda.device_count() > 1 and arg.use_gpu_driver else global_network
    # Let's use global_network directly, DataParallel adds complexity with state_dict saving/loading.
    # If you have multiple GPUs for the driver itself, DDP is preferred over DataParallel.
    # For now, assuming driver operates on a single specified CUDA device or CPU.

    meta_runners = [Runner.remote(i) for i in range(arg.num_meta)]
    if arg.use_wandb:
        wandb.watch(global_network, log_freq=arg.buffer_size // arg.minibatch_size * arg.update_epochs * 5, log_graph=True) # Log graph less frequently

    training_data_for_logging = [] # Accumulates stats over `summary_window` buffer updates
    
    # GradScaler for mixed precision training
    scaler = GradScaler('cuda')


    try:
        while current_episode_count < getattr(arg, 'max_episodes', 300000): # Add a max_episodes to args
            # --- Synchronize runner weights ---
            global_network.to(driver_device) # Ensure it's on driver_device before state_dict
            weights_to_send = global_network.state_dict()
            # If runner device is different, weights will be moved by Ray or in Runner.
            # For IPPO, local_net in Runner will be used for rollouts.
            weights_id = ray.put(weights_to_send)

            # --- Launch parallel rollouts ---
            meta_jobs = []
            # Randomize env parameters for this batch of rollouts
            # These are passed to the worker via the runner's job method
            current_budget_size = np.random.uniform(*arg.budget_size)
            current_graph_size = np.random.randint(*arg.graph_size)
            current_history_size = np.random.randint(*arg.history_size) # This is unpooled size for env
            current_target_size = np.random.randint(*arg.target_size)

            for i in range(arg.num_meta):
                meta_jobs.append(
                    meta_runners[i].job.remote(
                        weights_id, 
                        current_episode_count + i, # Unique episode ID for logging/saving images
                        current_budget_size,
                        current_graph_size,
                        current_history_size,
                        current_target_size
                    )
                )
            
            # --- Collect experiences ---
            buffer = {key: [] for key in logger.episode_buffer_keys}
            # For averaging performance metrics from this batch of rollouts
            batch_perf_metrics = {name: [] for name in logger.metric_names} 
            
            done_job_ids, meta_jobs_remaining = ray.wait(meta_jobs, num_returns=arg.num_meta, timeout=300.0) # Add timeout
            
            if meta_jobs_remaining:
                print(f"Warning: {len(meta_jobs_remaining)} runners timed out. Proceeding with collected data.")
                # Optionally, kill timed-out runners:
                # for job_ref in meta_jobs_remaining: ray.cancel(job_ref, force=True)

            collected_job_results = ray.get(done_job_ids)

            valid_rollouts = 0
            for job_result in collected_job_results:
                if job_result is None: continue # Runner might have failed
                episode_data_from_worker, metrics_from_worker = job_result
                
                # Basic check for valid data
                if not episode_data_from_worker or not episode_data_from_worker.get('done'):
                    print(f"Warning: Worker {valid_rollouts} returned empty or invalid data.") # Use worker ID if available
                    continue
                if len(episode_data_from_worker['done']) == 0:
                    print(f"Warning: Worker {valid_rollouts} rollout had zero steps.")
                    continue

                valid_rollouts += 1
                for key in logger.episode_buffer_keys:
                    buffer[key].extend(episode_data_from_worker[key])
                for name in logger.metric_names:
                    if name in metrics_from_worker and not np.isnan(metrics_from_worker[name]):
                        batch_perf_metrics[name].append(metrics_from_worker[name])
            
            if valid_rollouts == 0:
                print("No valid rollouts collected in this batch. Skipping PPO update.")
                if current_episode_count % (arg.num_meta * 10) == 0: # Save periodically even if no training
                    if arg.save_files: logger.save_model(current_episode_count)
                continue

            actual_buffer_timesteps = len(buffer['done'])
            if actual_buffer_timesteps == 0:
                print("Buffer is empty after aggregation. Skipping PPO update.")
                continue
            
            # current_episode_count += len(collected_job_results) # Increment by actual completed episodes
            current_episode_count += valid_rollouts
            print(f"DEBUG: current_episode_count for save check: {current_episode_count}") # 调试打印
            print(f"DEBUG: arg.save_files is: {arg.save_files}") # 调试打印
            print(f"Collected {actual_buffer_timesteps} timesteps from {valid_rollouts} rollouts.")

            # --- Prepare Tensors for PPO Update ---
            # Shapes assuming per-agent values/advantages
            # Note: history_pooled, edge, dist, dt_pooled, nodeidx, spatiope, spatiomask are per-step global/agent-specific inputs
            #       logp, action, value, advantage, return are per-step per-agent outputs/calculations
            #       reward, done are per-step team/global outputs
            try:
                b_hist_pool = torch.stack(buffer['history_pooled']).to(driver_device) # [T, PooledH, N, Feat]
                b_edge      = torch.stack(buffer['edge']).to(driver_device)           # [T, N, K]
                b_dist      = torch.stack(buffer['dist']).to(driver_device)           # [T, N, 1]
                b_dt_pool   = torch.stack(buffer['dt_pooled']).to(driver_device)      # [T, PooledH, 1]
                b_nodeidx   = torch.stack(buffer['nodeidx']).to(driver_device)       # [T, NumAgents, 1]
                b_spatiope  = torch.stack(buffer['spatiope']).to(driver_device)     # [T, N, PosEncDim]
                b_tmask_pool= torch.stack(buffer['temporalmask_pooled']).to(driver_device) # [T] or [T,1]
                b_smask     = torch.stack(buffer['spatiomask']).to(driver_device)     # [T, N, K]

                b_logp_old  = torch.stack(buffer['logp']).to(driver_device)          # [T, NumAgents, 1]
                b_action    = torch.stack(buffer['action']).to(driver_device)        # [T, NumAgents, 1]
                b_value_old = torch.stack(buffer['value']).to(driver_device)        # [T, NumAgents]
                b_return    = torch.stack(buffer['return']).to(driver_device)        # [T, NumAgents]
                b_advantage = torch.stack(buffer['advantage']).to(driver_device)    # [T, NumAgents]
                
                # Team reward collected at each step (used for GAE of each agent)
                b_reward_collected = torch.stack(buffer['reward']).to(driver_device) # [T]

                # Normalize advantages (per-agent advantages are normalized together)
                b_advantage = (b_advantage - b_advantage.mean()) / (b_advantage.std() + 1e-8)

            except RuntimeError as e_stack:
                print(f"Error stacking buffer to tensors: {e_stack}. Buffer size: {actual_buffer_timesteps}")
                # for k,v in buffer.items(): print(f"Key {k}, len {len(v)}, type {type(v[0]) if v else 'empty'}, shape {v[0].shape if v and hasattr(v[0],'shape') else 'N/A'}")
                continue

            # --- PPO Update Loop ---
            ppo_epoch_stats = {'p_loss': [], 'v_loss': [], 'entropy': [], 'clip_frac': [], 'approx_kl': []}
            
            # Ensure global_network is on the correct device for updates
            global_network.train() # Set to train mode
            global_network.to(driver_device)

            for _ in range(arg.update_epochs):
                indices = np.arange(actual_buffer_timesteps)
                np.random.shuffle(indices)
                for start in range(0, actual_buffer_timesteps, arg.minibatch_size):
                    end = start + arg.minibatch_size
                    mb_indices = indices[start:end]

                    mb_hist   = b_hist_pool[mb_indices]
                    mb_edge   = b_edge[mb_indices]
                    mb_dist   = b_dist[mb_indices]
                    mb_dt     = b_dt_pool[mb_indices]
                    mb_nodeidx= b_nodeidx[mb_indices]
                    mb_spiope = b_spatiope[mb_indices]
                    mb_tmask  = b_tmask_pool[mb_indices]
                    mb_smask  = b_smask[mb_indices]
                    
                    mb_logp_old = b_logp_old[mb_indices]     # [MB, NumAgents, 1]
                    mb_action   = b_action[mb_indices]       # [MB, NumAgents, 1]
                    # mb_value_old= b_value_old[mb_indices] # Not directly used in loss like this for new values
                    mb_return   = b_return[mb_indices]       # [MB, NumAgents]
                    mb_advantage= b_advantage[mb_indices]   # [MB, NumAgents]

                    with amp.autocast(device_type=driver_device.type, enabled=(driver_device.type == 'cuda')):
                        # logp_list_new: [MB, NumAgents, K_size]
                        # values_new:    [MB, NumAgents] (per-agent values)
                        logp_list_new, values_new = global_network( 
                            mb_hist, mb_edge, mb_dist, mb_dt, mb_nodeidx,
                            mb_spiope, mb_tmask, mb_smask
                        )

                        # --- Policy Loss (for each agent, then averaged) ---
                        # Gather logp of taken actions: mb_action is [MB, NumAgents, 1]
                        action_indices_for_gather = mb_action.squeeze(-1).long() # [MB, NumAgents]
                        # Clamp actions to be within valid range of K_size dimension
                        clamped_action_indices = torch.clamp(action_indices_for_gather, 0, logp_list_new.size(-1) - 1)
                        
                        # logp_new_taken needs to be [MB, NumAgents]
                        logp_new_taken = torch.gather(
                            logp_list_new, 
                            dim=2, # Gather along K_size dimension
                            index=clamped_action_indices.unsqueeze(-1) # Index must be [MB, NumAgents, 1]
                        ).squeeze(-1) # Result [MB, NumAgents]

                        logratio = logp_new_taken - mb_logp_old.squeeze(-1) # [MB, NumAgents]
                        ratio = logratio.exp()                          # [MB, NumAgents]

                        # mb_advantage is already [MB, NumAgents]
                        surr1 = ratio * mb_advantage
                        surr2 = torch.clamp(ratio, 1.0 - arg.clip_coef, 1.0 + arg.clip_coef) * mb_advantage
                        # Average over all agents and all samples in minibatch
                        policy_loss = -torch.min(surr1, surr2).mean() 

                        # --- Value Loss (for each agent, then averaged) ---
                        # values_new is [MB, NumAgents], mb_return is [MB, NumAgents]
                        values_new_clipped = b_value_old[mb_indices] + torch.clamp(
                            values_new - b_value_old[mb_indices],
                            -arg.clip_coef, # 使用与策略裁剪相同的系数
                            arg.clip_coef,
                        )
                        v_loss_unclipped = (values_new - mb_return) ** 2
                        v_loss_clipped = (values_new_clipped - mb_return) ** 2
                        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                        # --- Entropy Bonus (averaged over agents and samples) ---
                        # logp_list_new is log_softmax, so p_new = exp(logp_list_new)
                        p_new = torch.exp(logp_list_new)
                        entropy = -(p_new * logp_list_new).sum(dim=-1) # Entropy per agent: [MB, NumAgents]
                        mean_entropy = entropy.mean() # Mean over all agents and samples

                        loss = policy_loss + 0.5 * value_loss - 0.01 * mean_entropy
                    
                    global_optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(global_optimizer) # Before clipping
                    grad_norm = nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=0.5) # Common max_norm
                    scaler.step(global_optimizer)
                    scaler.update()

                    # Log stats for this minibatch update
                    ppo_epoch_stats['p_loss'].append(policy_loss.item())
                    ppo_epoch_stats['v_loss'].append(value_loss.item())
                    ppo_epoch_stats['entropy'].append(mean_entropy.item())
                    with torch.no_grad():
                        ppo_epoch_stats['clip_frac'].append(((ratio - 1.0).abs() > arg.clip_coef).float().mean().item())
                        ppo_epoch_stats['approx_kl'].append(((ratio - 1) - logratio).mean().item()) # (ratio-1) - log(ratio)
            
            lr_scheduler.step() # Step LR after all PPO epochs for this buffer

            # --- Log data for this completed buffer update cycle ---
            # Averaged collected rewards (b_reward_collected is [T])
            avg_collected_reward = b_reward_collected.mean().item()
            # Averaged GAE returns (b_return is [T, NumAgents])
            avg_gae_return = b_return.mean().item() # Mean over all agents and timesteps
            # Averaged old value estimates (b_value_old is [T, NumAgents])
            avg_old_value_est = b_value_old.mean().item()

            current_ppo_stats = [
                avg_collected_reward, avg_old_value_est,
                np.mean(ppo_epoch_stats['p_loss']), np.mean(ppo_epoch_stats['v_loss']),
                np.mean(ppo_epoch_stats['entropy']), grad_norm.item(), # Use last grad_norm
                avg_gae_return, np.mean(ppo_epoch_stats['clip_frac']),
                np.mean(ppo_epoch_stats['approx_kl'])
            ]
            
            # Averaged performance metrics from this batch of rollouts
            perf_data_for_log = [np.nanmean(batch_perf_metrics[name]) if batch_perf_metrics[name] else np.nan 
                                 for name in logger.metric_names]
            
            log_line_this_update = current_ppo_stats + perf_data_for_log
            if not np.isnan(log_line_this_update).all():
                 training_data_for_logging.append(log_line_this_update)

            if len(training_data_for_logging) >= arg.summary_window:
                logger.write_to_board(training_data_for_logging, current_episode_count)
                training_data_for_logging = []

            print("before save, saving param is:", arg.save_files)
            print("current episode count:", current_episode_count, arg.num_meta * 10, current_episode_count % (arg.num_meta * 10))
            if current_episode_count % (arg.num_meta * 10) == 0 : # Save model less frequently initially
                if arg.save_files: logger.save_model(current_episode_count)
                # More frequent saves later if needed, e.g. based on total timesteps
                # For example: if (current_episode_count * arg.episode_steps) % 1_000_000 < (arg.num_meta * arg.episode_steps):
                #    logger.save_model(current_episode_count)


    except KeyboardInterrupt:
        print("User interrupted training.")
    finally:
        print("Shutting down Ray and saving final model...")
        if arg.save_files: # Save final model
            logger.save_model(current_episode_count)
        if arg.use_wandb and wandb.run is not None:
            wandb.finish()
        if ray.is_initialized():
            ray.shutdown()
        print("Exiting.")


if __name__ == "__main__":
    # Example: Add a seed to arguments if not present
    if not hasattr(arg, 'seed'):
        arg.seed = 42 
    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arg.seed)
    main()