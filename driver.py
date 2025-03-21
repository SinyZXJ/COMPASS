import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import wandb
from torch.cuda.amp.grad_scaler import GradScaler
from torch import amp

from network import AttentionNet
from runner import Runner
from arguments import arg
from tqdm import tqdm

class Logger:
    def __init__(self):
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        self.cuda_devices = str(arg.cuda_devices)[1:-1]
        self.writer = SummaryWriter(arg.train_path) if arg.save_files else None

        self.episode_buffer_keys = [
            'history', 'edge', 'dist', 'dt',
            'nodeidx',      # shape: [batch_size, n_agents, 1]
            'logp',         # shape: [batch_size, n_agents, 1]
            'action',       # shape: [batch_size, n_agents, 1]
            'value',        # shape: [batch_size, n_agents]
            'temporalmask', 'spatiomask', 'spatiope',
            'done', 'reward', 'advantage', 'return'
        ]
        # metric_names 可以包含你需要统计的指标
        self.metric_names = [
            'avgnvisit', 'stdnvisit', 'avggapvisit', 'stdgapvisit',
            'avgrmse', 'avgunc', 'avgjsd', 'avgkld',
            'stdunc', 'stdjsd', 'covtr', 'f1', 'mi', 'js', 'rmse', 'scalex', 'scalet'
        ]
        np.random.seed(0)
        print('=== Multi-agent STAMP ===\n'
              f'Initializing : {arg.run_name}\n'
              f'Minibatch size : {arg.minibatch_size}, Buffer size : {arg.buffer_size}')
        if self.cuda_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_devices
            print(f'cuda devices : {self.cuda_devices} on', torch.cuda.get_device_name())
        ray.init()
        if arg.use_wandb:
            wandb.init(project=arg.project_name, name=arg.run_name, entity='your_entity', config=vars(arg),
                       notes=arg.wandb_notes, resume='allow', id=arg.wandb_id)
        if arg.save_files:
            os.makedirs(arg.model_path, exist_ok=True)
            os.makedirs(arg.gifs_path, exist_ok=True)

    def set(self, net, optimizer, lr_scheduler):
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def write_to_board(self, data, curr_episode):
        data = np.array(data)
        data = list(np.nanmean(data, axis=0))
        # 这里我们按照 data 的顺序写入 scalar
        # data 的前 9 个值对应了 PPO 训练过程中的信息
        # 后续则依次对应 metric_names
        reward, value, p_loss, v_loss, entropy, grad_norm, returns, clipfrac, approx_kl = data[:9]
        rest_data = data[9:]

        metrics = {
            'Loss/Learning Rate': self.lr_scheduler.get_last_lr()[0],
            'Loss/Value': value,
            'Loss/Policy Loss': p_loss,
            'Loss/Value Loss': v_loss,
            'Loss/Entropy': entropy,
            'Loss/Grad Norm': grad_norm,
            'Loss/Clip Frac': clipfrac,
            'Loss/Approx Policy KL': approx_kl,
            'Loss/Reward': reward,
            'Loss/Return': returns,
        }
        idx_offset = 0
        for name in self.metric_names:
            metrics[f'Perf/{name}'] = rest_data[idx_offset]
            idx_offset += 1

        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(tag=k, scalar_value=v, global_step=curr_episode)

        if arg.use_wandb:
            wandb.log(metrics, step=curr_episode)

    def load_saved_model(self):
        print('Loading model :', arg.run_name)
        checkpoint = torch.load(arg.model_path + '/checkpoint.pth')
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        print("Current episode set to :", curr_episode)
        print('Learning rate :', self.optimizer.state_dict()['param_groups'][0]['lr'])
        return curr_episode

    def save_model(self, curr_episode):
        print('Saving model', end='\n')
        checkpoint = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode": curr_episode,
            "lr_decay": self.lr_scheduler.state_dict()
        }
        print("1")
        path_checkpoint = "./" + arg.model_path + "/checkpoint.pth"
        print("2")
        print(path_checkpoint)
        torch.save(checkpoint, path_checkpoint)

def main():
    logger = Logger() # 创建logger,用来记录训练过程
    device = torch.device('cuda') if arg.use_gpu_driver else torch.device('cpu') # 选择设备
    local_device = torch.device('cuda') if arg.use_gpu_runner else torch.device('cpu')
    global_network = AttentionNet(arg.embedding_dim).to(device) # 创建全局网络（Multi-agent policy + calue）
    global_optimizer = optim.Adam(global_network.parameters(), lr=arg.lr) # 指定优化器
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=arg.lr_decay_step, gamma=0.96) # 指定学习率调度器
    logger.set(global_network, global_optimizer, lr_decay)

    # 当前的训练轮数
    curr_episode = 0
    training_data = []

    # 要不要从checkpoint开始
    if arg.load_model:
        curr_episode = logger.load_saved_model()

    # 通过ray启动多个Runner，并行收集数据
    meta_runners = [Runner.remote(i) for i in range(arg.num_meta)]

    # 要不要wandb(weights and biases)
    if arg.use_wandb:
        wandb.watch(global_network, log_freq=500, log_graph=True)

    dp_global_network = nn.DataParallel(global_network) # dataparallel来包装，使得可以多卡train

    try:
        # 主循环，其实类似single-agent
        while True:
            # 每轮建立一个新的、空的buffer，来收集本批次的数据
            meta_jobs = []
            buffer = {k: [] for k in logger.episode_buffer_keys}

            # actual_buffer_size = len(buffer['history'])
            # buffer_idxs = np.arange(actual_buffer_size)
            # np.random.shuffle(buffer_idxs)

            # 每一批都随机一个 budget, graph_size, history_size, target_size
            budget_size = np.random.uniform(*arg.budget_size)
            graph_size = np.random.randint(*arg.graph_size)
            history_size = np.random.randint(*arg.history_size)
            target_size = np.random.randint(*arg.target_size)

            # 同步当前全局网络权重给各个Runner
            if device != local_device:
                weights = global_network.to(local_device).state_dict()
                global_network.to(device)
            else:
                weights = global_network.state_dict()
            weights_id = ray.put(weights)

            # 给每个 runner 分配一个job,每个job都是一个episode
            for i, meta_agent in enumerate(meta_runners):
                meta_jobs.append(
                    meta_agent.job.remote(weights_id, curr_episode, budget_size,
                                          graph_size, history_size, target_size)
                )
                curr_episode += 1

            # 等待所有Runner完成, 收集返回的(episode_buffer, perf_metrics)
            done_id, meta_jobs = ray.wait(meta_jobs, num_returns=arg.num_meta)
            done_jobs = ray.get(done_id)

            # 建一个perf_metrics，存性能指标，简单初始化下
            perf_metrics = {}
            for n in logger.metric_names:
                perf_metrics[n] = []

            # 将各Runner返回的数据(episode_buffer)拼装到主进程的buffer
            for job in done_jobs:
                job_results, metrics = job
                for k in job_results.keys():
                    buffer[k] += job_results[k]
                for n in logger.metric_names:
                    perf_metrics[n].append(metrics[n])

            # 转成 torch 张量
            b_history_inputs = torch.stack(buffer['history'], dim=0)
            b_edge_inputs    = torch.stack(buffer['edge'], dim=0)
            b_dist_inputs    = torch.stack(buffer['dist'], dim=0)
            b_dt_inputs      = torch.stack(buffer['dt'], dim=0)
            b_current_inputs = torch.stack(buffer['nodeidx'], dim=0)   # shape: [buffer_size, n_agents, 1]
            b_logp           = torch.stack(buffer['logp'], dim=0)      # shape: [buffer_size, n_agents, 1]
            b_action         = torch.stack(buffer['action'], dim=0)    # shape: [buffer_size, n_agents, 1]
            b_value          = torch.stack(buffer['value'], dim=0)     # shape: [buffer_size, n_agents]
            b_reward         = torch.stack(buffer['reward'], dim=0)
            b_return         = torch.stack(buffer['return'], dim=0)
            b_advantage      = torch.stack(buffer['advantage'], dim=0)
            b_temporal_mask  = torch.stack(buffer['temporalmask'], dim=0)
            b_spatio_mask    = torch.stack(buffer['spatiomask'], dim=0)
            b_pos_encoding   = torch.stack(buffer['spatiope'], dim=0)

            # 开始PPO training，训练arg.update_epochs次
            actual_buffer_size = len(buffer['history'])
            buffer_idxs = np.arange(actual_buffer_size)
            np.random.shuffle(buffer_idxs)

            scaler = amp.GradScaler()
            ratio = None # 放的位置真的是对的吗？？？？？# 好像终于对了QAQ

            for epoch in range(arg.update_epochs):
                # 每次训练前打乱数据，做mini-batch
                np.random.shuffle(buffer_idxs)
                actual_buffer_size = len(buffer['history'])
                # print("Actual buffer size:", actual_buffer_size)
                
                #遍历mini-batch
                for start in tqdm(range(0, arg.buffer_size, arg.minibatch_size)):
                    # if start >= actual_buffer_size:
                    #     break  # 或 continue
                    end = min(start + arg.minibatch_size, actual_buffer_size)

                    # if end > actual_buffer_size:
                    #     end = actual_buffer_size
                    mb_idxs = buffer_idxs[start:end]
                    if len(mb_idxs) == 0:
                        continue

                    # 取这一小批数据
                    mb_history_inputs = b_history_inputs[mb_idxs].to(device)
                    mb_edge_inputs    = b_edge_inputs   [mb_idxs].to(device)
                    mb_dist_inputs    = b_dist_inputs   [mb_idxs].to(device)
                    mb_dt_inputs      = b_dt_inputs     [mb_idxs].to(device)
                    mb_current_inputs = b_current_inputs[mb_idxs].to(device)  # [mini_batch, n_agents, 1]
                    mb_pos_encoding   = b_pos_encoding  [mb_idxs].to(device)
                    mb_temporal_mask  = b_temporal_mask [mb_idxs].to(device)
                    mb_spatio_mask    = b_spatio_mask   [mb_idxs].to(device)

                    # 转device
                    mb_old_logp  = b_logp     [mb_idxs].to(device)  # [mb, n_agents, 1]
                    mb_action    = b_action   [mb_idxs].to(device)  # [mb, n_agents, 1]
                    mb_value     = b_value    [mb_idxs].to(device)  # [mb, n_agents]
                    mb_return    = b_return   [mb_idxs].to(device)  # [mb, n_agents]
                    mb_advantage = b_advantage[mb_idxs].to(device)  # [mb, n_agents]

                    with amp.autocast(device_type='cuda'):
                        # 前向计算
                        # print("enter forward again of episode", curr_episode)
                        logp_list, value_all = dp_global_network(
                            mb_history_inputs, mb_edge_inputs, mb_dist_inputs, mb_dt_inputs,
                            mb_current_inputs, mb_pos_encoding, mb_temporal_mask, mb_spatio_mask
                        )
                        # print(f"Debug - logp_list shape: {logp_list.shape}")
                        # print(f"Debug - mb_action shape: {mb_action.shape}")

                        # 对每个 agent 的动作 logp 进行 gather
                        # action: [mb, n_agents, 1], 需要先把 action squeeze(-1)
                        # ????
                        gather_actions = mb_action.squeeze(-1).long()
                        max_action = logp_list.shape[-1] - 1
                        gather_actions = torch.clamp(gather_actions, 0, max_action)
                        if (gather_actions > max_action).any():
                            print(f"Warning: Invalid actions detected. Clamping indices to [0, {max_action}]")
                            gather_actions = torch.clamp(gather_actions, 0, max_action)
                        # gather_actions = mb_action.squeeze(-1).long()  # [mb, n_agents]

                        # 生成一个 index 用于 gather => shape [mb, n_agents, 1]
                        logp_gathered = torch.zeros((mb_old_logp.size(0),arg.num_agents, 1), device=device)  # 取出当前动作a的logP
                        for ag in range(arg.num_agents):
                            logp_agent = logp_list[:, ag, :] # [mb, neighbor_size]
                            aidx = gather_actions[:, ag].view(-1,1) # [mb, 1]
                            logp_g = torch.gather(logp_agent, 1, aidx)  # [mb, 1]
                            logp_gathered[:, ag, 0] = logp_g.squeeze(1)

                        # ppo中raotio的含义：ratio = exp( new_logp - old_logp )
                        logratio = logp_gathered - mb_old_logp.squeeze(-1) # [mb, n_agents]
                        ratio = logratio.exp() # [mb, n_agents]
                        ratio = ratio.transpose(0, 1) # 新加的，为啥维度会突然倒过来了

                        surr1 = mb_advantage * ratio
                        surr2 = mb_advantage * ratio.clamp(1-0.2, 1+0.2)

                        # policy loss: agent级别的平均
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # value_loss: MSE
                        value_loss = nn.MSELoss()(value_all, mb_return).mean()

                        p_list = logp_list.exp()   # => [mb, n_agents, neighbor_size]
                        entropy_all = -(logp_list * p_list).sum(dim=-1).mean()  # scalar
                        loss = policy_loss + 0.2 * value_loss - 0.0 * entropy_all

                    # assert gather_actions.max() <= logp_list.shape[-1], f"Action index {gather_actions.max()} exceeds neighbor size {logp_list.shape[-1]}"
                    
                    global_optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(global_optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=5, norm_type=2)
                    scaler.step(global_optimizer)
                    scaler.update()
            lr_decay.step()

            if ratio is None:
                print("No valid minibatch => ratio is None => skip saving/checkpoint this round.")
                continue

            # 计算clip_frac, approx_kl
            with torch.no_grad():
                # approximate
                if ratio.shape != logratio.shape:
                    ratio = ratio.transpose(0, 1)
                clip_frac = ((ratio - 1.0).abs() > 0.2).float().mean()
                approx_kl = ((ratio - 1) - logratio).mean()

            # 收集 PPO 训练过程数据
            ppo_info = [
                b_reward.mean().item(),            # reward
                b_value.mean().item(),             # value
                policy_loss.item(),                # p_loss
                value_loss.item(),                 # v_loss
                entropy_all.item(),                # entropy
                grad_norm.item(),                  # grad_norm
                b_return.mean().item(),            # returns
                clip_frac.item(),                  # clip_frac
                approx_kl.item()                   # approx_kl
            ]
            # perf metrics
            perf_data = []
            for n in logger.metric_names:
                perf_data.append(np.nanmean(perf_metrics[n]))

            data_line = ppo_info + perf_data
            training_data.append(data_line)

            if len(training_data) >= arg.summary_window and arg.save_files:
                logger.write_to_board(training_data, curr_episode)
                training_data = []

            # 每64个episode保存模型
            print("Before save model")
            if curr_episode % 64 == 0 and arg.save_files:
                logger.save_model(curr_episode)

    except KeyboardInterrupt:
        print('User interrupt, abort remotes...')
        if arg.use_wandb:
            wandb.finish(quiet=True)
        for runner in meta_runners:
            ray.kill(runner)

if __name__ == "__main__":
    main()
