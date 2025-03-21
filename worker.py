import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from env import Env
from network import AttentionNet
from arguments import arg

class Worker:
    def __init__(
        self, meta_id, local_net, global_step,
        budget_size, graph_size=arg.graph_size[0], history_size=arg.history_size[0],
        target_size=arg.target_size[0], device='cuda',
        greedy=False, save_image=False
    ):
        self.meta_id = meta_id
        self.device = device
        self.greedy = greedy
        self.global_step = global_step
        self.save_image = save_image

        self.graph_size = graph_size
        self.history_size = history_size
        self.num_agents = arg.num_agents

        self.env = Env(
            graph_size=self.graph_size,
            k_size=arg.k_size,
            budget_size=budget_size,
            target_size=target_size
        )
        self.local_net = local_net
        self.avgpool = torch.nn.AvgPool1d(kernel_size=arg.history_stride, stride=arg.history_stride, ceil_mode=True)

        # buffer 关键字
        self.episode_buffer_keys = [
            'history','edge','dist','dt','nodeidx','logp',
            'action','value','temporalmask','spatiomask','spatiope',
            'done','reward','advantage','return'
        ]

    def reset_env_input(self):
        node_coords, graph, node_feature, budget = self.env.reset()
        # node_inputs形状 => [node, 2 + target*4 + 1]
        node_inputs = np.concatenate((node_coords, node_feature), axis=1)
        node_inputs = torch.Tensor(node_inputs).unsqueeze(0).to(self.device)

        # 跟原先类似，只是多 agent
        node_history = node_inputs.repeat(self.history_size, 1, 1)  # [history, node, input_dim]
        history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)) \
                                 .permute(2, 0, 1).unsqueeze(0)     # [1, hpool, node, input_dim]

        edge_inputs_list = []
        for idx, neighbors in enumerate(graph.values()):
            nb_list = list(map(int, neighbors))
            if len(nb_list) < arg.k_size:
                # 如果邻居数不足，则用当前节点的索引填充
                nb_list += [idx] * (arg.k_size - len(nb_list))
            else:
                nb_list = nb_list[:arg.k_size]
            edge_inputs_list.append(nb_list)
        edge_inputs = torch.tensor(edge_inputs_list).unsqueeze(0).to(self.device)   # [1, node, k_size]

        # 下面是对 spatio_pos_encoding
        spatio_pos_encoding = self.graph_pos_encoding(edge_inputs_list)
        spatio_pos_encoding = torch.from_numpy(spatio_pos_encoding).float().unsqueeze(0).to(self.device)  # [1, node, 32]

        # dt_history => [1, history, 1]
        dt_history = torch.zeros((1, self.history_size, 1), device=self.device)
        dt_pool_inputs = self.avgpool(dt_history.permute(0,2,1)).permute(0,2,1)

        # dist_inputs => [1, node, 1] (但多agent时，需要一个“当前节点”概念)
        # 由于在 Env 中， self.current_node_indices 是列表，需要先看看第一个 agent
        dist_inputs = self.calc_distance_to_nodes(self.env.current_node_indices[0])
        dist_inputs[dist_inputs > 1.5] = 1.5
        dist_inputs = torch.Tensor(dist_inputs).unsqueeze(0).to(self.device)

        # current_index => shape [1, n_agents, 1]
        current_index = []
        for i in range(self.num_agents):
            idx = self.env.current_node_indices[i]
            current_index.append([idx])
        current_index = torch.tensor([current_index], device=self.device) # => [1, n_agents, 1]

        spatio_mask = torch.zeros((1, self.graph_size+1, arg.k_size), dtype=torch.bool).to(self.device)
        temporal_mask = torch.tensor([1])
        return (
            node_coords, node_history, history_pool_inputs,
            edge_inputs, dist_inputs, dt_history, dt_pool_inputs,
            current_index, spatio_pos_encoding, temporal_mask, spatio_mask, budget
        )

    def run_episode(self, episode_number):
        perf_metrics = {}
        episode_buffer = {k: [] for k in self.episode_buffer_keys}

        (node_coords, node_history, history_pool_inputs,
         edge_inputs, dist_inputs, dt_history, dt_pool_inputs,
         current_index, spatio_pos_encoding, temporal_mask,
         spatio_mask, remain_budget) = self.reset_env_input()

        routes = [ [idx.item()] for idx in current_index[0,:,0] ]
        rmse_list = [self.env.RMSE]
        unc_list  = [self.env.unc_list]
        jsd_list  = [self.env.JS_list]
        kld_list  = [self.env.KL_list]
        budget_list = [0]
        step = 0

        for step in range(arg.episode_steps):
            if self.save_image: # 先不改，不确定
                self.env.plot(routes, self.global_step, step, arg.gifs_path, budget_list,
                              [0] + [r.item() for r in episode_buffer['reward']], jsd_list)

            with torch.no_grad(): # 这里相当于对logp_list做了拓展
                logp_list, value_all = self.local_net(
                    history_pool_inputs, edge_inputs, dist_inputs, dt_pool_inputs,
                    current_index, spatio_pos_encoding, temporal_mask, spatio_mask
                )
                # logp_list => [1, n_agents, neighbor_size]
                # value_all => [1, n_agents]

            # 如果是 greedy 则对 logp_list.exp() 做 argmax
            # 否则就做多项式采样
            logp_exp = logp_list.exp().squeeze(0) # => [n_agents, neighbor_size]
            actions = []
            chosen_logp = []
            # 对每个agent分别判断
            for ag in range(self.num_agents):
                if self.greedy:
                    aidx = torch.argmax(logp_exp[ag], dim=-1)
                else:
                    aidx = torch.multinomial(logp_exp[ag], 1).squeeze(-1)
                actions.append(aidx.item())
                chosen_logp.append(logp_list[0, ag, aidx].item())
            actions = np.array(actions)  # shape [n_agents]
            chosen_logp = np.array(chosen_logp)

            # 根据 action 找到对应下个节点
            #  edge_inputs => [1, node, k_size]
            next_node_indices = []
            for ag in range(self.num_agents):
                curr_nd = current_index[0, ag, 0].item()
                nxt_nd  = edge_inputs[0, curr_nd, actions[ag]].item()
                next_node_indices.append(int(nxt_nd))

            reward, done, node_feature, remain_budget, _ = self.env.step(next_node_indices, self.global_step)

            # 写入 episode buffer
            episode_buffer['history'].append(history_pool_inputs.squeeze(0)) # shape => [hpool, node, input_dim]
            episode_buffer['edge'].append(edge_inputs.squeeze(0))
            episode_buffer['dist'].append(dist_inputs.squeeze(0))
            episode_buffer['dt'].append(dt_pool_inputs.squeeze(0))
            episode_buffer['nodeidx'].append(current_index.squeeze(0)) # [n_agents, 1]
            # logp => [n_agents, 1]
            logp_tensor = torch.tensor(chosen_logp, dtype=torch.float32, device=self.device).view(1, self.num_agents)
            episode_buffer['logp'].append(logp_tensor)
            # action => [n_agents, 1]
            action_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device).view(self.num_agents,1)
            episode_buffer['action'].append(action_tensor)
            # value => [n_agents]
            episode_buffer['value'].append(value_all.squeeze(0))
            episode_buffer['temporalmask'].append(temporal_mask)
            episode_buffer['spatiomask'].append(spatio_mask)
            episode_buffer['spatiope'].append(spatio_pos_encoding.squeeze(0))
            episode_buffer['reward'].append(torch.tensor([reward], device=self.device))
            episode_buffer['done'].append(done)

            # 更新 for logging, agents轨迹存储在routes
            for ag in range(self.num_agents):
                routes[ag].append(next_node_indices[ag])
            rmse_list.append(self.env.RMSE)
            unc_list.append(self.env.unc_list)
            jsd_list.append(self.env.JS_list)
            kld_list.append(self.env.KL_list)
            budget_list.append(self.env.budget_init - remain_budget)

            # 准备下一步输入
            for ag in range(self.num_agents):
                current_index[0, ag, 0] = next_node_indices[ag]

            # 更新 node_inputs
            node_inputs = np.concatenate((node_coords, node_feature), axis=1)
            node_inputs = torch.Tensor(node_inputs).unsqueeze(0).to(self.device)

            # 更新 node_history
            node_history = torch.cat((node_history, node_inputs), dim=0)[-self.history_size:, :, :]
            history_pool_inputs = self.avgpool(node_history.permute(1,2,0)).permute(2,0,1).unsqueeze(0)

            # 更新 dt_history
            dt_history += (budget_list[-1] - budget_list[-2]) / (1.993*3)
            dt_history = torch.cat((dt_history, torch.zeros((1,1,1), device=self.device)), dim=1)[:, -self.history_size:, :]
            dt_pool_inputs = self.avgpool(dt_history.permute(0,2,1)).permute(0,2,1)

            # 计算所有智能体到各节点的距离并取最小距离
            all_dist = [self.calc_distance_to_nodes(idx) for idx in self.env.current_node_indices]
            all_dist = np.stack(all_dist, axis=1)              # shape: (num_nodes, n_agents)
            min_dist = np.min(all_dist, axis=1).reshape(-1, 1) # 每个节点对最近智能体的距离
            min_dist[min_dist > 1.5] = 1.5                     # 距离截断
            dist_inputs = torch.tensor(min_dist, dtype=torch.float32).unsqueeze(0).to(self.device)

            spatio_mask = torch.zeros((1, self.graph_size+1, arg.k_size), dtype=torch.bool).to(self.device)
            temporal_mask = torch.tensor([(len(budget_list)-1)//arg.history_stride + 1])

            if done: # 还没改，有问题的
                # save gif
                if self.save_image:
                    self.env.plot(routes, self.global_step, step + 1, arg.gifs_path, budget_list,
                                  [0] + [r.item() for r in episode_buffer['reward']], jsd_list)
                    self.make_gif(arg.gifs_path, episode_number)
                    self.save_image = False
                node_coords, node_history, history_pool_inputs, edge_inputs, dist_inputs, dt_history, dt_pool_inputs, \
                    current_index, spatio_pos_encoding, temporal_mask, spatio_mask = self.reset_env_input()
                routes = [ [idx.item()] for idx in current_index[0,:,0] ]
                rmse_list = [self.env.RMSE]
                jsd_list = [self.env.JS_list]
                kld_list = [self.env.KL_list]
                budget_list = [0]

        # save gif
        if self.save_image:
            self.env.plot(routes, self.global_step, step + 1, arg.gifs_path, budget_list,
                          [0] + [r.item() for r in episode_buffer['reward']], jsd_list)
            self.make_gif(arg.gifs_path, episode_number)
            self.save_image = False
        
        # GAE
        episode_len = len(episode_buffer['done'])  # 等价于你存了多少次 step 数据
        
        with torch.no_grad():
            if not done:
                # 如果未走到 done，就 bootstrap
                _, next_value_all = self.local_net(
                    history_pool_inputs, edge_inputs, dist_inputs, dt_pool_inputs,
                    current_index, spatio_pos_encoding, temporal_mask, spatio_mask
                )
                # next_value_all 形状 [1, n_agents]
                next_value_all = next_value_all.squeeze(0)  # => [n_agents]
            else:
                # 如果环境已经 done，则下一个价值=0
                next_value_all = torch.zeros((self.num_agents,), device=self.device)

            # 广义优势估计
            lastgaelam = torch.zeros((self.num_agents,), device=self.device)

            # 2) 用 reversed(range(episode_len)) 而不是 range(arg.episode_steps)
            for i in reversed(range(episode_len)):
                if i == episode_len - 1:
                    # 注意这里改为 episode_len - 1
                    nextnonterminal = 1.0 - float(episode_buffer['done'][i])
                    # nextnonterminal = 1.0 - float(done)
                    nextvalues      = next_value_all
                else:
                    nextnonterminal = 1.0 - float(episode_buffer['done'][i+1])
                    nextvalues      = episode_buffer['value'][i+1]  # i+1 不会越界

                delta = (
                    episode_buffer['reward'][i].item()
                    + arg.gamma * nextvalues * nextnonterminal
                    - episode_buffer['value'][i]
                )
                lastgaelam = delta + arg.gamma * arg.gae_lambda * nextnonterminal * lastgaelam

                adv = lastgaelam.clone()
                episode_buffer['advantage'].insert(0, adv)

            # 3) 同步计算 return
            for i in range(episode_len):
                ret = episode_buffer['advantage'][i] + episode_buffer['value'][i]
                episode_buffer['return'].append(ret)

        # 整理 perf_metrics
        # 这里简单放一些统计
        n_visit = list(map(len, self.env.visit_t))
        gap_visit = [np.diff(t) for t in self.env.visit_t]
        avgnvisit = np.mean(n_visit)
        stdnvisit = np.std(n_visit)
        if min(n_visit)>1:
            avggapvisit = np.mean([np.mean(g) for g in gap_visit])
            stdgapvisit = np.std([np.mean(g) for g in gap_visit])
        else:
            avggapvisit, stdgapvisit = np.nan, np.nan

        perf_metrics['avgnvisit'] = avgnvisit
        perf_metrics['stdnvisit'] = stdnvisit
        perf_metrics['avggapvisit'] = avggapvisit
        perf_metrics['stdgapvisit'] = stdgapvisit
        perf_metrics['avgrmse']  = np.mean(rmse_list)
        perf_metrics['avgunc']   = np.mean([np.mean(u) for u in unc_list])
        perf_metrics['avgjsd']   = np.mean([np.mean(j) for j in jsd_list])
        perf_metrics['avgkld']   = np.mean([np.mean(k) for k in kld_list])
        perf_metrics['stdunc']   = np.mean([np.std(u) for u in unc_list])
        perf_metrics['stdjsd']   = np.mean([np.std(j) for j in jsd_list])
        perf_metrics['covtr']    = self.env.cov_trace
        perf_metrics['f1']       = self.env.gp_wrapper.eval_avg_F1(self.env.ground_truth, self.env.curr_t)
        perf_metrics['mi']       = self.env.gp_wrapper.eval_avg_MI(self.env.curr_t)
        perf_metrics['js']       = self.env.JS
        perf_metrics['rmse']     = self.env.RMSE
        perf_metrics['scalex']   = 0.1
        perf_metrics['scalet']   = 3

        return episode_buffer, perf_metrics

    def graph_pos_encoding(self, edge_inputs):
        A_matrix = np.zeros((self.graph_size+1, self.graph_size+1))
        D_matrix = np.zeros((self.graph_size+1, self.graph_size+1))
        for i in range(self.graph_size+1):
            for j in range(self.graph_size+1):
                if (j in edge_inputs[i]) and (i != j):
                    A_matrix[i][j] = 1.0
        for i in range(self.graph_size+1):
            deg = len(edge_inputs[i]) - 1
            if deg<=0: deg=1
            D_matrix[i][i] = 1/np.sqrt(deg)
        L = np.eye(self.graph_size+1) - np.matmul(np.matmul(D_matrix, A_matrix), D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        # 取前 32 个特征向量
        eigen_vector = eigen_vector[:, 1:32+1]
        return eigen_vector

    def calc_distance_to_nodes(self, current_idx):
        all_dist = []
        current_coord = self.env.node_coords[current_idx]
        for i, point_coord in enumerate(self.env.node_coords):
            d_current_to_point = self.env.graph_ctrl.calc_distance(current_coord, point_coord)
            all_dist.append(d_current_to_point)
        return np.asarray(all_dist).reshape(-1,1)

    def make_gif(self, path, n):
        writer_filename = '{}/{}_cov_trace_{:.4g}.mp4'.format(path, n, self.env.cov_trace)
        with imageio.get_writer(writer_filename, fps=5, format='ffmpeg') as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')
        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)



if __name__ == '__main__':
    save_img = False
    if save_img:
        if not os.path.exists(arg.gifs_path):
            os.makedirs(arg.gifs_path)
    device = torch.device('cuda')
    localNetwork = AttentionNet(arg.embedding_dim).cuda()
    worker = Worker(0, localNetwork, 100000, budget_size=30, graph_size=200, history_size=50, target_size=3, save_image=save_img)
    worker.run_episode(0)
