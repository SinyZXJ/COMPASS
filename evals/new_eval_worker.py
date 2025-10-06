# new_eval_worker.py (Version 4 - Final)

import copy
import os
import imageio
import numpy as np
import time
import torch
import torch.nn.functional as F
from env import Env
from network import AttentionNet
from arguments import arg_eval

class WorkerEval:
    METRIC_NAMES = ['avgrmse', 'avgjsd', 'avgunc', 'minnvisit', 'avgnvisit', 'stdnvisit']

    def __init__(self, meta_id, local_net, global_step, device='cuda', save_image=False, config=None, eval_method='MODEL'):
        self.device = device
        self.meta_id = meta_id
        self.global_step = global_step
        self.save_image = save_image
        self.eval_method = eval_method
        self.local_net = local_net if self.eval_method == 'MODEL' else None

        if config:
            arg_eval.graph_size, arg_eval.history_size, arg_eval.target_size, arg_eval.target_speed = config
        
        self.num_agents = arg_eval.num_agents
        self.env = Env(
            graph_size=arg_eval.graph_size,
            k_size=arg_eval.k_size,
            budget_size=arg_eval.budget_size,
            target_size=arg_eval.target_size
        )
        self.last_visit_time = -np.ones(arg_eval.graph_size + 1)
        self.avgpool = torch.nn.AvgPool1d(kernel_size=arg_eval.history_stride, stride=arg_eval.history_stride, ceil_mode=True)

    def _get_forced_move_action(self, agent_id, neighbors, preferred_action_idx):
        """
        核心修复函数：检查首选动作是否导致原地不动。如果是，则强制选择一个不同的邻居。
        """
        curr_node = self.env.current_node_indices[agent_id]
        
        # 检查首选动作对应的下一个节点
        if preferred_action_idx < len(neighbors):
            preferred_next_node = neighbors[preferred_action_idx]
            if preferred_next_node != curr_node:
                return preferred_action_idx # 如果不原地不动，就采纳

        # 如果首选动作是原地不动，或索引无效，则强制重新选择
        print(f"  [DEBUG] Method '{self.eval_method}', Agent {agent_id} at node {curr_node}: "
        f"Attempted to stay still (Action idx {preferred_action_idx} -> node {preferred_next_node}). "
        f"Overriding decision...")
        valid_indices = [i for i, neighbor_node in enumerate(neighbors) if neighbor_node != curr_node]
        if not valid_indices:
            # 极端情况：所有邻居都是自己，只能原地不动
            return 0 
        
        # 从所有“有效移动”的选项中随机选一个
        return np.random.choice(valid_indices)

    def get_actions(self, **kwargs):
        """根据评估方法获取动作，并确保不会原地不动"""
        actions = []
        edge_inputs_list = kwargs['edge_inputs'].squeeze(0).tolist()

        if self.eval_method == 'AUCTION':
            # AUCTION 方法逻辑比较特殊，单独处理
            node_unc = np.mean(self.env.node_feature[:, :-1].reshape(self.env.graph_size + 1, self.env.n_targets, 4)[:, :, 1], axis=1)
            num_candidates = self.num_agents * 3
            candidate_nodes = np.argsort(node_unc)[-num_candidates:]
            assigned_nodes = [-1] * self.num_agents
            bids = []
            for ag in range(self.num_agents):
                curr_node_idx = self.env.current_node_indices[ag]
                for cand_node in candidate_nodes:
                    if cand_node == curr_node_idx: continue
                    dist = self.env.graph_ctrl.calc_distance(self.env.node_coords[curr_node_idx], self.env.node_coords[cand_node])
                    utility = node_unc[cand_node] - 0.5 * dist
                    bids.append((utility, ag, cand_node))
            bids.sort(key=lambda x: x[0], reverse=True)
            assigned_agents, assigned_candidates = set(), set()
            for utility, agent_id, node_id in bids:
                if agent_id in assigned_agents or node_id in assigned_candidates: continue
                assigned_nodes[agent_id] = node_id
                assigned_agents.add(agent_id)
                assigned_candidates.add(node_id)
            
            for ag in range(self.num_agents):
                curr_node_idx = self.env.current_node_indices[ag]
                target_node = assigned_nodes[ag]
                neighbors = edge_inputs_list[curr_node_idx]
                if target_node == -1: # 如果未分配到节点
                    valid_neighbors = [n for n in neighbors if n != curr_node_idx]
                    target_node = np.random.choice(valid_neighbors) if valid_neighbors else neighbors[0]
                
                if target_node in neighbors:
                    actions.append(neighbors.index(target_node))
                else:
                    target_coord = self.env.node_coords[target_node]
                    neighbor_coords = self.env.node_coords[neighbors]
                    dists = np.linalg.norm(neighbor_coords - target_coord, axis=1)
                    actions.append(np.argmin(dists))
            return actions

        # --- 对 MODEL, GREEDY, COVERAGE, RANDOM 的统一处理流程 ---
        for ag in range(self.num_agents):
            curr_node = self.env.current_node_indices[ag]
            neighbors = edge_inputs_list[curr_node]
            preferred_action_idx = 0

            if self.eval_method == 'MODEL':
                with torch.no_grad():
                    logp_list, _ = self.local_net(kwargs['history_pool'], kwargs['edge_inputs'], kwargs['dist_inputs'], kwargs['dt_pool'], kwargs['current_index'], kwargs['spatio_pe'], kwargs['temporal_mask'], kwargs['spatio_mask'])
                preferred_action_idx = torch.argmax(logp_list.exp().squeeze(0)[ag], dim=-1).item()
            elif self.eval_method == 'RANDOM':
                preferred_action_idx = np.random.randint(0, len(neighbors))
            elif self.eval_method == 'GREEDY':
                node_unc = np.mean(self.env.node_feature[:, :-1].reshape(self.env.graph_size + 1, self.env.n_targets, 4)[:, :, 1], axis=1)
                preferred_action_idx = np.argmax(node_unc[neighbors])
            elif self.eval_method == 'COVERAGE':
                self.last_visit_time[curr_node] = self.env.curr_t
                preferred_action_idx = np.argmin(self.last_visit_time[neighbors])
            
            # 使用辅助函数来确保智能体移动
            final_action_idx = self._get_forced_move_action(ag, neighbors, preferred_action_idx)
            actions.append(final_action_idx)
            
        return actions

    def run_episode(self, curr_eval):
        # 此函数的主体与 Version 2 相同，为保持简洁省略。
        # 核心修改是上面的 get_actions 方法。
        perf_metrics = {}
        trajectory_data = []
        node_coords, graph, node_feature, budget = self.env.reset(seed=self.global_step)
        node_inputs = torch.Tensor(node_feature).unsqueeze(0).to(self.device)
        node_history = node_inputs.repeat(arg_eval.history_size, 1, 1)
        history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)
        edge_inputs_list = [list(map(int, node)) for node in graph.values()]
        spatio_pos_encoding = self.graph_pos_encoding(edge_inputs_list)
        spatio_pos_encoding = torch.from_numpy(spatio_pos_encoding).float().unsqueeze(0).to(self.device)
        edge_inputs = torch.tensor(edge_inputs_list).unsqueeze(0).to(self.device)
        dt_history = torch.zeros((1, arg_eval.history_size, 1), device=self.device)
        dt_pool_inputs = self.avgpool(dt_history.permute(0, 2, 1)).permute(0, 2, 1)
        all_dist = [self.calc_distance_to_nodes(idx) for idx in self.env.current_node_indices]
        all_dist = np.stack(all_dist, axis=1)
        min_dist = np.min(all_dist, axis=1).reshape(-1, 1)
        min_dist[min_dist > 1.5] = 1.5
        dist_inputs = torch.tensor(min_dist, dtype=torch.float32).unsqueeze(0).to(self.device)
        current_index = torch.tensor([[[idx] for idx in self.env.current_node_indices]], device=self.device)
        spatio_mask = torch.zeros((1, arg_eval.graph_size + 1, arg_eval.k_size), dtype=torch.bool).to(self.device)
        temporal_mask = torch.tensor([1])

        for step in range(1024):
            action_kwargs = {
                'history_pool': history_pool_inputs, 'edge_inputs': edge_inputs, 'dist_inputs': dist_inputs,
                'dt_pool': dt_pool_inputs, 'current_index': current_index, 'spatio_pe': spatio_pos_encoding,
                'temporal_mask': temporal_mask, 'spatio_mask': spatio_mask
            }
            actions = self.get_actions(**action_kwargs)
            next_node_indices = []
            for ag in range(self.num_agents):
                curr_nd = self.env.current_node_indices[ag]
                aidx = actions[ag]
                neighbors = edge_inputs.squeeze(0).tolist()[curr_nd]
                nxt_nd = neighbors[aidx]
                next_node_indices.append(int(nxt_nd))
            _, done, node_feature, _, _ = self.env.step(next_node_indices, global_step=self.global_step, eval_speed=1/20.0)
            trajectory_data.append((self.env.budget_init - self.env.budget, self.env.unc, self.env.JS, self.env.RMSE))
            if done:
                perf_metrics['minnvisit'] = np.min([len(v) for v in self.env.visit_t]) if self.env.visit_t and any(self.env.visit_t) else np.nan
                perf_metrics['avgnvisit'] = np.mean([len(v) for v in self.env.visit_t]) if self.env.visit_t and any(self.env.visit_t) else np.nan
                perf_metrics['stdnvisit'] = np.std([len(v) for v in self.env.visit_t]) if self.env.visit_t and any(self.env.visit_t) else np.nan
                if trajectory_data:
                    _, uncs, jsds, rmses = zip(*trajectory_data)
                    perf_metrics['avgrmse'] = np.nanmean(rmses)
                    perf_metrics['avgjsd'] = np.nanmean(jsds)
                    perf_metrics['avgunc'] = np.nanmean(uncs)
                print(f'meta-{self.meta_id:02} | method: {self.eval_method} | eval: {curr_eval} done in {step} steps. Avg Unc: {perf_metrics.get("avgunc", -1):.4g}')
                break
            self.env.current_node_indices = next_node_indices
            current_index = torch.tensor([[[idx] for idx in self.env.current_node_indices]], device=self.device)
            new_node_inputs = torch.Tensor(node_feature).unsqueeze(0).to(self.device)
            node_history = torch.cat((node_history[1:], new_node_inputs), dim=0)
            history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)
        return perf_metrics, trajectory_data

    # ... (graph_pos_encoding, calc_distance_to_nodes, make_gif 方法保持不变) ...
    def graph_pos_encoding(self, edge_inputs_list):
        A_matrix = np.zeros((arg_eval.graph_size + 1, arg_eval.graph_size + 1))
        D_matrix = np.zeros((arg_eval.graph_size + 1, arg_eval.graph_size + 1))
        for i in range(arg_eval.graph_size + 1):
            neighbors = edge_inputs_list[i]
            for j in neighbors:
                if j != i:
                    A_matrix[i][j] = 1.0
            deg = len([n for n in neighbors if n != i])
            deg = max(deg, 1)
            D_matrix[i][i] = 1 / np.sqrt(deg)
        L = np.eye(arg_eval.graph_size + 1) - D_matrix @ A_matrix @ D_matrix
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:, 1:32 + 1]
        return eigen_vector

    def calc_distance_to_nodes(self, current_idx):
        all_dist = []
        current_coord = self.env.node_coords[current_idx]
        for point_coord in self.env.node_coords:
            d_current_to_point = self.env.graph_ctrl.calc_distance(current_coord, point_coord)
            all_dist.append(d_current_to_point)
        return np.asarray(all_dist).reshape(-1, 1)

    def make_gif(self, path, n):
        os.makedirs(path, exist_ok=True)
        writer_filename = f"{path}/{n}_cov_trace_{self.env.cov_trace:.4g}.mp4"
        try:
            with imageio.get_writer(writer_filename, fps=5, format='FFMPEG', codec='libx264') as writer:
                if not hasattr(self.env, 'frame_files') or not self.env.frame_files:
                    return
                for frame_path in self.env.frame_files:
                    image = imageio.v2.imread(frame_path)
                    writer.append_data(image)
            for filename in self.env.frame_files:
                if os.path.exists(filename):
                    os.remove(filename)
            self.env.frame_files = []
        except Exception as e:
            print(f"Error creating GIF/MP4: {e}")