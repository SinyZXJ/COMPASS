import copy
import os
import imageio
import numpy as np
import time
import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import Env
from network import AttentionNet
from arguments import arg_eval


class WorkerEval:
    def __init__(self, meta_id, local_net, global_step, device='cuda', greedy=True, save_image=False, config=None):
        self.device = device
        self.greedy = greedy
        self.meta_id = meta_id
        self.global_step = global_step
        self.save_image = save_image

        if config:
            arg_eval.graph_size, arg_eval.history_size, arg_eval.target_size, arg_eval.target_speed = config
        print(f'#node {arg_eval.graph_size}, #history {arg_eval.history_size}, #tgt {arg_eval.target_size}, speed {arg_eval.target_speed}')

        self.num_agents = arg_eval.num_agents
        self.env_speed = arg_eval.target_speed
        self.env = Env(
            graph_size=arg_eval.graph_size,
            k_size=arg_eval.k_size,
            budget_size=arg_eval.budget_size,
            target_size=arg_eval.target_size
        )
        self.local_net = local_net
        self.avgpool = torch.nn.AvgPool1d(kernel_size=arg_eval.history_stride, stride=arg_eval.history_stride, ceil_mode=True)
        self.perf_metrics = None
        self.planning_time = 0

    def run_episode(self, curr_eval):
        perf_metrics = {}
        node_coords, graph, node_feature, budget = self.env.reset(seed=self.global_step)
        node_inputs = np.concatenate((node_coords, node_feature), axis=1)
        node_inputs = torch.Tensor(node_inputs).unsqueeze(0).to(self.device)
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

        current_index = []
        for ag in range(self.num_agents):
            current_index.append([self.env.current_node_indices[ag]])
        current_index = torch.tensor([current_index], device=self.device)

        spatio_mask = torch.zeros((1, arg_eval.graph_size + 1, arg_eval.k_size), dtype=torch.bool).to(self.device)
        temporal_mask = torch.tensor([1])

        routes = [[idx] for idx in current_index[0, :, 0].tolist()]
        rmse_list = [self.env.RMSE]
        jsd_list = [self.env.JS]
        unc_list = [self.env.unc]
        budget_list = [0]
        rew_list = []
        unc_stddev_list = [np.std(self.env.unc_list)]
        jsd_stddev_list = [np.std(self.env.JS_list)]

        for step in range(1024):
            if self.save_image:
                # 这里先构造 div_array，保证其为二维数组
                div_array = np.array(self.env.JS_list) if self.env.JS_list is not None else np.zeros((len(budget_list), arg_eval.target_size))
                if div_array.ndim == 1:
                    div_array = div_array.reshape(-1, 1)
                self.env.plot(
                    routes, 
                    self.global_step, 
                    step, 
                    arg_eval.result_path, 
                    budget_list, 
                    rew_list, 
                    div_array
                )

            time_start = time.time()
            with torch.no_grad():
                logp_list, shared_value = self.local_net(
                    history_pool_inputs,
                    edge_inputs,
                    dist_inputs,
                    dt_pool_inputs,
                    current_index,
                    spatio_pos_encoding,
                    temporal_mask,
                    spatio_mask
                )
            logp_exp = logp_list.exp().squeeze(0)
            actions = []
            for ag in range(self.num_agents):
                if self.greedy:
                    aidx = torch.argmax(logp_exp[ag], dim=-1)
                else:
                    probs = logp_exp[ag]
                    if not torch.all(torch.isfinite(probs)) or torch.sum(probs) <= 0:
                        probs = torch.ones_like(probs) / probs.shape[-1]
                    aidx = torch.multinomial(probs, 1).squeeze(-1)
                actions.append(aidx.item())

            next_node_indices = []
            for ag in range(self.num_agents):
                curr_nd = current_index[0, ag, 0].item()
                aidx = actions[ag]
                if aidx >= edge_inputs.shape[2]:
                    aidx = edge_inputs.shape[2] - 1
                nxt_nd = edge_inputs[0, curr_nd, aidx].item()
                next_node_indices.append(int(nxt_nd))

            time_end = time.time()
            self.planning_time += (time_end - time_start)

            reward, done, node_feature, remain_budget, metrics = self.env.step(next_node_indices, global_step=self.global_step, eval_speed=self.env_speed)
            rew_list.append(reward)
            budget_list.append(self.env.budget_init - remain_budget)

            rmse_list.append(self.env.RMSE)
            jsd_list.append(self.env.JS)
            unc_list.append(self.env.unc)
            unc_stddev_list.append(np.std(self.env.unc_list))
            jsd_stddev_list.append(np.std(self.env.JS_list))

            for ag in range(self.num_agents):
                routes[ag].append(next_node_indices[ag])

            if done:
                if self.save_image:
                    div_array = np.array(self.env.JS_list) if self.env.JS_list is not None else np.zeros((len(budget_list), arg_eval.target_size))
                    if div_array.ndim == 1:
                        div_array = div_array.reshape(-1, 1)
                    self.env.plot(
                        routes, 
                        self.global_step, 
                        step + 1, 
                        arg_eval.result_path, 
                        budget_list, 
                        rew_list, 
                        div_array
                    )
                    self.make_gif(arg_eval.result_path, self.global_step)
                perf_metrics['minnvisit'] = np.min([len(v) for v in self.env.visit_t]) if self.env.visit_t else np.nan
                perf_metrics['avgnvisit'] = np.mean([len(v) for v in self.env.visit_t]) if self.env.visit_t else np.nan
                perf_metrics['stdnvisit'] = np.std([len(v) for v in self.env.visit_t]) if self.env.visit_t else np.nan
                perf_metrics['avgrmse'] = np.mean(rmse_list)
                perf_metrics['avgjsd'] = np.mean(jsd_list)
                perf_metrics['avgunc'] = np.mean(unc_list)
                perf_metrics['stdunc'] = np.mean(unc_stddev_list)
                perf_metrics['stdjsd'] = np.mean(jsd_stddev_list)
                print(f'\033[92mmeta{self.meta_id:02}:\033[0m episode {curr_eval} done at {step} steps, avg JS {perf_metrics["avgjsd"]:.4g}')
                break

            for ag in range(self.num_agents):
                current_index[0, ag, 0] = next_node_indices[ag]

            new_node_inputs = np.concatenate((node_coords, node_feature), axis=1)
            new_node_inputs = torch.Tensor(new_node_inputs).unsqueeze(0).to(self.device)
            node_history = torch.cat((node_history, new_node_inputs), dim=0)[-arg_eval.history_size :, :, :]
            history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)

            dt_history += (budget_list[-1] - budget_list[-2]) / (1.993 * 3)
            dt_history = torch.cat((dt_history, torch.zeros((1, 1, 1), device=self.device)), dim=1)[:, -arg_eval.history_size:, :]
            dt_pool_inputs = self.avgpool(dt_history.permute(0, 2, 1)).permute(0, 2, 1)

            all_dist = [self.calc_distance_to_nodes(idx) for idx in self.env.current_node_indices]
            all_dist = np.stack(all_dist, axis=1)
            min_dist = np.min(all_dist, axis=1).reshape(-1, 1)
            min_dist[min_dist > 1.5] = 1.5
            dist_inputs = torch.tensor(min_dist, dtype=torch.float32).unsqueeze(0).to(self.device)

            spatio_mask = torch.zeros((1, arg_eval.graph_size + 1, arg_eval.k_size), dtype=torch.bool).to(self.device)
            temporal_mask = torch.tensor([(len(budget_list) - 1) // arg_eval.history_stride + 1])

        print(f'episode {self.global_step} planning time: {self.planning_time / len(routes[0]):.3f}/step')
        return perf_metrics

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
                if not self.env.frame_files:
                    print(f"Warning: No frames found to create GIF for episode {n}")
                    return
                for frame_path in self.env.frame_files:
                    image = imageio.v2.imread(frame_path)
                    writer.append_data(image)
            print(f'GIF/MP4 saved to {writer_filename}')
            for filename in self.env.frame_files:
                if os.path.exists(filename):
                    os.remove(filename)
            self.env.frame_files = []
        except Exception as e:
            print(f"Error creating GIF/MP4: {e}")


if __name__ == '__main__':
    save_img = False
    if save_img and not os.path.exists(arg_eval.gifs_path):
        os.makedirs(arg_eval.gifs_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    localNetwork = AttentionNet(arg_eval.embedding_dim).to(device)
    checkpoint = torch.load(f'../{arg_eval.model_path}/checkpoint.pth', map_location=device)
    localNetwork.load_state_dict(checkpoint['model'])
    worker = WorkerEval(meta_id=0, local_net=localNetwork, global_step=2, device=device, save_image=save_img)
    worker.run_episode(curr_eval=0)
