# worker.py
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
        budget_size, graph_size, history_size, # graph_size here is num_random_nodes from args
        target_size, device='cuda',
        greedy=False, save_image=False
    ):
        self.meta_id = meta_id
        self.device = device
        self.greedy = greedy
        self.global_step = global_step
        self.save_image = save_image

        # graph_size passed from driver is the number of randomly generated nodes.
        # The environment (GraphController) adds one start node.
        self.num_random_nodes = graph_size 
        self.actual_num_nodes = self.num_random_nodes + 1 
        
        self.history_size_unpooled = history_size # Unpooled history size for env setup
        self.num_agents = arg.num_agents

        self.env = Env(
            graph_size=self.num_random_nodes, # Env's GraphController uses this 
            k_size=arg.k_size,
            budget_size=budget_size,
            target_size=target_size
        )
        self.local_net = local_net
        self.avgpool = torch.nn.AvgPool1d(kernel_size=arg.history_stride, stride=arg.history_stride, ceil_mode=True)

        self.episode_buffer_keys = [
            'history_pooled', 'edge', 'dist', 'dt_pooled', 'nodeidx', 'logp',
            'action', 
            'value',        # Per-agent: [num_agents]
            'temporalmask_pooled', 'spatiomask', 'spatiope',
            'done',         # Scalar
            'reward',       # Scalar (team reward used for each agent's GAE)
            'advantage',    # Per-agent: [num_agents]
            'return'        # Per-agent: [num_agents]
        ]
        self.metric_names = [ 
            'avgnvisit', 'stdnvisit', 'avggapvisit', 'stdgapvisit',
            'avgrmse', 'avgunc', 'avgjsd', 'avgkld',
            'stdunc', 'stdjsd', 'covtr', 'f1', 'mi', 'js', 'rmse', 'scalex', 'scalet'
        ]

    def reset_env_input(self):
        node_coords_from_env, graph_connections_from_env, node_feature_initial_from_env, budget_from_env = self.env.reset()
        
        assert node_coords_from_env.shape[0] == self.actual_num_nodes, \
            f"Mismatch: env.node_coords has {node_coords_from_env.shape[0]} nodes, expected {self.actual_num_nodes}"
        assert node_feature_initial_from_env.shape[0] == self.actual_num_nodes
        assert len(graph_connections_from_env) == self.actual_num_nodes

        node_feature_for_history_tensor = torch.Tensor(node_feature_initial_from_env).unsqueeze(0).to(self.device) 

        node_history_unpooled = node_feature_for_history_tensor.repeat(self.history_size_unpooled, 1, 1) # [UnpooledH, ActualN, Feat]
        # Pool for network input:
        # Input to avgpool: [Batch (ActualN), Channels (Feat), SeqLen (UnpooledH)]
        history_pooled_inputs = self.avgpool(node_history_unpooled.permute(1, 2, 0)) 
        # Output from avgpool: [ActualN, Feat, PooledH]
        # Permute back and add batch dim: [1, PooledH, ActualN, Feat]
        history_pooled_inputs = history_pooled_inputs.permute(2, 0, 1).unsqueeze(0)

        edge_inputs_list = []
        for node_idx_int in range(self.actual_num_nodes): 
            node_idx_str = str(node_idx_int)
            neighbors_data = graph_connections_from_env.get(node_idx_str, {})
            nb_list = []
            if isinstance(neighbors_data, dict):
                 nb_list = [int(n_str) for n_str in neighbors_data.keys()]
            elif isinstance(neighbors_data, (list, tuple)): # Should be list of neighbors from Graph.edges
                 nb_list = [int(n) for n in neighbors_data] # Assuming Graph.edges returns list of int node IDs

            # Pad or truncate to k_size
            if len(nb_list) < arg.k_size:
                nb_list += [node_idx_int] * (arg.k_size - len(nb_list)) 
            else:
                nb_list = nb_list[:arg.k_size]
            edge_inputs_list.append(nb_list)
        edge_inputs = torch.tensor(edge_inputs_list).unsqueeze(0).to(self.device) # [1, ActualNumNodes, K_size]

        spatio_pos_encoding_np = self.graph_pos_encoding(edge_inputs_list, num_nodes_for_laplacian=self.actual_num_nodes)
        spatio_pos_encoding = torch.from_numpy(spatio_pos_encoding_np).float().unsqueeze(0).to(self.device) # [1, ActualNumNodes, PosEncDim]

        dt_history_unpooled = torch.zeros((1, self.history_size_unpooled, 1), device=self.device) # [1, UnpooledH, 1]
        dt_pool_inputs = self.avgpool(dt_history_unpooled.permute(0, 2, 1)).permute(0, 2, 1) # [1, PooledH, 1]

        all_dist_to_nodes = [self.calc_distance_to_nodes(idx, num_total_nodes=self.actual_num_nodes) for idx in self.env.current_node_indices] 
        all_dist_stacked = np.stack(all_dist_to_nodes, axis=2) # [ActualNumNodes, 1, NumAgents] -> want [ActualNumNodes, NumAgents, 1]
        all_dist_stacked = all_dist_stacked.transpose(0,2,1) # [ActualNumNodes, NumAgents, 1]
        min_dist_per_node = np.min(all_dist_stacked, axis=1) # [ActualNumNodes, 1]
        min_dist_per_node[min_dist_per_node > 1.5] = 1.5
        dist_inputs = torch.tensor(min_dist_per_node, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, ActualNumNodes, 1]

        current_agent_indices_list = [[idx] for idx in self.env.current_node_indices]
        current_agent_indices_tensor = torch.tensor([current_agent_indices_list], device=self.device) # [1, NumAgents, 1]

        spatio_mask_pointer = torch.zeros((1, self.actual_num_nodes, arg.k_size), dtype=torch.bool).to(self.device)
        
        temporal_mask_pooled = torch.tensor([1], device=self.device) # Initial valid pooled length is 1

        return (
            node_coords_from_env, node_history_unpooled, history_pooled_inputs,
            edge_inputs, dist_inputs, dt_history_unpooled, dt_pool_inputs,
            current_agent_indices_tensor, spatio_pos_encoding, temporal_mask_pooled, spatio_mask_pointer, budget_from_env
        )

    def graph_pos_encoding(self, edge_inputs_list_from_graph, num_nodes_for_laplacian):
        A_matrix = np.zeros((num_nodes_for_laplacian, num_nodes_for_laplacian))
        D_matrix = np.zeros((num_nodes_for_laplacian, num_nodes_for_laplacian))

        for i in range(num_nodes_for_laplacian):
            actual_neighbors_of_i = set()
            if i < len(edge_inputs_list_from_graph):
                for neighbor_val in edge_inputs_list_from_graph[i]:
                    if 0 <= neighbor_val < num_nodes_for_laplacian:
                        actual_neighbors_of_i.add(neighbor_val)
            actual_neighbors_of_i.discard(i)

            for neighbor_j in actual_neighbors_of_i:
                A_matrix[i, neighbor_j] = 1.0
            
            deg = len(actual_neighbors_of_i)
            D_matrix[i,i] = 1.0 / np.sqrt(max(deg, 1)) 

        L = np.eye(num_nodes_for_laplacian) - D_matrix @ A_matrix @ D_matrix
        
        try:
            eigen_values, eigen_vector = np.linalg.eig(L)
        except np.linalg.LinAlgError:
            print(f"Warning: Eigenvalue computation failed for Laplacian of size {num_nodes_for_laplacian}. Returning zeros for PE.")
            return np.zeros((num_nodes_for_laplacian, 32))

        idx = eigen_values.argsort()
        eigen_vector = np.real(eigen_vector[:, idx])
        
        num_eigvec = 32
        
        # Handle cases where graph is too small or disconnected leading to fewer eigenvectors
        if eigen_vector.shape[1] <= 1: # Not enough eigenvectors to skip the first one
            selected_eigen_vector = eigen_vector[:, :min(num_eigvec, eigen_vector.shape[1])]
        else:
            max_idx_to_select = min(num_eigvec + 1, eigen_vector.shape[1])
            selected_eigen_vector = eigen_vector[:, 1:max_idx_to_select]

        current_cols = selected_eigen_vector.shape[1]
        if current_cols < num_eigvec:
            padding_width = num_eigvec - current_cols
            if padding_width > 0:
                padding = np.zeros((num_nodes_for_laplacian, padding_width))
                selected_eigen_vector = np.concatenate((selected_eigen_vector, padding), axis=1)
        elif current_cols > num_eigvec:
            selected_eigen_vector = selected_eigen_vector[:, :num_eigvec]
       
        if selected_eigen_vector.shape[1] != num_eigvec:
            final_pe = np.zeros((num_nodes_for_laplacian, num_eigvec))
            cols_to_copy = min(selected_eigen_vector.shape[1], num_eigvec)
            if cols_to_copy > 0 and selected_eigen_vector.size > 0 :
                final_pe[:, :cols_to_copy] = selected_eigen_vector[:, :cols_to_copy]
            return final_pe
           
        return selected_eigen_vector

    def calc_distance_to_nodes(self, current_agent_node_idx, num_total_nodes):
        all_dist = np.zeros((num_total_nodes, 1)) 
        
        # Ensure current_agent_node_idx is valid for self.env.node_coords
        if not (0 <= current_agent_node_idx < self.env.node_coords.shape[0]):
            print(f"Warning: current_agent_node_idx {current_agent_node_idx} is out of bounds for node_coords size {self.env.node_coords.shape[0]}")
            return all_dist # Return zeros or handle error appropriately

        current_agent_coord = self.env.node_coords[current_agent_node_idx]
        
        for i in range(num_total_nodes): 
            if i < self.env.node_coords.shape[0]: 
               point_coord = self.env.node_coords[i]
               dist_val = self.env.graph_ctrl.calc_distance(current_agent_coord, point_coord)
               all_dist[i,0] = dist_val
            else: 
               all_dist[i,0] = np.inf 
        return all_dist


    def run_episode(self, episode_number):
        perf_metrics_dict = {} # Will be populated with actual metrics later
        episode_buffer = {k: [] for k in self.episode_buffer_keys}

        (node_coords_env, node_history_unpooled, history_pooled_net_input,
         edge_net_input, dist_net_input, dt_history_unpooled, dt_pooled_net_input,
         current_agent_indices_net_input, spatio_pos_encoding_net_input,
         temporal_mask_pooled_net_input, spatio_mask_pointer_net_input, 
         remain_budget_env) = self.reset_env_input()

        routes_vis = [[idx.item()] for idx in current_agent_indices_net_input.squeeze(0)] 
        
        # Initialize metrics lists for logging within the episode
        rmse_list_log = [self.env.RMSE if hasattr(self.env, 'RMSE') and self.env.RMSE is not None else np.nan]
        unc_list_log = [np.mean(self.env.unc_list) if hasattr(self.env, 'unc_list') and self.env.unc_list is not None and len(self.env.unc_list) > 0 else np.nan]
        initial_js_per_target = list(getattr(self.env, 'JS_list', [])) # 从env获取初始JS_list
        if not initial_js_per_target or len(initial_js_per_target) != self.env.n_targets:
            initial_js_per_target = [np.nan] * self.env.n_targets
        js_divergences_history_for_plot = [initial_js_per_target[:self.env.n_targets]] # 保证长度正确

        budget_list_log = [0.0] 
        
        done_episode = False
        # rewards_for_plot 的初始化也应该在循环外，并且在循环内追加
        rewards_for_plot = [0.0] # 初始奖励为0

        for step_in_episode in range(arg.episode_steps):
            if self.save_image:
                # 修改: 将累积的JS散度历史传递给plot函数
                div_array_for_plot_np = np.array(js_divergences_history_for_plot)
                # div_array_for_plot_np 应该是 (num_steps_so_far+1, n_targets)
                
                self.env.plot(routes_vis, self.global_step, step_in_episode, arg.gifs_path, budget_list_log,
                              rewards_for_plot, # rewards_for_plot 列表在循环中增长
                              div_array_for_plot_np) # 传递 NumPy 数组

            # ... (hist_in_buf, edge_in_buf, etc. and network call remain the same) ...
            hist_in_buf = history_pooled_net_input.squeeze(0).clone() 
            edge_in_buf = edge_net_input.squeeze(0).clone()          
            dist_in_buf = dist_net_input.squeeze(0).clone()           
            dt_in_buf   = dt_pooled_net_input.squeeze(0).clone()      
            nodeidx_in_buf = current_agent_indices_net_input.squeeze(0).clone() 
            spem_in_buf = spatio_pos_encoding_net_input.squeeze(0).clone() 
            tmask_in_buf = temporal_mask_pooled_net_input.clone()          
            smask_in_buf = spatio_mask_pointer_net_input.squeeze(0).clone() 

            with torch.no_grad():
                logp_list_from_net, individual_agent_values_from_net = self.local_net(
                    history_pooled_net_input, edge_net_input, dist_net_input, dt_pooled_net_input,
                    current_agent_indices_net_input, spatio_pos_encoding_net_input,
                    temporal_mask_pooled_net_input, smask_in_buf 
                ) 

            logp_dist_per_agent = logp_list_from_net.exp().squeeze(0) 
            
            actions_chosen_indices = [] 
            logp_of_chosen_actions_list = [] 

            for ag_idx in range(self.num_agents):
                agent_action_probs = logp_dist_per_agent[ag_idx] 
                if self.greedy:
                    chosen_action_k_idx = torch.argmax(agent_action_probs, dim=-1)
                else:
                    if not torch.all(torch.isfinite(agent_action_probs)) or torch.sum(agent_action_probs) <= 1e-8: 
                         agent_action_probs = torch.ones_like(agent_action_probs) / agent_action_probs.shape[-1]
                    chosen_action_k_idx = torch.multinomial(agent_action_probs, 1).squeeze(-1)
                
                actions_chosen_indices.append(chosen_action_k_idx.item())
                logp_of_chosen_actions_list.append(logp_list_from_net[0, ag_idx, chosen_action_k_idx].item())

            next_node_indices_env = []
            current_nodes_for_action = current_agent_indices_net_input.squeeze(0) 
            for ag_idx in range(self.num_agents):
                current_node_of_agent = current_nodes_for_action[ag_idx, 0].item()
                k_neighbor_action_idx = actions_chosen_indices[ag_idx]
                actual_next_node = edge_net_input[0, current_node_of_agent, k_neighbor_action_idx].item()
                next_node_indices_env.append(int(actual_next_node))

            team_reward_env, done_episode, updated_node_features_env, remain_budget_env, _ = self.env.step(
                next_node_indices_env, self.global_step
            )
            # 修改: 在每个步骤后，将当前步骤的每个目标的JS散度值追加到历史列表中
            current_step_js_per_target = list(getattr(self.env, 'JS_list', []))
            if not current_step_js_per_target or len(current_step_js_per_target) != self.env.n_targets:
                current_step_js_per_target = [np.nan] * self.env.n_targets
            js_divergences_history_for_plot.append(current_step_js_per_target[:self.env.n_targets])


            episode_buffer['history_pooled'].append(hist_in_buf)
            episode_buffer['edge'].append(edge_in_buf)
            episode_buffer['dist'].append(dist_in_buf)
            episode_buffer['dt_pooled'].append(dt_in_buf)
            episode_buffer['nodeidx'].append(nodeidx_in_buf) 
            
            logp_tensor_buf = torch.tensor(logp_of_chosen_actions_list, dtype=torch.float32, device=self.device).unsqueeze(-1) 
            episode_buffer['logp'].append(logp_tensor_buf)
            
            action_tensor_buf = torch.tensor(actions_chosen_indices, dtype=torch.int64, device=self.device).unsqueeze(-1) 
            episode_buffer['action'].append(action_tensor_buf)
            
            episode_buffer['value'].append(individual_agent_values_from_net.squeeze(0).clone())
            
            episode_buffer['temporalmask_pooled'].append(tmask_in_buf) 
            episode_buffer['spatiomask'].append(smask_in_buf) 
            episode_buffer['spatiope'].append(spem_in_buf) 
            
            episode_buffer['reward'].append(torch.tensor(team_reward_env, dtype=torch.float32, device=self.device)) 
            rewards_for_plot.append(team_reward_env) # For correct length in plot call

            episode_buffer['done'].append(done_episode) 

            # --- Logging and State Update for next step ---
            # ... (route_vis, rmse_list_log, unc_list_log updates remain the same) ...
            for ag_idx in range(self.num_agents):
                routes_vis[ag_idx].append(next_node_indices_env[ag_idx])
            
            rmse_list_log.append(getattr(self.env, 'RMSE', np.nan))
            unc_list_log.append(np.mean(self.env.unc_list) if hasattr(self.env, 'unc_list') and self.env.unc_list is not None and len(self.env.unc_list) > 0 else np.nan)
            # jsd_list_log is no longer needed here as we use js_divergences_history_for_plot for plotting per-target JS
            budget_list_log.append(self.env.budget_init - remain_budget_env)


            # ... (rest of the state updates for network inputs remain the same) ...
            for ag_idx in range(self.num_agents):
                 current_agent_indices_net_input[0, ag_idx, 0] = next_node_indices_env[ag_idx]

            new_node_features_for_history = torch.Tensor(updated_node_features_env).unsqueeze(0).to(self.device)
            if node_history_unpooled.shape[0] >= self.history_size_unpooled: 
                 node_history_unpooled = torch.cat((node_history_unpooled[1:], new_node_features_for_history), dim=0)
            else: 
                 node_history_unpooled = torch.cat((node_history_unpooled, new_node_features_for_history), dim=0)[-self.history_size_unpooled:, :, :]

            history_pooled_net_input = self.avgpool(node_history_unpooled.permute(1,2,0)).permute(2,0,1).unsqueeze(0)

            dt_step_unpooled = (budget_list_log[-1] - budget_list_log[-2]) if len(budget_list_log) > 1 else 0.0
            dt_update_value = dt_step_unpooled / (arg.k_size / 2.0 if arg.k_size > 0 else 1.0) 
            
            dt_history_unpooled = torch.roll(dt_history_unpooled, shifts=-1, dims=1)
            dt_history_unpooled[0, -1, 0] = dt_update_value 
            dt_pooled_net_input = self.avgpool(dt_history_unpooled.permute(0,2,1)).permute(0,2,1)

            all_dist_to_nodes_next = [self.calc_distance_to_nodes(idx, self.actual_num_nodes) for idx in self.env.current_node_indices]
            all_dist_stacked_next = np.stack(all_dist_to_nodes_next, axis=2) 
            all_dist_stacked_next = all_dist_stacked_next.transpose(0,2,1) 
            min_dist_per_node_next = np.min(all_dist_stacked_next, axis=1)
            min_dist_per_node_next[min_dist_per_node_next > 1.5] = 1.5
            dist_net_input = torch.tensor(min_dist_per_node_next, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            num_valid_env_steps = len(budget_list_log) - 1 
            current_pooled_valid_len = (num_valid_env_steps // arg.history_stride) + 1
            max_pooled_len = history_pooled_net_input.shape[1] 
            temporal_mask_pooled_net_input = torch.tensor([min(current_pooled_valid_len, max_pooled_len)], device=self.device)


            if done_episode:
                if self.save_image:
                    div_array_for_plot_np_final = np.array(js_divergences_history_for_plot)
                    self.env.plot(routes_vis, self.global_step, step_in_episode + 1, arg.gifs_path, budget_list_log,
                                  rewards_for_plot, div_array_for_plot_np_final)
                    self.make_gif(arg.gifs_path, episode_number)
                    belief_traj_path = f"{arg.gifs_path}/{episode_number}_final_belief_traj.png"
                    self.env.save_belief_trajectory_only(routes_vis, belief_traj_path)

                    self.save_image = False 
                break
        
        # ... (GAE calculation and performance metrics remain the same as in your provided complete worker.py) ...
        # Ensure GAE and perf_metrics_dict population is correct based on previous full version.
        # The key change for this error was how div_list for plot is created and passed.

        # --- GAE Calculation (Copied from your previous complete version, should be correct for IPPO) ---
        episode_len = len(episode_buffer['done'])
        if episode_len == 0:
            print(f"Warning: Worker {self.meta_id} episode {episode_number} had zero steps.")
            for key_to_init in self.episode_buffer_keys: 
                if key_to_init not in episode_buffer or not episode_buffer[key_to_init]: 
                     episode_buffer[key_to_init] = [] 
            for name_metric in self.metric_names:
                perf_metrics_dict[name_metric] = np.nan
            return episode_buffer, perf_metrics_dict

        with torch.no_grad():
            final_spatio_mask_for_bootstrap = spatio_mask_pointer_net_input # From last state of loop

            if not done_episode: 
                _, next_individual_agent_values_bootstrap = self.local_net(
                    history_pooled_net_input, edge_net_input, dist_net_input, dt_pooled_net_input,
                    current_agent_indices_net_input, spatio_pos_encoding_net_input,
                    temporal_mask_pooled_net_input, final_spatio_mask_for_bootstrap
                ) 
                next_values_per_agent_bootstrap = next_individual_agent_values_bootstrap.squeeze(0) 
            else: 
                next_values_per_agent_bootstrap = torch.zeros(self.num_agents, device=self.device)

            advantages_list_this_episode = []             
            gae_per_agent = torch.zeros(self.num_agents, device=self.device)
            for i in reversed(range(episode_len)):
                if i == episode_len - 1:
                    nextnonterminal = 1.0 - float(done_episode) 
                    current_step_next_values_per_agent = next_values_per_agent_bootstrap
                else:
                    nextnonterminal = 1.0 - float(episode_buffer['done'][i+1]) 
                    current_step_next_values_per_agent = episode_buffer['value'][i+1] 

                reward_scalar_this_step = episode_buffer['reward'][i] 
                value_per_agent_this_step = episode_buffer['value'][i]   

                delta_per_agent = reward_scalar_this_step + arg.gamma * current_step_next_values_per_agent * nextnonterminal - value_per_agent_this_step
                
                gae_per_agent = delta_per_agent + arg.gamma * arg.gae_lambda * nextnonterminal * gae_per_agent
                advantages_list_this_episode.append(gae_per_agent.clone())
            
            advantages_list_this_episode.reverse() 
            episode_buffer['advantage'] = advantages_list_this_episode
            episode_buffer['return'] = [adv + val for adv, val in zip(episode_buffer['advantage'], episode_buffer['value'])]

        # --- Performance Metrics (Copied from your previous complete version) ---
        for name in self.metric_names: 
            perf_metrics_dict[name] = np.nan

        n_visit_counts = [len(vt_list) for vt_list in self.env.visit_t if isinstance(vt_list, list) and vt_list]
        if n_visit_counts:
            perf_metrics_dict['avgnvisit'] = np.mean(n_visit_counts)
            perf_metrics_dict['stdnvisit'] = np.std(n_visit_counts)
        
        gap_visit_means = []
        for vt_list_item in self.env.visit_t:
            if isinstance(vt_list_item, list) and len(vt_list_item) > 1:
                gaps = np.diff(vt_list_item)
                if len(gaps) > 0: gap_visit_means.append(np.mean(gaps))
        if gap_visit_means:
            perf_metrics_dict['avggapvisit'] = np.mean(gap_visit_means)
            perf_metrics_dict['stdgapvisit'] = np.std(gap_visit_means)

        if rmse_list_log and not all(np.isnan(el) for el in rmse_list_log if isinstance(el, (float, np.floating))): perf_metrics_dict['avgrmse']  = np.nanmean(rmse_list_log)
        if unc_list_log and not all(np.isnan(el) for el in unc_list_log if isinstance(el, (float, np.floating))): perf_metrics_dict['avgunc']   = np.nanmean(unc_list_log)
        # For plotting, we used js_divergences_history_for_plot. For the scalar avgjsd metric:
        # We need to average per-target JS values at each step, then average those over steps.
        avg_js_per_step = [np.nanmean(step_js_list) for step_js_list in js_divergences_history_for_plot if step_js_list] # js_divergences_history_for_plot is list of lists
        if avg_js_per_step and not all(np.isnan(el) for el in avg_js_per_step if isinstance(el, (float, np.floating))): perf_metrics_dict['avgjsd'] = np.nanmean(avg_js_per_step)

        
        final_unc_val = getattr(self.env, 'unc_list', [])
        final_js_val = getattr(self.env, 'JS_list', [])
        if isinstance(final_unc_val, list) and final_unc_val: perf_metrics_dict['stdunc'] = np.std(final_unc_val)
        if isinstance(final_js_val, list) and final_js_val: perf_metrics_dict['stdjsd'] = np.std(final_js_val)
        
        if hasattr(self.env, 'cov_trace'): perf_metrics_dict['covtr'] = self.env.cov_trace
        try:
             if hasattr(self.env, 'gp_wrapper') and self.env.gp_wrapper is not None and \
                hasattr(self.env, 'ground_truth') and self.env.ground_truth is not None:
                 if hasattr(self.env.gp_wrapper, 'eval_avg_F1'):
                    perf_metrics_dict['f1'] = self.env.gp_wrapper.eval_avg_F1(self.env.ground_truth, self.env.curr_t)
                 if hasattr(self.env.gp_wrapper, 'eval_avg_MI'):
                     perf_metrics_dict['mi'] = self.env.gp_wrapper.eval_avg_MI(self.env.curr_t)
        except AttributeError: 
             pass 
        except Exception:
             pass
        if hasattr(self.env, 'JS') and self.env.JS is not None: perf_metrics_dict['js'] = self.env.JS 
        if hasattr(self.env, 'RMSE') and self.env.RMSE is not None: perf_metrics_dict['rmse'] = self.env.RMSE 
        perf_metrics_dict['scalex']   = 0.1 
        perf_metrics_dict['scalet']   = 3   

        if self.save_image and not done_episode: 
            div_array_for_plot_np_final_step = np.array(js_divergences_history_for_plot)
            self.env.plot(routes_vis, self.global_step, step_in_episode + 1, arg.gifs_path, budget_list_log,
                          rewards_for_plot, div_array_for_plot_np_final_step)
            self.make_gif(arg.gifs_path, episode_number)

        return episode_buffer, perf_metrics_dict

    # ... (make_gif method remains the same as in your provided complete worker.py) ...
    def make_gif(self, path, n_episode): # Copied from your last full version
        os.makedirs(path, exist_ok=True)
        cov_trace_val = getattr(self.env, 'cov_trace', np.nan)
        cov_trace_str = f"{cov_trace_val:.4g}" if not np.isnan(cov_trace_val) else "nan"
        writer_filename = f"{path}/{n_episode}_cov_trace_{cov_trace_str}.mp4"
        try:
            if not hasattr(self.env, 'frame_files') or not self.env.frame_files:
                print(f"Warning: No frames found for MP4 for episode {n_episode}")
                return

            with imageio.get_writer(writer_filename, fps=5, format='FFMPEG', codec='libx264', 
                                    ffmpeg_params=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p']) as writer:
                 for frame_path in self.env.frame_files:
                      if not os.path.exists(frame_path):
                          print(f"Warning: Frame file not found during GIF creation: {frame_path}")
                          continue
                      try:
                          image = imageio.v2.imread(frame_path)
                          writer.append_data(image)
                      except Exception as e_read:
                          print(f"Error reading frame {frame_path} for MP4: {e_read}")
            print(f'MP4 saved to {writer_filename}')
        except Exception as e_write:
            print(f"Error creating MP4 for episode {n_episode}: {e_write}")
        finally: 
            if hasattr(self.env, 'frame_files') and self.env.frame_files:
                for filename in self.env.frame_files:
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except OSError as e_remove:
                        print(f"Error removing frame file {filename}: {e_remove}")
                self.env.frame_files = []
