import numpy as np
from itertools import product
from utils.graph_controller import GraphController
from utils.target_controller import VTSPGaussian
from matplotlib import pyplot as plt
from gaussian_process import GaussianProcessWrapper
from arguments import arg

def add_t(X, t: float):
    # 给 (x, y) 数组添加时间维度
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)

class Env:
    def __init__(self, graph_size, k_size, budget_size=None, target_size=None, start=None, obstacles=None):
        self.graph_size = graph_size
        self.k_size = k_size
        self.budget_init = budget_size
        self.budget = budget_size
        self.n_targets = target_size
        self.obstacles = obstacles

        # NEW for multi-agent
        self.num_agents = arg.num_agents  # 读ma数量
        # 指定初始位置
        if start is None:
            self.start_positions = np.random.rand(self.num_agents, 2)
        else:
            self.start_positions = np.array(start)
            if self.start_positions.shape[0] != self.num_agents:
                raise ValueError("给定 start 的数量和 num_agents 不符！")
        self.curr_t = 0.0
        # 记录每个 agent 拜访各目标的时间
        self.visit_t = [[] for _ in range(self.n_targets)]

        self.graph_ctrl = GraphController(self.graph_size, self.start_positions[0:1], self.k_size, self.obstacles)
        self.node_coords, self.graph = self.graph_ctrl.generate_graph()
        
        self.underlying_distrib = None
        self.ground_truth = None
        self.high_info_idx = None

        # GP
        self.gp_wrapper = None
        self.node_feature = None

        # evaluation metrics
        self.RMSE = None
        self.JS, self.JS_list = None, None
        self.KL, self.KL_list = None, None
        self.cov_trace = None
        self.unc, self.unc_list = None, None
        self.unc_sum, self.unc_sum_list = None, None

        # 新加的ma核心状态
        # current_node_indices[i] 表示第 i 个 agent 当前所在节点
        # dist_residuals[i] 表示 agent i 在下一次采样距离到达之前，还剩多少路程
        # sample_positions[i] 表示 agent i 的当前位置（float坐标，用于插值观测）
        # routes[i] 表示 agent i 的历史轨迹（节点 id）
        # random speed factors随机速度因子（动态调整移动速度）
        self.current_node_indices = [0 for _ in range(self.num_agents)]
        self.dist_residuals = [0.0 for _ in range(self.num_agents)]
        self.sample_positions = self.start_positions.copy()
        self.routes = [[] for _ in range(self.num_agents)]
        self.random_speed_factors = np.random.rand(self.num_agents)
        # ==============================================

        self.frame_files=[]

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # 构建随机目标
        self.underlying_distrib = VTSPGaussian(n_targets=self.n_targets)
        self.ground_truth = self.get_ground_truth()
        self.high_info_idx = self.get_high_info_idx() if arg.high_info_thre else None

        # 初始化 GP
        self.gp_wrapper = GaussianProcessWrapper(self.n_targets, self.node_coords)
        self.curr_t = 0.0
        self.budget = self.budget_init
        self.visit_t = [[] for _ in range(self.n_targets)]

        # 如果有先验测量
        if arg.prior_measurement:
            node_prior = self.underlying_distrib.mean
            # 每个目标都加上先验(这里仅在第0个agent位置上测量也行)，也可以分别放在 agent 各处
            self.gp_wrapper.add_init_measures(add_t(node_prior, self.curr_t))
            self.gp_wrapper.update_gps()

        # 评估量
        self.node_feature = self.gp_wrapper.update_node_feature(self.curr_t)
        self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)
        self.cov_trace = self.gp_wrapper.eval_avg_cov_trace(self.curr_t, self.high_info_idx)
        self.unc, self.unc_list = self.gp_wrapper.eval_avg_unc(self.curr_t, self.high_info_idx, return_all=True)
        self.unc_sum, self.unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(self.unc_list, self.high_info_idx, return_all=True)
        self.JS, self.JS_list = self.gp_wrapper.eval_avg_JS(self.ground_truth, self.curr_t, return_all=True)
        self.KL, self.KL_list = self.gp_wrapper.eval_avg_KL(self.ground_truth, self.curr_t, return_all=True)

        # New - Occupancy
        occupancy = np.zeros((self.node_feature.shape[0], 1), dtype=float)
        for idx in self.current_node_indices:
            occupancy[idx, 0] = 1.0
        self.node_feature = np.concatenate((self.node_feature, occupancy), axis=1)

        # ma状态重置
        # 随机选择每个 agent 在 graph 中最近的那个节点为起点
        for i in range(self.num_agents):
            # 找到离 self.start_positions[i] 最近的 node
            dists = np.linalg.norm(self.node_coords - self.start_positions[i], axis=1)
            self.current_node_indices[i] = int(np.argmin(dists))
            self.dist_residuals[i] = 0.0
            self.sample_positions[i] = self.start_positions[i].copy()
            self.routes[i] = [self.current_node_indices[i]]
            self.random_speed_factors[i] = np.random.rand()

        return self.node_coords, self.graph, self.node_feature, self.budget

    def step(self, next_node_indices, global_step=0, eval_speed=None):
        """
        ma-step：核心仍然是单智能体的循环插值移动思想，并引入重复观测惩罚
        "CTDE"
        
        args:
            next_node_indices: 长度为 num_agents 的列表，每个元素为对应 agent 下一目标节点的索引
            global_step: 全局步数（用于计算 curriculum 系数 alpha）
            eval_speed: 如果非 None，则表示评估模式下使用固定速度；否则采用训练模式速度
        
        return:
            reward: 团队奖励（基于各目标不确定度下降的加权和，其中对同一目标重复观测按比例惩罚）
            done: 是否预算耗尽（True 表示 episode 结束）
            node_feature: 更新后的节点特征（来自 GP 更新）
            budget: 剩余预算
            metrics: 记录的一些时序指标（例如预算消耗等，仅在评估模式下使用）
        """
        sample_length = 0.1  # 固定采样间隔
        alpha = min(global_step // 1000 * 0.1, 1) if arg.curriculum else 1
        metrics = {'budget': [], 'rmse': [], 'jsd': [], 'jsdall': [], 'jsdstd': [],
                'unc': [], 'uncall': [], 'uncstd': []}
        
        # 保存旧的不确定度列表用于计算奖励
        old_unc_list = self.unc_list if self.unc_list is not None else np.zeros(self.n_targets, dtype=float)
        self._obs_counts = [0 for _ in range(self.n_targets)]# 初始化本次 step 的观测计数（每个目标被观测的 agent 数），用于重复观测惩罚
        
        # 对每个 agent 计算从当前节点到目标节点的剩余距离和运动方向
        remain_lengths = []
        directions = []
        for i in range(self.num_agents):
            curr_idx = self.current_node_indices[i]
            nxt_idx = next_node_indices[i]
            vec = self.node_coords[nxt_idx] - self.node_coords[curr_idx]
            d_len = np.linalg.norm(vec)
            remain_lengths.append(d_len)
            if d_len > 1e-9:
                directions.append(vec / d_len)
            else:
                directions.append(np.zeros(2))
        
        # 开始循环插值移动，直到所有 agent 到达目标节点
        done_move = [False] * self.num_agents
        while not all(done_move):
            # 计算每个 agent 尚需达到采样间隔的距离
            next_lengths = []
            for i in range(self.num_agents):
                if done_move[i]:
                    next_lengths.append(np.inf)
                else:
                    next_lengths.append(max(sample_length - self.dist_residuals[i], 0.0))
            # 取所有未到达 agent 中最小的采样距离
            min_next = min(next_lengths)
        
            # 每个 agent 实际本次移动距离 = min(min_next, remain_lengths[i])
            move_dists = []
            for i in range(self.num_agents):
                if done_move[i]:
                    move_dists.append(0.0)
                else:
                    move_dists.append(min(min_next, remain_lengths[i]))
            real_move = min(move_dists)
            if real_move < 1e-9:
                for i in range(self.num_agents):
                    if remain_lengths[i] < 1e-9:
                        done_move[i] = True
                break
        
            # 计算时间步长 steplen
            if eval_speed is not None:
                steplen = eval_speed * real_move
            else:
                # 训练模式下，采用所有 agent 随机速度因子均值
                mean_factor = np.mean(self.random_speed_factors)
                steplen = 0.3 * real_move * alpha * mean_factor # 初始设置是0.1
        
            # 更新全局时间和预算
            self.curr_t += real_move
            self.budget -= real_move
        
            # 目标整体移动
            self.underlying_distrib.step(steplen)
        
            # 更新各 agent 位置、剩余距离和采样残差
            for i in range(self.num_agents):
                if not done_move[i]:
                    self.sample_positions[i] += directions[i] * real_move
                    remain_lengths[i] -= real_move
                    self.dist_residuals[i] += real_move
                    # 如果 agent 到达目标节点或刚好达到采样间隔，则进行观测
                    if remain_lengths[i] < 1e-9 or abs(self.dist_residuals[i] - sample_length) < 1e-6:
                        self._agent_observe(i)  # 执行观测，并更新 _obs_counts
                        self.dist_residuals[i] = 0.0
                # 已到达目标的 agent 保持不动
        
            # 每次观测后更新 GP
            self.gp_wrapper.update_gps()
        
            # 记录评估指标（如果在评估模式）
            if eval_speed is not None:
                metrics['budget'].append(self.budget_init - self.budget)
                # 此处可以扩展记录 RMSE、JSD、UNC 等指标
        
        # 所有 agent 均到达目标节点，更新各 agent 当前节点和轨迹
        for i in range(self.num_agents):
            self.current_node_indices[i] = next_node_indices[i]
            self.routes[i].append(next_node_indices[i])
        
        # 最后再更新一次 GP
        self.gp_wrapper.update_gps()

        # 在 step() 计算完新的节点特征后:
        self.node_feature = self.gp_wrapper.update_node_feature(self.curr_t)
        occupancy = np.zeros((self.node_feature.shape[0], 1), dtype=float)
        for idx in self.current_node_indices:
            occupancy[idx, 0] = 1.0
        self.node_feature = np.concatenate((self.node_feature, occupancy), axis=1)

        
        # 计算新的不确定度
        new_unc, new_unc_list = self.gp_wrapper.eval_avg_unc(self.curr_t, self.high_info_idx, return_all=True)
        
        # 重要！！！计算奖励：对每个目标，若被观察到，则奖励 = (old_unc - new_unc) / (观测次数)，再累加
        r = 0.0
        for t in range(self.n_targets):
            if self._obs_counts[t] > 0:
                r += max(old_unc_list[t] - new_unc_list[t], 0.0) / self._obs_counts[t]
        reward = 5.0 * r - 0.1
        
        # 更新环境评估量
        self.unc, self.unc_list = new_unc, new_unc_list
        self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)
        self.JS, self.JS_list = self.gp_wrapper.eval_avg_JS(self.ground_truth, self.curr_t, return_all=True)
        self.KL, self.KL_list = self.gp_wrapper.eval_avg_KL(self.ground_truth, self.curr_t, return_all=True)
        self.cov_trace = self.gp_wrapper.eval_avg_cov_trace(self.curr_t, self.high_info_idx)
        self.unc_sum, self.unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(self.unc_list, self.high_info_idx, return_all=True)
        # self.node_feature = self.gp_wrapper.update_node_feature(self.curr_t)
        self.ground_truth = self.get_ground_truth()
        self.high_info_idx = self.get_high_info_idx() if arg.high_info_thre else None
        
        # 判断是否结束
        done = (self.budget <= 0.0)
        
        return reward, done, self.node_feature, self.budget, metrics


    def _agent_observe(self, agent_id):
        """
        当某个 agent 达到采样间隔或到达目标节点时，执行一次观测，
        同时记录该 agent 对每个目标是否有效观测（FOV=0.1）。
        """
        target_mean = self.underlying_distrib.mean  # 目标当前真实位置 (n_targets, 2)
        agent_pos = self.sample_positions[agent_id]
        for t in range(self.n_targets):
            dist_to_target = np.linalg.norm(agent_pos - target_mean[t])
            if dist_to_target < 0.08:
                measure_coord = target_mean[t].reshape(-1, 2)
                measure_value = 1.0
                self.visit_t[t].append(self.curr_t)
                self._obs_counts[t] += 1  # 记录该目标被该 agent 观测到
            else:
                measure_coord = agent_pos.reshape(-1, 2)
                measure_value = 0.0
            self.gp_wrapper.GPs[t].add_observed_point(add_t(measure_coord, self.curr_t), measure_value)


    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 40)
        x2 = np.linspace(0, 1, 40)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distrib.fn(x1x2)
        return ground_truth

    def get_high_info_idx(self):
        high_info_idx = []
        for i in range(self.n_targets):
            idx = np.argwhere(self.ground_truth[:, i] > arg.high_info_thre)
            high_info_idx += [idx.squeeze(1)]
        return high_info_idx

    def plot(self, routes, n, step, path, budget_list, rew_list, div_list):
        # 将 div_list 转换为 numpy 数组
        div_list = np.array(div_list)
        y_pred_sum = []
        plt.switch_backend('agg')
        # 整体画布大小（根据 target 数量调整）
        plt.figure(figsize=(self.n_targets * 2.8 + 3.6, 6))
        target_cmap = ['r', 'g', 'b', 'm', 'y', 'c', 'lightcoral', 'lightgreen', 'lightblue', 'orange', 'gold', 'pink']
        assert len(target_cmap) >= self.n_targets
        target_mean = self.underlying_distrib.mean

        # 将所有 agent 当前的位置构成数组，便于传给 GP.plot 绘制视野圈等
        agent_locs = np.array([self.node_coords[idx] for idx in self.current_node_indices])

        # 绘制每个 target 的图像（每个 GP 绘制一个子图）
        for i, gp in enumerate(self.gp_wrapper.GPs):
            y_pred = gp.plot(self.ground_truth, target_id=i, target_num=self.n_targets,
                            target_loc=target_mean, all_pred=y_pred_sum,
                            high_idx=self.high_info_idx, agent_loc=agent_locs)
            y_pred_sum.append(y_pred)

        # 定义每个 agent 的颜色
        agent_colors = ['cyan', 'magenta', 'yellow', 'blue', 'green', 'red']
        while len(agent_colors) < self.num_agents:
            agent_colors.append('black')

        # 绘制每个 agent 的轨迹
        for ag_idx, route in enumerate(routes):
            # route 为单个 agent 的节点索引列表
            points = [self.graph_ctrl.find_point_from_node(nd) for nd in route]
            x = [pt[0] for pt in points]
            y = [pt[1] for pt in points]
            # 绘制轨迹线段，alpha 随轨迹位置逐渐减弱以突出最新部分
            for i in range(len(x) - 1):
                alpha = max(0.02 * (i - len(x)) + 1, 0.1)
                plt.plot(x[i:i+2], y[i:i+2], color=agent_colors[ag_idx], linewidth=2, zorder=5, alpha=alpha)
            # 绘制起始和末端标记
            plt.scatter(x[0], y[0], color=agent_colors[ag_idx], s=30, zorder=10, marker='o', edgecolors='k')
            plt.scatter(x[-1], y[-1], color=agent_colors[ag_idx], s=18, zorder=10, label=f'Agent {ag_idx}')
        
        plt.legend()
        
        # 添加 JSDiv 子图：这里使用 subplot 的方式，将画布分成两行，第二行绘制 JSDiv 曲线
        # 此处我们假设 subplot 的总列数为 self.n_targets+1（与单智能体版本一致）
        ax1 = plt.subplot(2, self.n_targets+1, 2 * self.n_targets + 2)
        plt.grid(linestyle='--')
        plt.xlim(0, self.budget_init)
        plt.ylim(0, 1.4)
        # div_list 的每一列代表一个 target 的 JS divergence 曲线
        for target_div in range(div_list.shape[1]):
            plt.plot(budget_list, div_list[:, target_div], alpha=0.5, color=target_cmap[target_div])
        plt.ylabel('JSDiv')
        plt.title('{:g}/{:g} Reward:{:.3f}'.format(self.budget_init - self.budget, self.budget_init, rew_list[-1]))
        
        plt.tight_layout()
        plt.savefig(f"{path}/{n}_{step}_samples.png", dpi=150)
        frame = '{}/{}_{}_samples.png'.format(path, n, step, self.graph_size)
        self.frame_files.append(frame)


if __name__ == '__main__':
    pass

