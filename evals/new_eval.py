# eval_driver.py

import csv
import os
import ray
import torch
import numpy as np
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network import AttentionNet
from new_eval_worker import WorkerEval # 确保这是修改后的 eval_worker
from arguments import arg_eval


def main(config=None):
    """
    多智能体评估驱动函数，现在支持多种方法并行评估。
    """
    if config:
        arg_eval.graph_size, arg_eval.history_size, arg_eval.target_size, arg_eval.target_speed = config

    os.makedirs(arg_eval.result_path, exist_ok=True)
    device = torch.device('cuda') if arg_eval.use_gpu_driver and torch.cuda.is_available() else torch.device('cpu')
    local_device = torch.device('cuda') if arg_eval.use_gpu_runner and torch.cuda.is_available() else torch.device('cpu')

    # --- 模型加载 (仅当评估 MODEL 方法时需要) ---
    global_network = AttentionNet(arg_eval.embedding_dim).to(device)
    weights = None
    try:
        checkpoint = torch.load(f'{arg_eval.model_path}/checkpoint.pth', map_location=device)
        global_network.load_state_dict(checkpoint['model'])
        print(f'Successfully loaded model: {arg_eval.run_name}...')
        if device != local_device:
            weights = global_network.to(local_device).state_dict()
            global_network.to(device)
        else:
            weights = global_network.state_dict()
    except FileNotFoundError:
        print(f"Warning: Model file not found at {arg_eval.model_path}/checkpoint.pth. 'MODEL' method will be skipped.")
        weights = None

    # --- 初始化 Ray ---
    if not ray.is_initialized():
        # 获取项目根目录 (evals/ 的上一级目录)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        ray.init(runtime_env={"working_dir": project_root})

    # --- 定义要运行的评估方法 ---
    methods_to_run = ['MODEL', 'AUCTION', 'GREEDY', 'COVERAGE', 'RANDOM']
    if weights is None:
        methods_to_run.remove('MODEL') # 如果模型没加载成功，则跳过

    # --- 存储所有方法的结果 ---
    all_metrics_results = []
    all_traj_results = []

    # --- 循环评估每种方法 ---
    for method in methods_to_run:
        print(f"\n{'='*20} Running Evaluation for Method: {method.upper()} {'='*20}\n")
        
        # 为当前方法重置性能指标列表
        perf_metrics = {n: [] for n in WorkerEval.METRIC_NAMES}
        traj_data_for_method = []
        
        meta_runners = [Runner.remote(i) for i in range(arg_eval.num_meta)]
        curr_test = 1

        try:
            while curr_test <= arg_eval.num_eval:
                jobList = []
                # 下发评估任务
                for i, meta_agent in enumerate(meta_runners):
                    if curr_test > arg_eval.num_eval: break
                    # 将方法名称 (method) 传递给 worker
                    jobList.append(meta_agent.job.remote(weights, curr_test, config, method))
                    curr_test += 1

                # 等待所有 meta 完成
                done_id, jobList = ray.wait(jobList, num_returns=len(jobList))
                done_jobs = ray.get(done_id)

                # 收集结果
                for job in done_jobs:
                    if not job: continue
                    metrics, trajectory = job

                    # 仅当 metrics 不为空时，记录
                    if not metrics: continue

                    for n in WorkerEval.METRIC_NAMES:
                        if n in metrics and metrics[n] is not None:
                            perf_metrics[n].append(metrics[n])
                    
                    if trajectory:
                        traj_data_for_method.extend(trajectory)

            # --- 评估结束，统计当前方法的最终结果 ---
            perf_data = []
            for n in WorkerEval.METRIC_NAMES:
                if len(perf_metrics[n]) > 0:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                else:
                    perf_data.append(np.nan)
            
            # 打印当前方法的结果
            print(f"\n--- Results for Method: {method} ---")
            print(
                f'Graph {arg_eval.graph_size}, History {arg_eval.history_size}, '
                f'#T {arg_eval.target_size}, Budget {arg_eval.budget_size}, K {arg_eval.k_size}, Speed {1/arg_eval.target_speed}'
            )
            for i, name in enumerate(WorkerEval.METRIC_NAMES):
                print(f'{name}:\t{perf_data[i]:.4f}')
            
            # 保存当前方法的结果以备最终写入CSV
            # Metric
            num_runs = len(perf_metrics.get('avgjsd', []))
            for i in range(num_runs):
                row = {'method': method}
                for name in WorkerEval.METRIC_NAMES:
                    row[name] = perf_metrics[name][i] if i < len(perf_metrics.get(name,[])) else np.nan
                all_metrics_results.append(row)
            
            # Trajectory
            for budget, unc, jsd, rmse in traj_data_for_method:
                all_traj_results.append({
                    'method': method, 'budget': budget, 'unc': unc, 'jsd': jsd, 'rmse': rmse
                })

        except KeyboardInterrupt:
            print(f"CTRL_C pressed during {method} evaluation. Killing remote workers.")
            for a in meta_runners: ray.kill(a)
        finally:
            # 清理当前方法的 runners
            for a in meta_runners: ray.kill(a)
            time.sleep(2) # 等待 ray actor 终止

    # --- 所有方法评估完毕，统一保存到 CSV ---
    if arg_eval.save_results and (all_metrics_results or all_traj_results):
        print("\n--- Saving all results to CSV files ---")
        os.makedirs(arg_eval.result_path + '/all_methods', exist_ok=True)
        
        # --- 写 metric ---
        metric_filename = (
            f'{arg_eval.result_path}/all_methods/'
            f'tgt{arg_eval.target_size}_s{round(1/arg_eval.target_speed)}_b{arg_eval.budget_size}_g{arg_eval.graph_size}_metric.csv'
        )
        metric_field_names = ['method'] + WorkerEval.METRIC_NAMES
        with open(metric_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metric_field_names)
            writer.writeheader()
            writer.writerows(all_metrics_results)
        print(f"Metrics saved to {metric_filename}")

        # --- 写 traj ---
        traj_filename = (
            f'{arg_eval.result_path}/all_methods/'
            f'tgt{arg_eval.target_size}_s{round(1/arg_eval.target_speed)}_b{arg_eval.budget_size}_g{arg_eval.graph_size}_traj.csv'
        )
        traj_field_names = ['method', 'budget', 'unc', 'jsd', 'rmse']
        with open(traj_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=traj_field_names)
            writer.writeheader()
            writer.writerows(all_traj_results)
        print(f"Trajectories saved to {traj_filename}")

    if ray.is_initialized():
        ray.shutdown()


@ray.remote(num_cpus=1, num_gpus=len(arg_eval.cuda_devices) / arg_eval.num_meta if arg_eval.use_gpu_runner else 0)
class Runner:
    """
    在 eval 中对应的多智能体 Runner (不变)
    """
    def __init__(self, meta_id):
        self.meta_id = meta_id
        self.device = torch.device('cuda' if arg_eval.use_gpu_runner and torch.cuda.is_available() else 'cpu')
        self.local_net = AttentionNet(arg_eval.embedding_dim).to(self.device)

    def set_weights(self, weights):
        if weights:
            self.local_net.load_state_dict(weights)

    def job(self, global_weights, eval_number, config=None, method='MODEL'):
        """
        评估任务：接收全局权重和评估方法，执行 WorkerEval。
        """
        print(f'meta-{self.meta_id:02} | method: {method} | eval: {eval_number} starts')
        self.set_weights(global_weights)

        # 是否存图 (仅对 MODEL 方法存图)
        save_img = method == 'MODEL' and arg_eval.save_img_gap != 0 and (eval_number % arg_eval.save_img_gap == 0)

        worker = WorkerEval(
            meta_id=self.meta_id,
            local_net=self.local_net,
            global_step=eval_number,
            device=self.device,
            save_image=save_img,
            config=config,
            eval_method=method # 传递评估方法
        )
        metrics, trajectory = worker.run_episode(eval_number)
        return metrics, trajectory


if __name__ == '__main__':
    main()