import numpy as np
import math
from utils.tsp_controller import TSPSolver


class VTSPGaussian:
    def __init__(self, n_targets=2):
        self.n_targets = n_targets
        self.n_tsp_nodes = 50
        self.tsp_coord = self.get_tsp_nodes()
        self.tsp_idx = [0] * self.n_targets
        self.mean = self.tsp_coord[:, 0, :]
        self.sigma = np.array([0.1] * self.n_targets)
        self.max_value = 1 / (2 * np.pi * self.sigma ** 2)
        self.trajectories = [self.mean.copy()]

    def get_tsp_nodes(self):
        tsp_solver = TSPSolver()
        coord = np.random.rand(self.n_targets, self.n_tsp_nodes, 2)
        for i in range(self.n_targets):
            index = tsp_solver.run_solver(coord[i])
            coord[i] = coord[i][index]
        return coord

    def step(self, steplen):
        """Moves targets along their TSP paths for a given steplen."""
        if steplen <= 0:
             # If no movement length, just return current mean
             # self.trajectories += [self.mean.copy()] # Optional: record position even if no movement
             return self.mean

        for i in range(self.n_targets):
            remaining_steplen_for_agent = steplen
            # Continue moving as long as there's budget left for this agent in this step
            while remaining_steplen_for_agent > 1e-9:
                current_node_idx = self.tsp_idx[i]
                # Calculate next node index with wrap-around using modulo
                next_node_idx = (current_node_idx + 1) % self.n_tsp_nodes # Use self.n_tsp_nodes

                current_tsp_coord = self.tsp_coord[i, current_node_idx, :]
                next_tsp_coord = self.tsp_coord[i, next_node_idx, :]

                # Vector from current agent position to the next TSP node
                vec_to_next_tsp_node = next_tsp_coord - self.mean[i]
                dist_to_next_tsp_node = np.linalg.norm(vec_to_next_tsp_node)

                # If agent is already (almost) at the next TSP node
                if dist_to_next_tsp_node < 1e-9:
                    # Consider agent as having reached this node
                    # Update the target node index and continue from there in the next iteration
                    self.tsp_idx[i] = next_node_idx
                    # Don't consume steplen here, let the next while loop iteration handle movement
                    continue # Re-evaluate from the new tsp_idx

                # Determine how much to move in this iteration
                move_distance = min(remaining_steplen_for_agent, dist_to_next_tsp_node)

                # Move the agent
                # Check if dist_to_next_tsp_node is not zero before division
                if dist_to_next_tsp_node > 1e-9:
                    self.mean[i] += vec_to_next_tsp_node * (move_distance / dist_to_next_tsp_node)
                # else: agent is already at the node, position doesn't change relative to it

                # Consume the budget used
                remaining_steplen_for_agent -= move_distance

                # Check if the agent reached or passed the target TSP node in this move
                # Use a small tolerance (1e-9) for floating point comparison
                if dist_to_next_tsp_node <= move_distance + 1e-9:
                    # Agent reached the node, update the index for the next iteration
                    self.tsp_idx[i] = next_node_idx

        self.trajectories += [self.mean.copy()]
        return self.mean

    def fn(self, X):
        y = np.zeros((X.shape[0], self.n_targets))
        row_mat, col_mat = X[:, 0], X[:, 1]
        for target_id in range(self.n_targets):
            gaussian_mean = self.mean[target_id]
            sigma_x1 = sigma_x2 = self.sigma[target_id]
            covariance = 0
            r = covariance / (sigma_x1 * sigma_x2)
            coefficients = 1 / (2 * math.pi * sigma_x1 * sigma_x2 * np.sqrt(1 - math.pow(r, 2)))
            p1 = -1 / (2 * (1 - math.pow(r, 2)))
            px = np.power((row_mat - gaussian_mean[0]) / sigma_x1, 2)
            py = np.power((col_mat - gaussian_mean[1]) / sigma_x2, 2)
            pxy = 2 * r * (row_mat - gaussian_mean[0]) * (col_mat - gaussian_mean[1]) / (sigma_x1 * sigma_x2)
            distribution_matrix = coefficients * np.exp(p1 * (px - pxy + py))
            y[:, target_id] += distribution_matrix
        y /= self.max_value
        return y



if __name__ == '__main__':
    pass
