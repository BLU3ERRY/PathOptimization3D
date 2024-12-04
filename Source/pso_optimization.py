import numpy as np
from typing import Callable, Tuple, List

def optimize(
    cost_function: Callable[[List[float]], float],
    n_dimensions: int,
    boundaries: List[Tuple[float, float]],
    n_particles: int,
    n_iterations: int,
    verbose: bool = False,
    callback: Callable[[List[float], int], None] = None
) -> Tuple[List[float], float, List[List[float]], List[float]]:
    """
    Optimize the given cost function using Particle Swarm Optimization (PSO).
    """
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter

    swarm_position = np.array([
        [np.random.uniform(boundaries[d][0], boundaries[d][1]) for d in range(n_dimensions)]
        for _ in range(n_particles)
    ])
    swarm_velocity = np.zeros((n_particles, n_dimensions))
    swarm_best_position = np.copy(swarm_position)
    swarm_best_score = np.array([cost_function(p.tolist()) for p in swarm_position])

    global_best_index = np.argmin(swarm_best_score)
    global_best_position = list(swarm_position[global_best_index])
    global_best_score = swarm_best_score[global_best_index]

    global_history = [list(global_best_position)]
    evaluation_history = [global_best_score]

    if verbose:
        print(f"Initial global best score: {global_best_score}")

    for iteration in range(n_iterations):
        for i in range(n_particles):
            r1 = np.random.rand(n_dimensions)
            r2 = np.random.rand(n_dimensions)
            swarm_velocity[i] = (
                w * swarm_velocity[i]
                + c1 * r1 * (swarm_best_position[i] - swarm_position[i])
                + c2 * r2 * (global_best_position - swarm_position[i])
            )
            swarm_position[i] += swarm_velocity[i]
            for d in range(n_dimensions):
                swarm_position[i][d] = np.clip(swarm_position[i][d], boundaries[d][0], boundaries[d][1])
            score = cost_function(list(swarm_position[i]))
            if score < swarm_best_score[i]:
                swarm_best_score[i] = score
                swarm_best_position[i] = swarm_position[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = list(swarm_position[i])

        global_history.append(list(global_best_position))
        evaluation_history.append(global_best_score)

        if callback:
            callback(global_best_position, iteration)

        if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
            print(f"Iteration {iteration+1}/{n_iterations}, Best Score: {global_best_score}")

    return global_best_position, global_best_score, global_history, evaluation_history
