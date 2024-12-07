# pso_optimization.py

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
    --------------------------------
    cost_function: 함수, 랩 타임을 측정하는 척도, 최적화 대상.
    n_dimensions: int, 최적화 대상 SECTOR의 개수
    boundaries: sector의 값의 최소, 최대값을 담은 tuple의 list. 일반적으로 [(0.0, 1.0), ..., (0.0, 1.0)].
    n_particles: int, 각 SECTOR 마다 적용하는 swarm particle의 개수.
    n_iterations: int, 최적화 계산 반복 횟수
    """
    inertia_weight = 0.5  # Inertia weight
    cognitive_param = 1.5  # Cognitive parameter
    social_param = 1.5  # Social parameter

    #swarm은 여러 개의 particle의 모임.
    #swarm position의 각 i번째 행은 i번째 particle이며, 각 particle은 n_dimension개의 sector의 값 (0.0부터 1.0까지)이 담겨있음.
    #(n_particles, n_dimensions), float, 초기화는 무작위로 설정.
    swarm_position = np.array([
        [np.random.uniform(boundaries[d][0], boundaries[d][1]) for d in range(n_dimensions)]
        for _ in range(n_particles)
    ])

    #0으로 초기화.
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

    #본 계산 작업
    for iteration in range(n_iterations):
        for i in range(n_particles):
            r1 = np.random.rand(n_dimensions)
            r2 = np.random.rand(n_dimensions)
            swarm_velocity[i] = (
                inertia_weight * swarm_velocity[i]
                + cognitive_param * r1 * (swarm_best_position[i] - swarm_position[i])
                + social_param * r2 * (global_best_position - swarm_position[i])
            )
            swarm_position[i] += swarm_velocity[i]
            swarm_position[i] = np.clip(swarm_position[i], [b[0] for b in boundaries], [b[1] for b in boundaries])
            score = cost_function(swarm_position[i].tolist())
            if score < swarm_best_score[i]:
                swarm_best_score[i] = score
                swarm_best_position[i] = swarm_position[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm_position[i].tolist()

        global_history.append(list(global_best_position))
        evaluation_history.append(global_best_score)

        if callback:
            callback(global_best_position, iteration)

        if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
            print(f"Iteration {iteration + 1}/{n_iterations}, Best Score: {global_best_score}")

    return global_best_position, global_best_score, global_history, evaluation_history
