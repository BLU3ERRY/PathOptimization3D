# pso_optimization.py

import numpy as np
from typing import Callable, Tuple, List

def optimize(
    cost_function: Callable[[List[float]], float],
    n_dimensions: int,
    boundaries: List[Tuple[float, float]],
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
    learning_rate = 0.1
    delta = 0.01

    #도로 중앙 (0.5)을 달리는 것으로 초기화
    position = np.full((n_dimensions), 0.5, dtype=float)
    
    score_curr = cost_function(position.tolist())
    
    if verbose:
        print(f"Initial score: {score_curr}")
    
    d_mat = delta*np.eye(n_dimensions, dtype=float)

    evaluation_history = []

    #본 계산 작업
    for iteration in range(n_iterations):

        #경사값 구하기
        score_plus_epsilon = np.array([cost_function((position + d_mat[idx]).tolist()) for idx in range(n_dimensions)])
        gradient = (score_plus_epsilon - score_curr)/delta

        #경사 하강
        position -= learning_rate*gradient
        position = np.clip(position, [b[0] for b in boundaries], [b[1] for b in boundaries])

        score_curr = cost_function(position.tolist())

        if callback:
            callback(position, iteration)

        if verbose and (iteration % 5 == 4 or iteration == n_iterations - 1):
            print(f"Iteration {iteration + 1}/{n_iterations}, Current Score: {score_curr}")

        evaluation_history.append(score_curr)

    return list(position), score_curr, None, evaluation_history
