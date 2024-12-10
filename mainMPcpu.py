import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Callable
import time

# Add the 'Source' directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, 'Source')
data_dir = os.path.join(current_dir, 'Data')

if source_dir not in sys.path:
    sys.path.append(source_dir)

from Source.utilities import (
    plot_3d_lines,
    compute_track_boundaries,
    smooth_reference_path,
    plot_track_boundaries_with_normals,
    plot_sectors_with_boundaries,
    plot_lap_time_history,
)
from Source.pso_optimization import optimize
from Source.cost_functions import (
    calculate_lap_time_with_constraints,
    convert_sectors_to_racing_line,
)
from Source.constants import (
    TRACK_WIDTH,
    DRIVE_WIDTH,
    N_SECTORS,
    N_OPTIMIZABLE_SECTORS,
    N_PARTICLES,
    N_ITERATIONS,
    BOUNDARIES,
    VEHICLE_HEIGHT,
)

def load_reference_path(file_path: str) -> np.ndarray:
    """Load the reference path data from a CSV file."""
    try:
        data = pd.read_csv(file_path, header=None)
        reference_path = data.values.astype(float)
        if len(reference_path) == 0:
            raise ValueError("Reference path is empty.")
        return reference_path
    except FileNotFoundError:
        raise RuntimeError(f"File '{file_path}' not found.")
    except Exception as e:
        raise RuntimeError(f"Failed to load reference path: {e}")

def preprocess_reference_path(reference_path: np.ndarray) -> np.ndarray:
    """Smooth the reference path and ensure it is suitable for processing."""
    try:
        smoothed_path = smooth_reference_path(reference_path, smooth_factor=1.0)
        return smoothed_path
    except Exception as e:
        print(f"Warning: {e}. Using original reference path.")
        return reference_path

def vectorized_cost_function(
    solutions: np.ndarray,
    drive_inside_sectors: np.ndarray,
    drive_outside_sectors: np.ndarray,
    fixed_start_point: np.ndarray,
    fixed_end_point: np.ndarray
) -> np.ndarray:
    """
    Vectorized cost function evaluation for multiple solutions.
    """
    sampled_indices = np.linspace(0, len(drive_inside_sectors) - 1, solutions.shape[1], dtype=int)
    sampled_drive_inside = drive_inside_sectors[sampled_indices]
    sampled_drive_outside = drive_outside_sectors[sampled_indices]

    diff_inside = solutions - sampled_drive_inside[:, 0]
    diff_outside = solutions - sampled_drive_outside[:, 0]
    lap_times = np.sum(diff_inside**2 + diff_outside**2, axis=1)
    return lap_times

def save_racing_line_midpoints(global_solution: List[float], drive_rightside_sectors: np.ndarray, drive_leftside_sectors: np.ndarray, fixed_start_point: np.ndarray, fixed_end_point: np.ndarray) -> None:
    """
    Save the racing line midpoints to a CSV file.
    """
    racing_line = np.array(convert_sectors_to_racing_line(
        global_solution,
        drive_rightside_sectors,
        drive_leftside_sectors,
        fixed_start_point,
        fixed_end_point
    ))

    racing_line = np.concatenate((racing_line, np.expand_dims(np.array([0.5] + global_solution + [0.5]), axis=-1)), axis=-1)
    
    racing_line_file = os.path.join(data_dir, 'racing_line_midpointsMP.csv')

    # Save using pandas DataFrame for better performance
    df_racing_line = pd.DataFrame(racing_line)
    df_racing_line.to_csv(racing_line_file, index=False, header=False)
    
    print(f"Final racing line midpoints saved to '{racing_line_file}'.")

def main() -> None:
    start = time.time()
    try:
        # Loading and preprocessing data
        reference_path_file = os.path.join(data_dir, 'reference_path.csv')
        reference_path = load_reference_path(reference_path_file)
        reference_path = preprocess_reference_path(reference_path)

        plot_3d_lines([reference_path], title="Reference Path (Center Line)")

        rightside_points, leftside_points, normals = compute_track_boundaries(reference_path, TRACK_WIDTH)
        drive_rightside_points, drive_leftside_points, _ = compute_track_boundaries(reference_path, DRIVE_WIDTH)

        drive_rightside_points[:, 2] += VEHICLE_HEIGHT / 2
        drive_leftside_points[:, 2] += VEHICLE_HEIGHT / 2

        sectors_indices = np.linspace(0, len(reference_path) - 1, N_SECTORS, dtype=int)
        drive_rightside_sectors = drive_rightside_points[sectors_indices]
        drive_leftside_sectors = drive_leftside_points[sectors_indices]
        mid_sectors = reference_path[sectors_indices]

        fixed_start_point = mid_sectors[0].copy()
        fixed_end_point = mid_sectors[-1].copy()
        fixed_start_point[2] += VEHICLE_HEIGHT / 2
        fixed_end_point[2] += VEHICLE_HEIGHT / 2

        # Sample sectors for test run
        sample_sectors = [0.5] * N_OPTIMIZABLE_SECTORS
        cost_function = lambda sectors: vectorized_cost_function(
            np.array(sectors).reshape(1, -1),
            drive_rightside_sectors,
            drive_leftside_sectors,
            fixed_start_point,
            fixed_end_point
        )[0]

        sample_lap_time = cost_function(sample_sectors)
        print(f"Sample lap time: {sample_lap_time}")

        # PSO optimization
        global_solution, global_evaluation, global_history, evaluation_history = optimize(
            cost_function=cost_function,
            n_dimensions=N_OPTIMIZABLE_SECTORS,
            boundaries=BOUNDARIES,
            n_particles=N_PARTICLES,
            n_iterations=N_ITERATIONS,
            verbose=True,
        )

        save_racing_line_midpoints(global_solution, drive_rightside_sectors, drive_leftside_sectors, fixed_start_point, fixed_end_point)

        plot_lap_time_history(evaluation_history)
        print("All tasks completed successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    print(f"***run time(sec): {time.time() - start:.2f}")

if __name__ == "__main__":
    main()
