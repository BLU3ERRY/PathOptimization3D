#!/usr/bin/python3
# main.py

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
from Source.gradient_descent import optimize
from Source.cost_functions import (
    calculate_lap_time_with_constraints,
    convert_sectors_to_racing_line,
)
from Source.constants import (
    TRACK_WIDTH,
    DRIVE_WIDTH,
    N_SECTORS,
    N_OPTIMIZABLE_SECTORS,
    N_ITERATIONS,
    BOUNDARIES,
    VEHICLE_HEIGHT,
)

def load_reference_path(file_path: str) -> np.ndarray:
    """Load the reference path data from a CSV file."""
    try:
        data = pd.read_csv(file_path, header=None)
        reference_path = data.values.astype(float)
        # print(reference_path)
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

def initialize_plotting() -> tuple:
    """Initialize real-time plotting."""
    plt.ion()
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    plt.title("Real-Time Racing Line Optimization")
    return fig, axis

def update_plot(
    current_solution: List[float],
    iteration: int,
    racing_line_plot,
    axis,
    drive_inside_sectors,
    drive_outside_sectors,
    fixed_start_point,
    fixed_end_point,
) -> None:
    """Update the racing line plot in real-time."""
    try:
        racing_line = convert_sectors_to_racing_line(
            current_solution,
            drive_inside_sectors,
            drive_outside_sectors,
            fixed_start_point,
            fixed_end_point,
        )
        racing_line = np.array(racing_line)
        racing_line_plot.set_data(racing_line[:, 0], racing_line[:, 1])
        racing_line_plot.set_3d_properties(racing_line[:, 2])
        axis.relim()
        axis.autoscale_view()
        plt.draw()
        plt.pause(0.001)
    except Exception as e:
        print(f"Error in update_plot callback: {e}")

def cost_function_wrapper(
    sectors: List[float],
    drive_inside_sectors: np.ndarray,
    drive_outside_sectors: np.ndarray,
    fixed_start_point: np.ndarray,
    fixed_end_point: np.ndarray
) -> float:
    """Wrapper for the cost function."""
    return calculate_lap_time_with_constraints(
        sectors,
        drive_inside_sectors,
        drive_outside_sectors,
        fixed_start_point,
        fixed_end_point,
    )

def main() -> None:
    start = int(time.time())
    """Main function to execute the optimization of the racing line."""
    try:
        #불러오기
        reference_path_file = os.path.join(data_dir, 'reference_path.csv')
        reference_path = load_reference_path(reference_path_file)
        
        #스플라인 전처리
        reference_path = preprocess_reference_path(reference_path)

        plot_3d_lines([reference_path], title="Reference Path (Center Line)")

        #트랙은 8m 너비. 이 기준으로 안쪽 지점과 바깥쪽 지점 계산.
        rightside_points, leftside_points, normals = compute_track_boundaries(
            reference_path, TRACK_WIDTH
        )

        #안전상 주행하는 한계는 트랙보다 작은 범위(DRIVE_WIDTH)로만 제한.
        drive_rightside_points, drive_leftside_points, _ = compute_track_boundaries(
            reference_path, DRIVE_WIDTH
        )

        # Apply z-axis offset to drive boundaries
        drive_rightside_points[:, 2] += VEHICLE_HEIGHT / 2
        drive_leftside_points[:, 2] += VEHICLE_HEIGHT / 2

        #입력 레퍼런스 지점들은 750개 이상의 지점들로 이루어짐.
        #이것을 N_SECTOR개 만으로 샘플링을 함.
        #sectors_indices는 샘플링된 지점들의 번호들의 모임.
        sectors_indices = np.linspace(0, len(reference_path) - 1, N_SECTORS, dtype=int)

        #샘플링된 우측 주행지점, 좌측 주행 지점, 그리고 중앙 지점들의 좌표.
        drive_rightside_sectors = drive_rightside_points[sectors_indices]
        drive_leftside_sectors = drive_leftside_points[sectors_indices]
        mid_sectors = reference_path[sectors_indices]

        # Apply z-axis offset to fixed start and end points
        fixed_start_point = mid_sectors[0].copy()
        fixed_end_point = mid_sectors[-1].copy()
        fixed_start_point[2] += VEHICLE_HEIGHT / 2
        fixed_end_point[2] += VEHICLE_HEIGHT / 2

        # Plot with updated fixed_start_point and fixed_end_point
        plot_track_boundaries_with_normals(
            reference_path,
            rightside_points,
            leftside_points,
            drive_rightside_points,
            drive_leftside_points,
            normals,
            fixed_start_point,
            fixed_end_point,
        )

        plot_sectors_with_boundaries(
            reference_path,
            drive_rightside_sectors,
            drive_leftside_sectors,
            mid_sectors
        )

        fig, axis = initialize_plotting()
        axis.plot(
            reference_path[:, 0],
            reference_path[:, 1],
            reference_path[:, 2],
            color='#000000',
            linestyle='dashed',
            label='Center Line'
        )
        axis.plot(
            drive_rightside_points[:, 0],
            drive_rightside_points[:, 1],
            drive_rightside_points[:, 2],
            color='#880000',
            linestyle='dashed',
            label='Drive Rightside Boundary'
        )
        axis.plot(
            drive_leftside_points[:, 0],
            drive_leftside_points[:, 1],
            drive_leftside_points[:, 2],
            color='#008800',
            linestyle='dashed',
            label='Drive Leftside Boundary'
        )
        racing_line_plot, = axis.plot([], [], [], 'b-', label='Racing Line')
        axis.legend()

        def callback(current_solution: List[float], iteration: int) -> None:
            update_plot(
                current_solution,
                iteration,
                racing_line_plot,
                axis,
                drive_rightside_sectors,
                drive_leftside_sectors,
                fixed_start_point,
                fixed_end_point,
            )

        cost_function = lambda sectors: cost_function_wrapper(
            sectors,
            drive_rightside_sectors,
            drive_leftside_sectors,
            fixed_start_point,
            fixed_end_point,
        )

        #0.5: 경로의 중앙을 주행하는 것을 나타냄.
        #먼저 경로의 중앙을 주행하는 경우를 출력한다.
        sample_sectors = [0.5] * N_OPTIMIZABLE_SECTORS
        sample_lap_time = cost_function(sample_sectors)
        print(f"Sample lap time: {sample_lap_time}")

        #PSO 최적화 수행
        global_solution, global_evaluation, global_history, evaluation_history = optimize(
            cost_function=cost_function,
            n_dimensions=N_OPTIMIZABLE_SECTORS,
            boundaries=BOUNDARIES,
            n_iterations=N_ITERATIONS,
            verbose=True,
            callback=callback,
        )

        plt.ioff()

        racing_line = np.array(convert_sectors_to_racing_line(
            global_solution,
            drive_rightside_sectors,
            drive_leftside_sectors,
            fixed_start_point,
            fixed_end_point,
        ))

        fig_final = plt.figure()
        axis_final = fig_final.add_subplot(111, projection='3d')
        plt.title("Final Racing Line within Drive Boundaries")
        axis_final.plot(
            racing_line[:, 0],
            racing_line[:, 1],
            racing_line[:, 2],
            color='#000000',
            linestyle='dashed',
            label='Racing Line'
        )
        axis_final.plot(
            rightside_points[:, 0],
            rightside_points[:, 1],
            rightside_points[:, 2],
            color='#FF0000',
            linestyle='dashed',
            label='Track Rightside Boundary'
        )
        axis_final.plot(
            leftside_points[:, 0],
            leftside_points[:, 1],
            leftside_points[:, 2],
            color='#00FF00',
            linestyle='dashed',
            label='Track Leftside Boundary'
        )
        axis_final.plot(
            drive_rightside_points[:, 0],
            drive_rightside_points[:, 1],
            drive_rightside_points[:, 2],
            color='#880000',
            linestyle='dashed',
            label='Drive Leftside Boundary'
        )
        axis_final.plot(
            drive_leftside_points[:, 0],
            drive_leftside_points[:, 1],
            drive_leftside_points[:, 2],
            color='#008800',
            linestyle='dashed',
            label='Drive Rightside Boundary'
        )
        axis_final.scatter(
            racing_line[0, 0],
            racing_line[0, 1],
            racing_line[0, 2],
            color='k',
            marker='o',
            label='Start Point'
        )
        axis_final.scatter(
            racing_line[-1, 0],
            racing_line[-1, 1],
            racing_line[-1, 2],
            color='k',
            marker='X',
            label='End Point'
        )
        axis_final.legend()
        plt.show()


        racing_line = np.concatenate((racing_line, np.expand_dims(np.array([0.5] + global_solution + [0.5]), axis=-1)), axis=-1)
        racing_line_file = os.path.join(data_dir, 'racing_line_midpoints_gdescent.csv')
        # print(racing_line)
        # df_racing_line = pd.DataFrame(racing_line)
        with open(racing_line_file, "w") as f:
            for ptinfo in racing_line:
                f.write(", ".join(map(str, ptinfo)) + "\n")
        # df_racing_line.to_csv(racing_line_file, index=False)
        print(f"Final racing line midpoints saved to '{racing_line_file}'.")

        plot_lap_time_history(evaluation_history)
        print("All tasks completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
    print("***run time(sec) :", int(time.time()) - start)
    
if __name__ == "__main__":
    main()
