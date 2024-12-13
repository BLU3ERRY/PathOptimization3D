# cost_functions.py

from typing import List
import numpy as np
from constants import WHEEL_SPACING, STEERING_RANGE, MAX_SPEED
from math import radians, tan

def calculate_lap_time_with_constraints(
    sectors: List[float],
    drive_inside_sectors: np.ndarray,
    drive_outside_sectors: np.ndarray,
    fixed_start_point: np.ndarray,
    fixed_end_point: np.ndarray
) -> float:
    """
    Calculate lap time based on the racing line sectors.
    Returns lap time in seconds if constraints are satisfied; otherwise, returns infinity.
    """
    epsilon = 1e-6
    racing_line = convert_sectors_to_racing_line(
        sectors,
        drive_inside_sectors,
        drive_outside_sectors,
        fixed_start_point,
        fixed_end_point
    )
    if not racing_line:
        return np.inf
    racing_line = np.array(racing_line)
    distances = np.linalg.norm(np.diff(racing_line, axis=0), axis=1)
    total_distance = np.sum(distances)
    lap_time = total_distance / MAX_SPEED

    for i, point in enumerate(racing_line):
        if i < len(drive_inside_sectors):
            drive_inside = drive_inside_sectors[i]
            drive_outside = drive_outside_sectors[i]
        else:
            drive_inside = drive_inside_sectors[-1]
            drive_outside = drive_outside_sectors[-1]
        vec_boundary = drive_outside - drive_inside
        vec_point = point - drive_inside
        proj_length = np.dot(vec_point, vec_boundary) / np.dot(vec_boundary, vec_boundary)
        if proj_length < -epsilon or proj_length > 1.0 + epsilon:
            return np.inf

    max_steering_angle_rad = radians(abs(STEERING_RANGE[1]))
    if max_steering_angle_rad == 0:
        return np.inf
    min_turn_radius = WHEEL_SPACING / tan(max_steering_angle_rad + epsilon)
    if min_turn_radius == 0:
        return np.inf
    max_curvature = 1.0 / min_turn_radius

    for j in range(1, len(racing_line) - 1):
        p_prev = racing_line[j - 1]
        p_curr = racing_line[j]
        p_next = racing_line[j + 1]
        curvature = calculate_curvature(p_prev, p_curr, p_next)
        lap_time += 0.1*((abs(curvature) - max_curvature) if curvature > max_curvature else 0.0)

    return lap_time

def calculate_curvature(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> float:
    """Calculate the curvature given three consecutive points."""
    ba = p_prev - p_curr
    bc = p_next - p_curr
    cross_prod = np.cross(ba, bc)
    cross_prod_norm = np.linalg.norm(cross_prod)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    sin_angle = cross_prod_norm / (norm_ba * norm_bc)
    if sin_angle == 0:
        return 0.0
    curvature = 2 * sin_angle / np.linalg.norm(p_prev - p_next)
    return curvature

def convert_sectors_to_racing_line(
    sectors: List[float],
    drive_inside_sectors: np.ndarray,
    drive_outside_sectors: np.ndarray,
    fixed_start_point: np.ndarray,
    fixed_end_point: np.ndarray
) -> List[List[float]]:
    """Convert sector parameters to a full racing line path."""
    racing_line = [fixed_start_point.tolist()]
    for i, param in enumerate(sectors):
        param = np.clip(param, 0.0, 1.0)
        point = drive_inside_sectors[i + 1] + param * (drive_outside_sectors[i + 1] - drive_inside_sectors[i + 1])
        racing_line.append(point.tolist())
    racing_line.append(fixed_end_point.tolist())
    return racing_line
