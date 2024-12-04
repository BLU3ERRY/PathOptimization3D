import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy import interpolate

def compute_track_boundaries(
    reference_path: np.ndarray,
    track_width: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the inner and outer boundaries of the track.
    Returns the normal vectors for each reference point.
    """
    inside_points = []
    outside_points = []
    normals = []
    n_points = len(reference_path)

    for i in range(n_points):
        direction_vector = (
            reference_path[i + 1] - reference_path[i] if i < n_points - 1 else reference_path[i] - reference_path[i - 1]
        )
        norm = np.linalg.norm(direction_vector)
        direction_vector = direction_vector / norm if norm != 0 else np.array([1.0, 0.0, 0.0])
        normal_vector = np.array([-direction_vector[1], direction_vector[0], 0.0])
        normal_norm = np.linalg.norm(normal_vector)
        normal_vector = normal_vector / normal_norm if normal_norm != 0 else np.array([0.0, 0.0, 1.0])
        current_point = reference_path[i]
        inside_point = current_point - normal_vector * (track_width / 2)
        outside_point = current_point + normal_vector * (track_width / 2)
        inside_points.append(inside_point)
        outside_points.append(outside_point)
        normals.append(normal_vector)

    return np.array(inside_points), np.array(outside_points), np.array(normals)

def plot_track_boundaries_with_normals(
    reference_path: np.ndarray,
    inside_points: np.ndarray,
    outside_points: np.ndarray,
    drive_inside_points: np.ndarray,
    drive_outside_points: np.ndarray,
    normals: np.ndarray
) -> None:
    """
    Plot the track boundaries along with normal vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Track Boundaries with Normal Vectors")
    ax.plot(reference_path[:, 0], reference_path[:, 1], reference_path[:, 2], 'r-', label='Center Line')
    ax.plot(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], 'b-', label='Track Inside Boundary')
    ax.plot(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2], 'b-', label='Track Outside Boundary')
    ax.plot(drive_inside_points[:, 0], drive_inside_points[:, 1], drive_inside_points[:, 2], 'g--', label='Drive Inside Boundary')
    ax.plot(drive_outside_points[:, 0], drive_outside_points[:, 1], drive_outside_points[:, 2], 'g--', label='Drive Outside Boundary')
    start_point = reference_path[0]
    end_point = reference_path[-1]
    ax.scatter(start_point[0], start_point[1], start_point[2], color='k', marker='o', label='Start Point')
    ax.scatter(end_point[0], end_point[1], end_point[2], color='k', marker='X', label='End Point')
    ax.legend()
    plt.show()

def plot_3d_lines(lines: List[np.ndarray], title: str = "3D Plot") -> None:
    """Plot multiple 3D lines."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for line in lines:
        x, y, z = line[:, 0], line[:, 1], line[:, 2]
        ax.plot(x, y, z)
    plt.title(title)
    plt.show()

def smooth_reference_path(reference_path: np.ndarray, smooth_factor: float = 0.0) -> np.ndarray:
    """Smooth the reference path using spline interpolation."""
    if smooth_factor < 0:
        raise ValueError("Smooth factor must be non-negative.")
    mask = np.any(np.diff(reference_path, axis=0) != 0, axis=1)
    reference_path = reference_path[:-1][mask]
    reference_path = reference_path[np.isfinite(reference_path).all(axis=1)]
    if len(reference_path) < 4:
        raise ValueError("Not enough unique points to perform smoothing.")
    x, y, z = reference_path[:, 0], reference_path[:, 1], reference_path[:, 2]
    tck, _ = interpolate.splprep([x, y, z], s=smooth_factor, per=True)
    smoothed_x, smoothed_y, smoothed_z = interpolate.splev(np.linspace(0, 1, len(reference_path)), tck)
    smoothed_path = np.vstack([smoothed_x, smoothed_y, smoothed_z]).T
    return smoothed_path

def plot_sectors_with_boundaries(
    reference_path: np.ndarray,
    drive_inside_sectors: np.ndarray,
    drive_outside_sectors: np.ndarray,
    mid_sectors: np.ndarray
) -> None:
    """Plot the sectors with boundaries on the track."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Track Sectors with Boundaries")
    for i in range(len(drive_inside_sectors)):
        ax.plot(
            [drive_inside_sectors[i][0], drive_outside_sectors[i][0]],
            [drive_inside_sectors[i][1], drive_outside_sectors[i][1]],
            [drive_inside_sectors[i][2], drive_outside_sectors[i][2]],
            'g--'
        )
    ax.plot(reference_path[:, 0], reference_path[:, 1], reference_path[:, 2], 'r-', label='Center Line')
    ax.plot(drive_inside_sectors[:, 0], drive_inside_sectors[:, 1], drive_inside_sectors[:, 2], 'g--', label='Drive Inside Boundary')
    ax.plot(drive_outside_sectors[:, 0], drive_outside_sectors[:, 1], drive_outside_sectors[:, 2], 'g--', label='Drive Outside Boundary')
    ax.legend()
    plt.show()

def plot_lap_time_history(evaluation_history: List[float]) -> None:
    """Plot the history of lap times over iterations."""
    plt.figure()
    plt.plot(evaluation_history, label='Total Lap Time')
    plt.xlabel('Iteration')
    plt.ylabel('Lap Time (s)')
    plt.title('Lap Time per Iteration')
    plt.legend()
    plt.show()
