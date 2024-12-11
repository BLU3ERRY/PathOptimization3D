# utilities.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy import interpolate

def compute_track_boundaries(
    reference_path: np.ndarray,
    track_width: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    --------------------
    reference_path: np.ndarray (N x 3): 레퍼런스 지점들의 좌표
    track_width: float
    --------------------
    출력 튜플
        - 우측 지점들: np.ndaray (N x 3)
        - 좌측 지점들: np.ndarray (N x 3)
        - 노멀 벡터들(방향 벡터와 수직인 벡터들 모음): np.ndarray (N x 3)
    """
    rightside_points = []
    leftside_points = []
    normals = []
    n_points = len(reference_path)

    for i in range(n_points):
        if i < n_points - 1:
            direction_vector = reference_path[i + 1] - reference_path[i]
        else:
            direction_vector = reference_path[i] - reference_path[i - 1]
        norm = np.linalg.norm(direction_vector)
        if norm != 0:
            direction_vector /= norm
        else:
            direction_vector = np.array([1.0, 0.0, 0.0])

        #노멀 벡터: 방향 벡터에서 z성분은 무시하고 CCW 90도 회전 후 크기를 1로 만들기: 차 와 트랙 진행 방향 기준으로 왼쪽을 향함.
        normal_vector = np.array([-direction_vector[1], direction_vector[0], 0.0])
        normal_norm = np.linalg.norm(normal_vector)
        if normal_norm != 0:
            normal_vector /= normal_norm
        #FailSafe 코드
        else:
            normal_vector = np.array([0.0, 0.0, 1.0])

        current_point = reference_path[i]
        #좌측 지점과 우측 지점 정의
        rightside_point = current_point - normal_vector * (track_width / 2)
        leftside_point = current_point + normal_vector * (track_width / 2)
        rightside_points.append(rightside_point)
        leftside_points.append(leftside_point)
        normals.append(normal_vector)

    return np.array(rightside_points), np.array(leftside_points), np.array(normals)

def plot_track_boundaries_with_normals(
    reference_path: np.ndarray,
    rightside_points: np.ndarray,
    leftside_points: np.ndarray,
    drive_rightside_points: np.ndarray,
    drive_leftside_points: np.ndarray,
    normals: np.ndarray,
    fixed_start_point: np.ndarray,
    fixed_end_point: np.ndarray,
) -> None:
    """
    Plot the track boundaries along with normal vectors.
    """
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    plt.title("Track Boundaries with Normal Vectors")
    axis.plot(
        reference_path[:, 0],
        reference_path[:, 1],
        reference_path[:, 2],
        linestyle='dashed',
        label='Center Line',
        color='#000000'
    )
    axis.plot(
        rightside_points[:, 0],
        rightside_points[:, 1],
        rightside_points[:, 2],
        linestyle='dashed',
        label='Track Rightside Boundary',
        color='#FF0000'
    )
    axis.plot(
        leftside_points[:, 0],
        leftside_points[:, 1],
        leftside_points[:, 2],
        linestyle='dashed',
        label='Track Leftside Boundary',
        color='#00FF00'
    )
    axis.plot(
        drive_rightside_points[:, 0],
        drive_rightside_points[:, 1],
        drive_rightside_points[:, 2],
        linestyle='dashed',
        label='Drive Rightside Boundary',
        color='#880000'
    )
    axis.plot(
        drive_leftside_points[:, 0],
        drive_leftside_points[:, 1],
        drive_leftside_points[:, 2],
        linestyle='dashed',
        label='Drive Outside Boundary',
        color='#008800'
    )
    # Use fixed_start_point and fixed_end_point with z-offset
    axis.scatter(
        fixed_start_point[0],
        fixed_start_point[1],
        fixed_start_point[2],
        color='k',
        marker='o',
        label='Start Point'
    )
    axis.scatter(
        fixed_end_point[0],
        fixed_end_point[1],
        fixed_end_point[2],
        color='k',
        marker='X',
        label='End Point'
    )
    axis.legend()
    plt.show()

def plot_3d_lines(lines: List[np.ndarray], title: str = "3D Plot") -> None:
    """Plot multiple 3D lines."""
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    for line in lines:
        x, y, z = line[:, 0], line[:, 1], line[:, 2]
        axis.plot(x, y, z)
    plt.title(title)
    plt.show()

def smooth_reference_path(reference_path: np.ndarray, smooth_factor: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    레퍼런스 전처리 과정
    -------------------------
    reference_path: np.ndarray (N x 4), 입력 경로의 좌표 및 위험 구간 표기
    smooth_factor: float (0 이상.)
    -------------------------
    smoothed_path: np.ndarray (M x 3), 전처리된 결과 경로 좌표
    is_danger: np.ndarray (M), 트랙의 위험 유무
    """
    if smooth_factor < 0:
        raise ValueError("Smooth factor must be non-negative.")
    #마스크: 이전 지점과 다른 지점만 고르는 마스크 같음.
    mask = np.any(np.diff(reference_path, axis=0) != 0, axis=1)
    reference_path = reference_path[:-1][mask]
    reference_path = reference_path[np.isfinite(reference_path).all(axis=1)]
    if len(reference_path) < 4:
        raise ValueError("Not enough unique points to perform smoothing.")
    x, y, z, is_danger = reference_path[:, 0], reference_path[:, 1], reference_path[:, 2], reference_path[:, 3]
    tck, _ = interpolate.splprep([x, y, z], s=smooth_factor, per=True)
    smoothed_x, smoothed_y, smoothed_z = interpolate.splev(np.linspace(0, 1, len(reference_path)), tck)
    smoothed_path = np.vstack([smoothed_x, smoothed_y, smoothed_z]).T
    return (smoothed_path, is_danger)

def plot_sectors_with_boundaries(
    reference_path: np.ndarray,
    drive_rightside_sectors: np.ndarray,
    drive_leftside_sectors: np.ndarray,
    mid_sectors: np.ndarray
) -> None:
    """Plot the sectors with boundaries on the track."""
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    plt.title("Track Sectors with Boundaries")
    for i in range(len(drive_rightside_sectors)):
        axis.plot(
            [drive_rightside_sectors[i][0], drive_leftside_sectors[i][0]],
            [drive_rightside_sectors[i][1], drive_leftside_sectors[i][1]],
            [drive_rightside_sectors[i][2], drive_leftside_sectors[i][2]],
            'g--'
        )
    axis.plot(
        reference_path[:, 0],
        reference_path[:, 1],
        reference_path[:, 2],
        linestyle='dashed',
        color="#000000",
        label='Center Line'
    )
    axis.plot(
        drive_rightside_sectors[:, 0],
        drive_rightside_sectors[:, 1],
        drive_rightside_sectors[:, 2],
        linestyle='dashed',
        color='#880000',
        label='Drive Rightside Boundary'
    )
    axis.plot(
        drive_leftside_sectors[:, 0],
        drive_leftside_sectors[:, 1],
        drive_leftside_sectors[:, 2],
        linestyle='dashed',
        color='#008800',
        label='Drive Leftside Boundary'
    )
    axis.legend()
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
