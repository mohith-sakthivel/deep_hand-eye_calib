import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def vizualize_poses(poses, size=0.2):
    translation = poses[..., :3]
    rotation = np.array([R.from_quat(q).as_matrix() for q in poses[..., [4, 5, 6, 3]]])
    poses_homo = np.zeros((len(poses), 4, 4), dtype=float)
    poses_homo[:, :3, :3] = rotation
    poses_homo[:, :3, 3] = translation
    poses_homo[:, 3, 3] = 1

    frames = []
    for i in range(len(poses_homo)):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(poses_homo[i])
        frames.append(frame)

    o3d.visualization.draw_geometries(frames)
