import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.sum(q2 * q1, axis=-1))
    d = np.minimum(1.0, np.maximum(-1.0, d))
    theta = np.rad2deg(2 * np.arccos(d))
    return theta


def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """

    if isinstance(q, np.ndarray):
        q = np.arccos(q[..., 0:1]) * q[..., 1:] / np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    elif isinstance(q, torch.Tensor):
        q = torch.arccos(q[..., 0:1]) * q[..., 1:] / torch.linalg.norm(q[..., 1:], dim=-1, keepdims=True)
    return q


def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    if isinstance(q, np.ndarray):
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))
    elif isinstance(q, torch.Tensor):
        n = torch.linalg.norm(q, dim=-1, keepdims=True)
        q = torch.hstack((torch.cos(n), torch.sinc(n / torch.pi) * q))

    return q
    

def invert_homo(T):
    """
    Inverts a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    T = np.array(T)
    R, p =  T[0: 3, 0: 3], T[0: 3, 3]
    return np.r_[np.c_[R.T, -np.dot(R.T, p)], [[0, 0, 0, 1]]]


def homo_to_quat(T):
    quat_pose = np.empty(7)
    quat_pose[:3] = T[:3, 3]
    quat_pose[3:] = R.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
    quat_pose[3:] *= np.sign(quat_pose[3])  # constrain to hemisphere
    return quat_pose


def homo_to_log_quat(T):
    quat_pose = np.empty(6)
    quat_pose[:3] = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
    quat *= np.sign(quat[0])  # constrain to hemisphere
    quat_pose[3:] = qlog(quat)
    return quat_pose


def quaternion_to_axis_angle(q, in_degrees: bool = False):
    """
    Convert quaternion into axis angle representation
    """
    axis = q[..., 1:] / np.linalg.norm(q[..., 1:])

    angle = 2 * np.arccos(q[..., 0])
    if in_degrees:
        angle = np.rad2deg(angle)

    return axis, angle
