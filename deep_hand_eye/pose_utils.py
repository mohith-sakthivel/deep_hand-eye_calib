import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    # d = abs(np.dot(q1, q2))
    d = abs(q2 @ q1.T)
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta

def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    s = q.shape

    if all(q[..., 1:] == 0):
        q = np.zeros([3])
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q

def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))
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
