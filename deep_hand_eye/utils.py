import numpy as np

class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


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
