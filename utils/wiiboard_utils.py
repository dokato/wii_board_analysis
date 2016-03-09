import numpy as np

def path_length(x, y):
    """
    Calculate length of path from x and y coordinates
    """
    rx = np.asarray([x[0]] + [x[i]-x[i-1] for i in range(1,len(x))])
    ry = np.asarray([y[0]] + [y[i]-y[i-1] for i in range(1,len(y))])
    return np.sum(np.sqrt(rx**2+ry**2))

def maximal_sway(x, y):
    """
    Returns maximal sway in each axis
    """
    mnx, mny = min(x), min(y)
    mxx, mxy = max(x), max(y)
    return mxx-mnx, mxy-mny

def mean_total_velocity(x, y, fs, timewin=1):
    """
    Calculate mean velocity in windows *timewin* given in seconds.
    """
    rx = np.asarray([abs(x[0])] + [abs(x[i]-x[i-1]) for i in range(1,len(x))])
    ry = np.asarray([abs(y[0])] + [abs(y[i]-y[i-1]) for i in range(1,len(y))])
    return np.sqrt(rx**2+ry**2)/(len(x)/fs)

def mean_velocity(x, fs, timewin=1):
    """
    Calculate mean velocity in windows *timewin* given in seconds.
    """
    rx = np.asarray([abs(x[0])] + [abs(x[i]-x[i-1]) for i in range(1,len(x))])
    return rx/(len(x)/fs)

def romberg(xo, yo, xz, yz):
    """
    Romberg coefficient - ratio of paths of open eyes to closed.
    """
    path_o = path_length(xo, yo)
    path_z = path_length(xz, yz)
    return path_o/path_z

def rms(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.mean(x**2)), np.sqrt(np.mean(y**2))
