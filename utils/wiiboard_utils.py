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
    return np.abs(mxx-mnx), np.abs(mxy-mny)

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
    
def plot_movement(signals, properties, PERSON, lim_X, lim_Y):
    fig=py.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for i  in range(len(properties[0])):
        x = signals[properties[1][i]][0]
        y = signals[properties[1][i]][1]
        ax.plot(x, y, 'r')
        ax.set_ylabel('position COPy [cm]')
        ax.set_xlabel('position COPx [cm]')
        ax.set_xlim((-lim_X, lim_X))
        ax.set_ylim((-lim_Y, lim_Y))
    ax.set_title(PERSON)
    py.tight_layout()
    py.show()
