import os

import numpy as np
import transformations as tf

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from mpl_toolkits.mplot3d import Axes3D

TOL = 1e-6
def transform_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    tinv = -1*np.dot(R.T, t)
    Tinv = T.copy()
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = tinv
    return Tinv

def projection(point3d, tol=TOL):
    return np.array(point3d[:2]) / (point3d[2] if point3d[2] != 0.0 else TOL)

def eucl2homo(pts_vertical):
    pts = pts_vertical
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def matplotlibusetex():
    from matplotlib import rc
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True) 

def homo2eucl(pts_horizontal):
    pts = pts_horizontal
    return pts[:-1, :] / pts[-1, :]

def apply_transform(T, point):
    point = np.asarray(point)
    if len(point.shape) == 2:
        return homo2eucl(np.dot(T, eucl2homo(point).T)).T
    else:
        return np.dot(T, np.hstack((point, 1)))[:3]

def transform_from_quat(quat, trans):
    T = tf.quaternion_matrix(quat)
    T[:3, 3] = trans
    return T

def camera_projection(point3d, K):
    return projection(np.dot(K, point3d))

def undo_camera_projection(point2d, K):
    return np.dot(np.linalg.inv(K), np.hstack((point2d, 1)))

def absolute_path(fname, dir=None, relfile=None):
    if dir is not None:
        pass # do nothing
    elif relfile is not None:
        dir = os.path.dirname(relfile)
    else:
        dir = os.getcwd()
    return os.path.join(dir, fname)

def rotation_matrix_to_angle(R):
    return np.arccos((np.trace(R) - 1) / 2.)

def angle_between_rotations(R0, R1):
    return rotation_matrix_to_angle(np.dot(R0.T, R1))

def axes3d():
    fig = plt.figure()
    ax = Axes3D(fig)
    return ax

def matplotscatter(x, y, z, xlabel='x', ylabel='y', zlabel='z',
                   axes=None,
                   title='scatter', c=None, cmap=colormap.jet, **kwargs):
    if axes is None:
        ax = axes3d()
    else:
        ax = axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    if c is None:
        c = z
    p = ax.scatter(x, y, z, c=c, cmap=cmap, **kwargs)
    ax.figure.colorbar(p)
    return ax.figure, ax, p

def projectionto3d(K, pts2d):
    pts2d = np.asarray(pts2d)
    if len(pts2d.shape) == 1:
        pts2d = pts2d.reshape(1, -1)
    Kinv = np.linalg.inv(K)
    pts3d = np.dot(Kinv, eucl2homo(pts2d).T).T
    return pts3d

def floatunique(x, tol=TOL):
    sx = np.sort(x.flat)
    dx = np.diff(sx)
    ind = dx > tol
    return sx[np.hstack(([True], ind))]

def _dataforerrorbar(x, z, tol=TOL):
    uniqx = floatunique(x)
    z_mean = list()
    z_err = list()
    for xi in uniqx:
        zi = z[np.abs(x - xi) < tol]
        z_mean.append(np.mean(zi))
        z_err.append(np.std(zi))
    return uniqx, z_mean, z_err

def newaxes(xlabel, ylabel, title, subplot=111):
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(subplot)
    fig.subplots_adjust(bottom=0.15, right=0.95, top=0.95, left=.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(5))
    return ax

def errorbar_from_3d(x, y, z, xlabels=['X', 'X'], axes=None,
                     ylabel='Y', titles=['errorbar1', 'errorbar2'],
                     color='b', label='error', **kwargs):
    if axes is None:
        ax1 = newaxes(xlabels[0], ylabel, titles[0])
        ax2 = newaxes(xlabels[1], ylabel, titles[1])
    else:
        ax1, ax2 = axes
    uniqx, z_mean, z_err = _dataforerrorbar(x, z)
    ax1.errorbar(uniqx, z_mean, yerr=z_err, color=color, label=label)
    uniqy, z_mean, z_err = _dataforerrorbar(y, z)
    ax2.errorbar(uniqy, z_mean, yerr=z_err, color=color, label=label)
    return ax1, ax2
