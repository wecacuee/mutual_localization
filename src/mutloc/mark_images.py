import sys
import os
import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt

from mutloc import filewriters
from mutloc import config

def absolute_path(fname, dir=None, relfile=None):
    if dir is None:
        try:
            curfile = relfile or __file__
            dir = os.path.dirname(curfile)
        except NameError:
            dir = os.getcwd()
    return os.path.join(dir, fname)

def init_plot(msg):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(msg)
    return fig, ax

def get_user_input_points(cvimg, n=1, msg=' '):
    fig, ax = init_plot(msg)
    ax.imshow(cvimg[..., ::-1], interpolation='none')
    point0 = fig.ginput(n=n, timeout=-1)
    plt.close(fig)
    return point0

def readfile_approxsync():
    approxsync_file = "data/bnwmarker20120816/selected/approxsynced.txt"
    tab = np.loadtxt(absolute_path(approxsync_file),
                     dtype={'names' : ('time0', 'cam0', 'time1', 'cam1'),
                            'formats' : ('f4', 'S40', 'f4', 'S40')})
    for _, f0, _, f1 in tab:
        absf0 = absolute_path(f0, relfile=absolute_path(approxsync_file))
        absf1 = absolute_path(f1, relfile=absolute_path(approxsync_file))
        yield absf0, absf1, f0, f1

def readfile(indices=xrange(10000)):
    data_dir = data_directory()
    files = ('img%04d_0.png', 'img%04d_1.png')
    for i in indices:
        absfs = [os.path.join(data_dir, temp % i) for temp in files]
        if not all(os.path.exists(f) for f in absfs):
            break
        absfs = [i] + absfs + [temp % i for temp in files]
        yield absfs

def dumpfile_numpy(lines, fname):
    filewriters.NumpyTxt(include_dtype=True).dump(lines, fname)

def dumpfile_pickle(lines):
    fname = absolute_path('data/bnwmarker20120816/selected/markedimgs.pickle')
    filewriters.Pickle().dump(lines, fname)

def read_markedimgs(fname):
    if not os.path.exists(fname):
        return []
    lines = filewriters.NumpyTxt(include_dtype=True).load(fname)
    rows = []
    for f0, f1, pt0, pt1, cp in lines:
        rows.append((f0, f1,
                     np.asarray(pt0, np.float32),
                     np.asarray(pt1, np.float32),
                     np.asarray(cp, np.float32)))
    return rows

def markedimgs_file():
    return os.path.join(data_directory(), 'markedimgs.txt')

def markedimgs_directory(conf, confdir):
    return os.path.join(confdir, os.path.dirname(conf.marked_images()))

def get_user_inputs(img0, img1):
    point0 = get_user_input_points(img0, n=2, msg='Pick up the two marker points')
    point1 = get_user_input_points(
        img1, n=5,
        msg='Pick up the two marker points and 3 correspondence point')
    return point0, point1

def main(configfile, indices):
    conf = config.Config(open(configfile))
    global data_directory
    confdir = os.path.dirname(configfile)
    data_directory = lambda : markedimgs_directory(conf, confdir)

    lines = read_markedimgs(markedimgs_file())
    if indices is not None:
        images = readfile(indices)
    else:
        images = readfile()
    for i, absf0, absf1, f0, f1 in images:
        img0, img1 = cv2.imread(absf0), cv2.imread(absf1)
        if img0 is None:
            raise RuntimeError("Unable to load {0}".format(absf0))
        if i < len(lines):
            # hack to allow user to modify file markedimgs to remark a
            # particular image
            _, _, m0, m1, tp = lines[i]
            if (m0 < 0).any() or (m1 < 0).any() or (tp < 0).any():
                point0, point1 = get_user_inputs(img0, img1)
                lines[i] = (f0, f1, point0, point1[:2], point1[2:])
        else:
            point0, point1 = get_user_inputs(img0, img1)
            lines.append((f0, f1, point0, point1[:2], point1[2:]))
    dumpfile_numpy(lines, markedimgs_file())

if __name__ == '__main__':
    import sys
    configfile = sys.argv[1]
    if len(sys.argv) > 2:
        indices = [int(i) for i in sys.argv[2:]]
    else:
        indices = None
    main(configfile, indices)
