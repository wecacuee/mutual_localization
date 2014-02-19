import os
import pickle
import re

import numpy as np
import cv2

class Pickle(object):
    # Sample files
    #'data/bnwmarker20120816/selected/markedimgs.pickle'),
    def load(self, fname):
        return pickle.load(open(fname))

    def dump(self, lines, fname):
        pickle.dump(lines, open(fname, 'w'))

class NumpyTxtMarkedImages(object):
    def __init__(self):
        pass
    # Sample files
    # data/bnwmarker20120827/markedimgs.txt
    def load(self, fname):
        table = np.loadtxt(fname, dtype='S50')
        img0markers = np.array(table[:, 2:6], np.float32).reshape(-1, 2, 2)
        img1markers = np.array(table[:, 6:10], np.float32).reshape(-1, 2, 2)
        img_pos_target = np.array(table[:, 10:12], np.float32)
        lines = [row for row in zip(table[:, 0], 
                                   table[:, 1],
                                   img0markers, img1markers, img_pos_target)]
        return lines

    def dump(self, lines, fname):
        composite_array = np.hstack(
            [np.array([r[i] for r in lines]).reshape(len(lines), -1)
             for i in range(5)])
        np.savetxt(fname, composite_array, fmt='%s')

def np_like_loadtxt(fname, dtype, delimiter=' '):
    with open(fname) as f:
        for row in f.readlines():
            row = row.strip()
            yield row.split(delimiter)

def isfloat(obj):
    if np.isscalar(obj): 
        return type(obj) == float
    else:
        return obj.dtype == np.float

def stringfy(obj, fmt):
    return fmt % obj if isfloat(obj) else str(obj)

def np_like_savetxt(fname, lines, fmt, delimiter=' '):
    with open(fname, 'w') as f:
        for row in lines:
            f.write(delimiter.join([stringfy(c, fmt) for c in row]))
            f.write("\n")

class NumpyTxt(object):
    def __init__(self, include_dtype=True):
        self.include_dtype = include_dtype

    # Sample files
    # data/bnwmarker20120830/markedimgs.txt
    # data/bnwmarker20120831/markedimgs.txt
    def load(self, fname):
        lines = list()
        table = np_like_loadtxt(fname, dtype='S50')
        for row in table:
            arr_row = list()
            i = 0
            while i < len(row):
                if self.include_dtype:
                    dtype = row[i]
                    i += 1
                shape_len = int(row[i])
                i += 1
                if shape_len:
                    shape = np.array(row[i:i+shape_len], np.int)
                    i += shape_len
                    totlen = reduce(lambda x, y: x*y, shape, 1)
                else:
                    shape = tuple()
                    totlen = 1
                data = row[i:i+totlen]
                i += totlen
                if self.include_dtype:
                    arr = np.array(data, dtype).reshape(shape)
                else:
                    arr = np.array(data).reshape(shape)
                arr_row.append(arr)
            lines.append(arr_row)
        return lines

    def dump(self, lines, fname):
        printoptions = np.get_printoptions()
        np.set_printoptions(precision=12)
        np_lines = list()
        for row in lines:
            np_row = list()
            for cell in row:
                cell = np.asarray(cell)
                cell_shape = cell.shape
                cell_shape_len = len(cell.shape)
                cell_flatten = cell.flatten()
                data_flatten = []
                if self.include_dtype:
                    data_flatten.append(str(cell.dtype))
                data_flatten.append(cell_shape_len)
                data_flatten.extend(cell_shape)
                data_flatten.extend(cell_flatten)
                np_row.extend(data_flatten)
            np_lines.append(np_row)
        np_like_savetxt(fname, np_lines, fmt='%.12f')
        np.set_printoptions(**printoptions)

class NumpyTxtHeader(object):
    """Writes meta data on the header as opposed to each line
    """
    delimiter = ' '
    comments = '#'
    def parse_header(self, header):
        columns = dict()
        row = header.split(self.delimiter)
        col_count = 0
        i = 0
        while i < len(row):
            dtype = row[i]
            i += 1
            shape_len = int(row[i])
            i += 1
            if shape_len:
                shape = np.array(row[i:i+shape_len], np.int)
                i += shape_len
                totlen = reduce(lambda x, y: x*y, shape, 1)
            else:
                shape = tuple()
                totlen = 1
            columns[col_count] = dict(shape=shape, dtype=dtype, totlen=totlen)
            col_count += 1
        return columns

    def compose_header(self, sample_row):
        header_list = list()
        for cell in sample_row:
            cell = np.asarray(cell)
            cell_shape = cell.shape
            cell_shape_len = len(cell.shape)
            header_list.append(cell.dtype)
            header_list.append(cell_shape_len)
            header_list.extend(cell_shape)
        return self.delimiter.join(map(str, header_list))

    def load(self, fname):
        lines = list()
        fileh = open(fname)

        lines = list(fileh)
        commented = [line.lstrip(self.comments)
                     for line in lines if line.startswith('#')]
        header = commented[0].rstrip()
        columns = self.parse_header(header)

        table = np.loadtxt(lines, dtype='S500',
                           delimiter=self.delimiter,
                           comments=self.comments)
        outarray = list()
        for row in table:
            arr_row = list()
            i = 0
            for col in columns.values():
                arr = np.array(row[i:i+col['totlen']], dtype=col['dtype'])
                i += col['totlen']
                arr = arr.reshape(col['shape'])
                arr_row.append(arr)
            outarray.append(arr_row)
        return outarray

    def dump(self, lines, fname):
        printoptions = np.get_printoptions()
        np.set_printoptions(precision=12)
        np_lines = list()
        lines = list(lines)
        for row in lines:
            np_row = list()
            for cell in row:
                cell = np.asarray(cell)
                cell_flatten = cell.flatten()
                np_row.extend(cell_flatten)
            np_lines.append(np.array(np_row))
        np_lines = np.vstack(np_lines)

        fileh = open(fname, 'w')
        header_line = self.compose_header(lines[0])
        fileh.write("%s%s\n" % (self.comments, header_line))

        np.savetxt(fileh, np_lines, fmt='%s', delimiter=self.delimiter)
        np.set_printoptions(**printoptions)

def mkdir(fname):
    dir = os.path.dirname(fname)
    if not os.path.exists(dir):
        os.makedirs(dir)

class PMVSWriter(object):
    def __init__(self, K, pmvs_dir="pmvs"):
        self.root = pmvs_dir
        self.K = K
        self.options = dict(timages=[], oimages=[], level=1, csize=2,
                            threshold=0.7, wsize=7, minImageNum=3, CPU=4,
                            useVisData=0, sequence=1, quad=2.5, maxAngle=10)

    def option_str(self, opt, optval):
        if hasattr(optval, '__len__'):
            return "%s %d %s" % (opt, len(optval), " ".join(map(str, optval)))
        else:
            return "{0} {1}".format(opt, optval)

    def write_options(self):
        fname = os.path.join(self.root, "pmvs_options.txt")
        mkdir(fname)
        with open(fname, 'w') as f:
            f.write("\n".join([self.option_str(k, v)
                               for k, v in self.options.items()]))
        mkdir(os.path.join(self.root, "models", "dummy.txt"))
    
    def save_img(self, i, img):
        fname = os.path.join(self.root, "visualize", "%08.d.ppm" % i)
        mkdir(fname)
        cv2.imwrite(fname, img)

    def add_image(self, i, T, img):
        self.options['timages'].append(i)
        self.write_transform(i, T)
        self.save_img(i, img)

    def write_transform(self, i, T):
        P = np.dot(self.K, T[:3])
        fname = os.path.join(self.root, "txt", "%08d.txt" % i)
        mkdir(fname)
        with open(fname, 'w') as f:
            f.write("CONTOUR\n")
            f.write("\n".join([" ".join(["%.12f" % c for c in row])
                               for row in P]))
class BundlerReader(object):
    def load(self, fname):
        cameraattrs = list()
        with open(fname) as file:
            camerano = -1
            pointno = -1
            for line in file:
                if line.startswith("#"):
                    continue
                ncam, npts = map(int, line.strip().split())
                camerano += 1
                for i in range(ncam):
                    camline = file.next().strip()
                    f, k1, k2 = [float(f) for f in camline.split()]
                    R = np.array([file.next().strip().split()
                                  for i in range(3)], np.float32)
                    t = np.array(file.next().strip().split(), np.float32)
                    cameraattrs.append(dict(f=f, k1=k1, k2=k2, R=R, t=t))
                break
        return cameraattrs
