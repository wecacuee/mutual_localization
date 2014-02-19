import numpy as np
from scipy import optimize
from scipy import spatial
import transformations as tf

def eucl2homo(pts):
    return np.hstack((pts, np.ones((len(pts), 1))))

def applytransform(T, pts):
    ptsh = eucl2homo(pts)
    ptsh_transformed = np.dot(T, ptsh.T)
    pts_transformed = ptsh_transformed[:3]
    return pts_transformed.T

class AlignPointClouds(object):
    def __init__(self, densepc, sparsepc):
        """
        densepc : Nx3
        sparsepc : Mx3, M < N
        """
        self.densekdtree = spatial.KDTree(densepc)
        self.sparsepc = sparsepc

    def _unpack(self, x):
        trans = x[:3]
        eulxyz = x[3:6] * np.pi / 180
        scale = x[6]
        return trans, eulxyz, scale

    def _x2transform(self, x):
        trans, eulxyz, scale = self._unpack(x)
        origin = [0, 0, 0]
        T = tf.concatenate_matrices(tf.translation_matrix(trans),
                                    tf.euler_matrix(eulxyz[0], eulxyz[1],
                                                    eulxyz[2], 'sxyz'),
                                    tf.scale_matrix(scale, origin))
        return T

    def _transform_sparsepc(self, x):
        T = self._x2transform(x)
        newpc = applytransform(T, self.sparsepc)
        return newpc

    def _cost_function(self, x):
        dist, indices = self.densekdtree.query(self._transform_sparsepc(x))
        mdist = np.sum(dist) / len(dist)
        self._last_mdist = mdist
        return mdist

    def _callback(self, x):
        print("x:{0}".format(x))
        print("mean dist {0}".format(self._last_mdist))

    def _align(self, x0):
        output = optimize.fmin_powell(
            self._cost_function,
            x0,
            maxiter=40,
            full_output=True,
            callback=self._callback)
        self.x = output[0]
        self.minerr = output[1]

    def align(self, x0=np.array([0., 0., 0., 1., 0., 0., 0., 1.])):
        self._align(x0)
        return self._x2transform(self.x)

    def err(self):
        return self.minerr

if __name__ == '__main__':
    import sys
    import pointcloud as pc
    densefname = sys.argv[1]
    sparsefname = sys.argv[2]
    densepcd = pc.pcdtoPointCloud(open(densefname))
    densepcd = densepcd[:, :3]
    densepcd = densepcd[~np.isnan(densepcd).any(axis=1)]
    sparsepcd = pc.readpset(open(sparsefname))
    sparsepcd = sparsepcd[:, :3]
    sparsepcd = sparsepcd[~np.isnan(sparsepcd).any(axis=1)]
    apc = AlignPointClouds(densepcd, sparsepcd)
    if len(sys.argv) > 3:
        x0 = np.array(sys.argv[3:], np.float32)
        print "Cost at x0:{0}".format(apc._cost_function(x0))
        print apc.align(x0)
    else:
        print apc.align()
    print apc.err()

