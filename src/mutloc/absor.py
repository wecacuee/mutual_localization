# vim: set fileencoding=utf-8
"""
Given coordinates of a point set in two coordinate frames find the relative
pose between two frames. See [2]_

Examples
--------
>>> pointsL = [np.array([ 0.,  0.,  1.]), np.array([ 0.,  1.,  0.]), np.array([ 1.,  0.,  0.])]
>>> pointsR = [np.array([ 0.,  0.,  2.]), np.array([ 0.,  1.,  1.]), np.array([ 1.,  0.,  1.])]
>>> np.allclose(absor(pointsL, pointsR),
... np.array([[ 1.,  0., -0.,  0.],
...           [ 0.,  1.,  0., -0.],
...           [-0.,  0.,  1.,  1.],
...           [ 0.,  0.,  0.,  1.]]))
True
"""
import numpy as np
import transformations as tf
import mutloc.utils as utils
import matplotlib.mlab as mlab
DEBUG = True

TOL = 1e-6
def _absor_from_transformation(pointsL, pointsR, tol=TOL):
    return tf.affine_matrix_from_points(np.array(pointsL).T,
                                        np.array(pointsR).T,
                                        shear=False, scale=False, usesvd=True)

def ransac(data, fit_model, n, p, w, find_inliers, find_error):
    """
    input:
        data - a set of observations
        model - a model that can be fitted to data 
        n - the minimum number of data required to fit the model
        k - the number of iterations performed by the algorithm
        find_inliers - a function that returns a boolean array of inliers
    output:
        best_model - model parameters which best fit the data (or nil if no good model is found)
        best_consensus_set - data points from which this model has been estimated
        best_error - the error of this model relative to the data 
    """
    # k : number of iterations 
    k = int(np.log(1 - p)/np.log(1 - w**n))
    best_model = None
    best_inliers = None
    best_error = np.inf
    for i in xrange(k):
        maybe_indices = np.random.randint(0, len(data), n)
        maybe_inliers = data[maybe_indices]
        maybe_model = fit_model(maybe_inliers)
        inliers = find_inliers(maybe_model, data)
        if np.sum(inliers) / float(len(inliers)) < w:
            if DEBUG:
                print("Found only %d/%d inliers" % (np.sum(inliers),
                                                    len(inliers)))
            # not a good fit
            continue

        # this implies that we may have found a good model, now test how
        # good it is
        this_model = fit_model(data[inliers])
        this_error = find_error(this_model, consensus_set)
        if this_error < best_error:
            # we have found a model which is better than any of the
            # previous ones, keep it until a better one is found
            best_error = this_error
            best_inliers = inliers
            best_error = this_model
    return best_model, best_inliers, best_error

def absor_ransac(pointsL, pointsR, tol=TOL):
    # http://en.wikipedia.org/wiki/RANSAC#The_algorithm
    # data : pointsL, pointsR
    data = np.hstack((pointsL, pointsR))
    # model : [pointsR;1] = np.dot(T, [pointsL;1])
    def fit_model(data_sample):
        pointsL = data_sample[:, :3]
        pointsR = data_sample[:, 3:6]
        T = tf.affine_matrix_from_points(pointsL, pointsR,
                                     shear=False, scale=False)
        return T
    # n : minimum number of data points required to fit the model = 3
    n = 3

    # Let p be the probability that the RANSAC algorithm in some iteration selects only inliers 
    p = 0.99
    # w = number of inliers in data / number of points in data
    # A common case is that w is not well known beforehand, but some rough value can be given
    w = 0.6
    # t - a threshold value for determining when a datum fits a model
    # don't know
    def find_inliers(maybe_model, data):
        pointsL = data[:, :3]
        pointsR = data[:, 3:6]
        T = maybe_model
        pointsR_maybe = utils.apply_transform(T, pointsL)
        # our threshold increases with square of depth
        return (mlab.vector_lengths(pointsR_maybe - pointsR) 
                < 1.5*pointsR[:, 2]**2)

    def find_error(this_model, consensus_set):
        pointsL = consensus_set[:, :3]
        pointsR = consensus_set[:, 3:6]
        T = this_model
        pointsR_maybe = utils.apply_transform(T, pointsL)
        # our threshold increases with square of depth
        return np.sum(mlab.vector_lengths(pointsR_maybe - pointsR))
    best_model, best_inliers, best_error = ransac(
        data, fit_model, n, p, w, find_inliers, find_error)
    return best_model, best_inliers, best_error

absor = _absor_from_transformation
