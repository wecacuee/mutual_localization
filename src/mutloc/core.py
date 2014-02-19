"""
Given points to line correspondence in two coordinate frames solve for the
relative pose.
"""
import numpy as np
import itertools as it
import scipy.optimize as optimize

import absor as reg
import scalefactors as su
import transformations as tf
from utils import transform_inv, projection, apply_transform, transform_from_quat

import log
import logging
logger = log.getLogger(__name__)
logger.setLevel(logging.WARN)

TOL = 1e-6

def _compute_err(T, frame0scaled, frame1scaled):
    Tinv = transform_inv(T)
    errs0 = [np.linalg.norm(
        projection(apply_transform(Tinv, f1)) - projection(f0))
        for f0, f1 in frame0scaled]
    errs1 = [np.linalg.norm(
        projection(apply_transform(T, f0)) - projection(f1))
        for f0, f1 in frame1scaled]
    tot_err = sum(errs0 + errs1)
    return tot_err

def correspondence(frame1scaled, frame2scaled, method="analytic", tol=TOL):
    if method == "analytic":
        corr_points = correspondance_points(frame1scaled, frame2scaled,
                                            tol=tol)
        Troots = [ (reg.absor([c[0] for c in corrp],
                                      [c[1] for c in corrp], tol=tol))
                  for corrp in corr_points]
        Troots_sorted = sorted(Troots, key=lambda T: _compute_err(T,
                                                                  frame1scaled,
                                                                  frame2scaled))
        return Troots_sorted
    elif method == "numeric":
        quat0 = [0., 1., 0. , 0.]
        trans0 = [0., 0., 1.]
        ret, cov_dict = optimize.leastsq(_objective_func(frame1scaled,
                                                     frame2scaled),
                                     quat0 + trans0)
        quat = ret[:4]
        trans = ret[4:7]
        return [(transform_from_quat(quat, trans), 0)]
    else:
        raise ValueError("Unknown method {0}".format(method))

def finetune(Tbest, frame1scaled, frame2scaled):
    def _unpack(x):
        trans = x[:3]
        eulxyz = x[3:]
        T = tf.concatenate_matrices(tf.translation_matrix(trans),
                                    tf.euler_matrix(eulxyz[0], eulxyz[1],
                                                    eulxyz[2], 'sxyz'))
        return T

    def _pack(T):
        eulxyz = tf.euler_from_matrix(T, axes='sxyz')
        trans = tf.translation_from_matrix(T)
        return np.hstack((trans, eulxyz))
    def optfunc(x):
        T = _unpack(x)
        return _compute_err(T, frame1scaled, frame2scaled)
    print("error start {0}".format(_compute_err(Tbest, frame1scaled, frame2scaled)))
    output = optimize.fmin_bfgs(optfunc,
                                x0=_pack(Tbest),
                                maxiter=100, 
                                full_output=True,
                                disp=1)
    Tnewbest = _unpack(output[0])
    err = output[1]
    print("error end {0}".format(output[1]))
    return Tnewbest

def _objective_func(frame1scaled, frame2scaled):
    def error(x):
        quat = x[:4]
        trans = np.array(x[4:7])
        T = transform_from_quat(quat, trans)
        Tinv = transform_inv(T)
        err0 = np.hstack([projection(f0) - projection(apply_transform(Tinv, f1))
                               for f0, f1 in frame1scaled])
        err1 = np.hstack([projection(apply_transform(T, f0)) - projection(f1)
                               for f0, f1 in frame2scaled])
        err3 = (np.linalg.norm(quat) - 1)
        return np.hstack((err0, err1, err3))
    return error

def correspondance_points_numeric(frame1scaled_n, frame2scaled_n, tol=TOL):
    frame1scaled, frame2scaled = normalize_points(
        frame1scaled_n, frame2scaled_n)
    possible_scales = su.numeric_find_scale_factors(
        frame1scaled, frame2scaled, tol=tol)
    return _corr_points_from_scales(possible_scales, frame1scaled,
                                    frame2scaled)

def scale_numeric(frame1scaled_n, frame2scaled_n):
    frame1scaled, frame2scaled = normalize_points(
        frame1scaled_n, frame2scaled_n)
    n = len(frame1scaled) + len(frame2scaled)
    neqns = n*(n-1)/2
    P = generate_polynomials(frame1scaled, frame2scaled)
    def objective_func(x):
        x = x[:n]
        X = np.array([i**2 for i in x]
               + list(map(lambda c: c[0]*c[1], it.combinations(x, 2)))
               + list(x) + [1])
        return np.dot(P, X)
    root = optimize.fsolve(objective_func, np.zeros(neqns))
    nroots = ([root[i] / np.linalg.norm(frame1scaled_n[i][0])
               for i in range(len(frame1scaled_n))] +
              [root[i] / np.linalg.norm(frame2scaled_n[i][1])
               for i in range(len(frame2scaled_n))])
    return nroots


def correspondance_points(frame1scaled_n, frame2scaled_n, tol=TOL):
    """
    This method uses the computed scale factors to generate the marker
    position pairs in both coordinate frames.

    Returns:
        [(M_1 coordinates in frame 1, M_1 coordinates in frame 2),
         (M_2 coordinates in frame 1, M_2 coordinates in frame 2),
         ...
         ...]
    """
    frame1scaled, frame2scaled = \
            normalize_points(frame1scaled_n, frame2scaled_n)
    #print "Frame 1 normalized:", frame1scaled 
    #print "Frame 2 normalized:", frame2scaled
    possible_scales = su.find_scale_factors(frame1scaled, frame2scaled,
                                            tol=tol)
    points = _corr_points_from_scales(possible_scales, frame1scaled,
                                    frame2scaled)
    return points

# create a better alias for correspondance_points
resolve_scale_ambiguity = correspondance_points

solve_mutual_localization = correspondence

def _corr_points_from_scales(possible_scales, frame1scaled,frame2scaled):
    corr_points = list()
    for solutions in possible_scales:
        corrp = ([(pi*solutions[i], qi)
             for i, (pi, qi) in enumerate(frame1scaled)]
            + [(pi, qi*solutions[i]) 
               for i, (pi, qi) in enumerate(frame2scaled, len(frame1scaled))])
        corr_points.append(corrp)
    return corr_points

def normalize_points(frame1scaled_n, frame2scaled_n):
    frame1scaled = [(np.array(f1)/np.linalg.norm(f1), np.array(f2))
                    for f1, f2 in frame1scaled_n]
    frame2scaled = [(np.array(f1), np.array(f2)/np.linalg.norm(f2))
                    for f1, f2 in frame2scaled_n]
    return frame1scaled, frame2scaled

