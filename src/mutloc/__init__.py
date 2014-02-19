import numpy as np
from scipy import optimize
import config
import transformations as tf
from utils import apply_transform, transform_inv
from findmarkers import find_markers
from core import correspondence

import log
import logging
logger = log.getLogger(__name__)
logger.setLevel(logging.WARN)

TOL = 1e-6
###################################################
# epipolar geometry related functions; possibly another module
###################################################
def relativecamerapose(frame0image_corr, frame1image_corr,
                        calibmat0, calibmat1, method):
    transformation = correspondence(frame0scaled, frame1scaled,
                                    method=method, tol=TOL)
    return transformation


####################################
# main function
####################################


def _distance_of_other_marker(T, m1):
    return np.linalg.norm(apply_transform(transform_inv(T), m1))

def filter_by_distance(Troots, conf):
    # All markers of frame1 should be farther away then the distance of frame0
    # markers. See test/data/two-camera/autotest/camera0/05-05.png and 
    # test/data/two-camera/autotest/experimental_05_05.blend for nearly
    # ambiguous configurations.
    markers0, markers1 = conf.markers()
    dm0 = np.linalg.norm(markers0[0])
    Tfiltered = [T for T in Troots
                 if all(_distance_of_other_marker(T, m1) > dm0 
                        for m1 in markers1)]
    return Tfiltered

def compute_transformation(img0, img1, method="analytic"):
    """
    img0 :: Image from camera 0
    img1 :: Image from camera 1
    """
    image0markers = find_markers((img0, img1), 0, 1)
    logger.info("Image0 markers: %s" % str(image0markers))
    image1markers = find_markers((img0, img1), 1, 0)
    logger.info("Image1 markers: %s" % str(image1markers))
    Troots = compute_transform_from_marker_pos(image0markers, image1markers,
                                           method=method)

                                                   
    return filter_by_distance(Troots)

def image_markers_to_vector_pairs(image0markers, image1markers,
                                  conf, method="analytic"):
    markers0, markers1 = conf.markers()
    frame0image_corr = zip(image0markers, markers1)
    frame1image_corr = zip(markers0, image1markers)
    K0, K1 = conf.intrinsic_camera_matrix()
    K0inv, K1inv = [np.linalg.inv(K) for K in (K0, K1)]
    frame0scaled = [(np.dot(K0inv, np.hstack((f0img, 1))), f1)
             for f0img, f1 in frame0image_corr]
    frame1scaled = [(f0, np.dot(K1inv, np.hstack((f1img, 1))))
             for f0, f1img in frame1image_corr]
    logger.info("Points: {0}".format(frame0scaled + frame1scaled))
    return frame0scaled, frame1scaled

def compute_transform_from_marker_pos(image0markers, image1markers,
                                      conf, method="analytic"):
    frame0scaled, frame1scaled = image_markers_to_vector_pairs(
        image0markers, image1markers, conf)

    transf = correspondence(frame0scaled, frame1scaled,
                            method=method, tol=TOL)
    return transf

def compute_best_transform(image0markers, image1markers, conf):
    Troots = compute_transform_from_marker_pos(image0markers,
                                               image1markers, conf,
                                               method='analytic')
    Tfiltered = filter_by_distance(Troots, conf)
    if len(Tfiltered):
        Tbest = Tfiltered[0]
    else:
        Tbest = Troots[0]
    return Tbest

def finetune(Tbest, image0markers, image1markers, conf):
    """
    Not used
    Intended to finetune the solution obtained for further accurancy
    """
    def proj(v):
        x, y, z = v
        return np.asarray([x/z, y/z])

    K0, K1 = conf.intrinsic_camera_matrix()
    pihat_s = image0markers
    qihat_s = image1markers
    pi_s, qi_s = conf.markers()
    R = Tbest[:3, :3]
    t = Tbest[:3, 3]
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
        r"""
        .. :math:
            \newcommand{\vect}[1]{\mathbf{#1}}
            \newcommand{\hvect}[1]{\hat{\vect{#1}}}
            \[
            (R^*, \vect{t}^*) &= \arg \min_{(R,t)}\\
                              & \sum_{i \in \{1, 2\}} \| \hvect{p_i} -
            f(K_pR^{-1}(\vect{q}_i - \vect{t})) \|^2\\
                            + & \sum_{i \in \{3, 4\}} \| \hvect{q_i} -
            f(K_q(R\vect{p}_i + \vect{t})) \|^2
            \]
        """
        Rinv = np.linalg.inv(R)
        ei_s = [np.linalg.norm(pihat -
                                 proj(np.dot(K0, np.dot(Rinv, qi - t))))
                for pihat, qi in zip(pihat_s, qi_s)]
        ei_s += [np.linalg.norm(qihat -
                                 proj(np.dot(K1, np.dot(R, pi) + t)))
                for qihat, pi in zip(qihat_s, pi_s)]
        return sum(ei_s)

    output = optimize.fmin_bfgs(optfunc,
                                x0=_pack(Tbest),
                                maxiter=20, 
                                full_output=True,
                                disp=1)
    Tnewbest = _unpack(output[0])
    err = output[1]
    print("Reprojection error in pixels {0}".format(output[1]))
    return Tnewbest
