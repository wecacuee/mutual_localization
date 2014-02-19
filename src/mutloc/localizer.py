import os
import itertools
import pickle
import itertools

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab

import mutloc
from mutloc.utils import absolute_path, apply_transform
import mutloc.surfmatcher as surfmatcher

BASELINE_THRESH=0.01 # radians
DEBUG = True
MAYAVI_PLOT = False
if MAYAVI_PLOT:
    from mutloc.mayaviutils import plot_triangulation, plot_mutloc
def plot_lines(vis, lines, window):
    mutloc.findmarkers.plot_lines(vis[..., ::-1], lines, window)


def plot_points(vis, points, window):
    mutloc.findmarkers.plot_points(vis[..., ::-1], points, window)

def init_plot(msg):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(msg)
    return fig, ax

def get_user_input_points(cvimg, n=1, msg=' '):
    fig, ax = init_plot(msg)
    ax.imshow(cvimg[..., ::-1])
    point0 = fig.ginput(n=n, timeout=-1)
    plt.close(fig)
    return point0

def triangulate_observation_pair(matrix_target_loc0, matrix_target_loc1, conf):
    T0, target0_loc = matrix_target_loc0
    T1, target1_loc = matrix_target_loc1
    K1 = conf.intrinsic_camera_matrix()[1]
    projM0 = np.dot(K1, T0[:3])
    projM1 = np.dot(K1, T1[:3])
    point4d = cv2.triangulatePoints(projM0, projM1,
                                    np.array(target0_loc).T,
                                    np.array(target1_loc).T)
    return (point4d[:3] / point4d[3]).T

class MarkerLocalizer(object):
    def localize_markers(self, loc, image0, image1, tag='c'):
        image0markers = mutloc.findmarkers.find_barcode(
            image0, tag="img0/%s" % loc.tag, canny_params=(50,240))
        image1markers = mutloc.findmarkers.find_barcode(
            image1, tag="img1/%s" % loc.tag, canny_params=(50,240))

        image0markers = [p[::-1] for p in image0markers]
        image1markers = [p[::-1] for p in image1markers]
        return image0markers, image1markers

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

class ManualMarkerLabel(object):
    def __init__(self, n_target_points):
        self._loaded_file = False
        self.n_target_points = n_target_points

    def _init(self, loc):
        self.markedimgs = self.loc.conf.marked_images()

    def localize_markers(self, loc, image0, image1, tag='c'):
        lines = read_markedimgs(markedimgs_file())
        image0markers = get_user_input_points(image0, n=2,
                                              msg='Pick two marker points')
        points = get_user_input_points(image1, n=(2 + self.n_target_points),
                                              msg='Pick two marker points and'
                                              + '{0} target '
                                              + 'points'.format(self.n_target_points))
        image1markers = points[:2]
        target_points = points[2:]
        return image0markers, image1markers

class LabeledMarkers(object):
    def __init__(self, iterable, keyfunc, markersfunc):
        self.img2markers = [
            (keyfunc(itr), markersfunc(itr)) for itr in iterable]

    def localize_markers(self, loc, image0, image1, tag='c'):
        for k, v in self.img2markers:
            if (k == image1).all():
                return v

class ColoredBallLocalizer(object):
    def __init__(self):
        pass
        
    def localize_markers(self, loc, image0, image1, tag='c'):
        images = image0, image1
        image0markers = [mutloc.findmarkers.find_color_balls_given_hue(
            image0, hue, tag='%s0' % tag)
            for hue in loc.conf.marker_hue()[0]]
        image1markers = [mutloc.findmarkers.find_color_balls_given_hue(
            image1, hue, tag='%s0' % tag)
            for hue in loc.conf.marker_hue()[1]]
        return image0markers, image1markers

class SurfPointMatcher(object):
    def hastarget(self, loc, img0, tag='c'):
        return True

    def matchpoints(self, loc, img0, img1, tag='c'):
        p1, p2, kp_points = surfmatcher.correspondence_points(img0, img1,
                                                              tag=tag)
        return p1, p2

class ManualPointMatcher(object):
    def matchpoints(self, loc, img0, img1, tag='c'):
        pass
        #image0 = loc.observation_queue[idx0][-1]
        #image1 = loc.observation_queue[idx1][-1]
        #point0 = get_user_input_points(image0,
        #                               msg='Pick one correspondence points')
        #point1 = get_user_input_points(image1,
        #                               msg='Pick the same correspondence point')
        #return point0, point1

class LabeledTarget(object):
    def __init__(self, img_target):
        self.img_target = img_target

    def hastarget(self, loc, img0, tag='c'):
        return True

    def matchpoints(self, loc, img0, img1, tag='c'):
        for img, target_loc in self.img_target:
            if np.all(img == img0):
                target_loc0 =  target_loc
            elif np.all(img == img1):
                target_loc1 =  target_loc
        return np.asarray(target_loc0), np.asarray(target_loc1)

class ColoredBallTarget(object):
    def __init__(self, config):
        self.target_pos = config.target_pos()
        self.target_hue = config.target_hue()

    def hastarget(self, loc, img0, tag='c'):
        try:
            pts0 = mutloc.findmarkers.find_color_balls_given_hue(
                img0,
                self.target_hue, 
                tag='%s0_target' % tag)
        except mutloc.findmarkers.MarkerNotFoundException, e:
            return False
        return True

    def matchpoints(self, loc, img0, img1, tag='c'):
        try:
            pts0 = mutloc.findmarkers.find_color_balls_given_hue(
                img0,
                self.target_hue, 
                tag='%s0_target' % tag)
            pts1 = mutloc.findmarkers.find_color_balls_given_hue(
                img1,
                self.target_hue,
                tag='%s1_target' % tag)
            return pts0.reshape(-1, 2), pts1.reshape(-1, 2)
        except mutloc.findmarkers.MarkerNotFoundException, e:
            raise UnableToLocalizeTargetException('Target not found')

class UnableToLocalizeTargetException(Exception):
    """to be raised if target is not localized"""
    def __init__(self, reason):
        self.msg = reason

def point2linedistance(point, (linest, lineend)):
     return np.linalg.norm(np.cross(point - linest, lineend - linest)) / \
             np.linalg.norm(lineend - linest)

def angularbaseline(T0, T1, optical_axis=[0, 0, 1]):
    T0, T1 = [np.linalg.inv(T) for T in (T0, T1)]
    oax0, oax1 = [apply_transform(T, optical_axis)
                  for T in (T0, T1)]
    o0, o1 = [apply_transform(T, np.zeros(3))
                  for T in (T0, T1)]
    baseline_rad = np.pi - np.arccos(np.dot(oax0 - o0, o1 - o0)) - \
            np.arccos(np.dot(oax1 - o1, o0 - o1))
    return baseline_rad

def baseline(T0, T1, optical_axis=[0, 0, 1]):
    T0, T1 = [np.linalg.inv(T) for T in (T0, T1)]
    optic_axis_frame0 = apply_transform(T0, optical_axis)
    origin0, origin1  = [apply_transform(T, [0, 0, 0])
                         for T in (T0, T1)]
    return point2linedistance(origin1, (optic_axis_frame0, origin0))

class MaxBaselineObservationPair(object):
    def choose_observation_pair(self, loc):
        observation_queue = loc.observation_queue
        if len(observation_queue) >= 2:
            obs1 = observation_queue[-1]
            obs0 = max(observation_queue[:-1],
                       key=lambda x:angularbaseline(x[1], obs1[1]))
            bline = angularbaseline(obs0[1], obs1[1])
            if bline < BASELINE_THRESH:
                raise UnableToLocalizeTargetException("Baseline=%f too small"
                                                      % bline)
            return obs0, obs1
        else:
            raise UnableToLocalizeTargetException("Not enough observations")
        return obs0, obs1

class RandomObservationPair(object):
    def choose_observation_pair(self, loc):
        observation_queue = loc.observation_queue
        if len(observation_queue) >= 2:
            possible_pairs = itertools.combinations(observation_queue, 2)
            possible_pairs = list(possible_pairs)
            for i in range(len(possible_pairs)):
                choiceidx = np.random.randint(0, len(possible_pairs))
                obs0, obs1 = possible_pairs[choiceidx]
                bline = angularbaseline(obs0[1], obs1[1])
                if bline  >= BASELINE_THRESH:
                    break
            if bline < BASELINE_THRESH:
                raise UnableToLocalizeTargetException("Baseline=%f too small"
                                                      % bline)
            return obs0, obs1
        else:
            raise UnableToLocalizeTargetException("Not enough observations")
        return obs0, obs1

class FirstAndLastObservationPair(object):
    def choose_observation_pair(self, loc):
        observation_queue = loc.observation_queue
        if len(observation_queue) >= 2:
            obs0, obs1 = observation_queue[0], observation_queue[-1]
            bline = angularbaseline(obs0[1], obs1[1])
            if bline < BASELINE_THRESH:
                raise UnableToLocalizeTargetException("Baseline=%f too small"
                                                      % bline)
            return obs0, obs1
        else:
            raise UnableToLocalizeTargetException("Not enough observations")
        return obs0, obs1

class Localizer(object):
    def __init__(self, config):
        self.conf = config
        self.tag = "c"
        self.observation_queue = []
        self.marker_localizer = MarkerLocalizer()
        self.pointmatcher = ManualPointMatcher()
        self.observationchoice = MaxBaselineObservationPair()

    def compute_relative_pose(self, image0, image1):
        image0markers, image1markers = \
                self.marker_localizer.localize_markers(
                    self, image0, image1, tag='%s/' % self.tag)
        if DEBUG:
            plot_lines(image0, np.array(image0markers, np.int).reshape(-1, 4),
                       window='%s0_markers' % self.tag)
            plot_lines(image1, np.array(image1markers, np.int).reshape(-1, 4),
                       window='%s1_markers' % self.tag)

        Tbest = mutloc.compute_best_transform(image0markers, image1markers,
                                       self.conf)
        if MAYAVI_PLOT:
            K0, K1 = self.conf.intrinsic_camera_matrix()
            truemarkers0, truemarkers1 = self.conf.markers()
            plot_mutloc((cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY), K0,
                         image0markers, truemarkers0),
                        (cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), K1,
                         image1markers, truemarkers1),
                        np.linalg.inv(Tbest))

        if self.pointmatcher.hastarget(self, image1, tag=self.tag):
            self.observation_queue.append((self.tag, Tbest, image1))
        return Tbest

    def choose_observation_pair(self):
        return self.observationchoice.choose_observation_pair(self)

    def triangulate_image_pair(self, obs0, obs1):
        tag0, T0, img0 = obs0
        tag1, T1, img1 = obs1

        t0, t1 = T0[:3, 3], T1[:3, 3]
        K0, K1 = self.conf.intrinsic_camera_matrix()

        target_pts0, target_pts1 = self.pointmatcher.matchpoints(
            self, img0, img1, tag='matchpoints/%s' % self.tag)

        if DEBUG:
            plot_points(img0, target_pts0,
                        window='target/%s0' % self.tag)
            plot_points(img1, target_pts1,
                        window='target/%s1' % self.tag)


        target_loc3d = triangulate_observation_pair(
            (T0, target_pts0), (T1, target_pts1), self.conf)
        #print("[%s, %s]Base line:%s" % (tag0, tag1, abs(t0[0] - t1[0])))
        #print("[%s, %s]Target loc:%s" % (tag0, tag1, str(target_loc3d)))
        if MAYAVI_PLOT:plot_triangulation((cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY),
                                     K1, np.linalg.inv(T0), target_pts0),
                                    (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                                     K1, np.linalg.inv(T1), target_pts1),
                                    target_loc3d)
        return target_loc3d

    def compute_point4d(self, image0, image1, tag):
        self.tag = tag
        rel_pose = self.compute_relative_pose(image0, image1)
        obs0, obs1 = self.choose_observation_pair()
        return self.triangulate_image_pair(obs0, obs1)


def target_localization(obs0, obs1, conf):
    image00markers, image01markers, target_pos0 = obs0
    T0 = mutloc.compute_best_transform(image00markers, image01markers, conf)

    image10markers, image11markers, target_pos1 = obs1
    T1 = mutloc.compute_best_transform(image10markers, image11markers, conf)
    return triangulate_observation_pair((T0, target_pos0),
                                        (T1, target_pos1),
                                        conf)
