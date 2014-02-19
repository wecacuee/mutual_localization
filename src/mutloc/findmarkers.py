import numpy as np
import cv2

import config
import scipy.ndimage as ndimage
from scipy.misc import imrotate
from datetime import date
import matplotlib.pyplot as plt
import scipy
import itertools
import matplotlib.mlab as mlab
import os

import log
import logging
logger = log.getLogger(__name__)
logger.setLevel(logging.WARN)

DEBUG = False
HUE_TOL = 7
TOL=1e-6
MIN_CONTOUR_POINTS = 35
"""Contours with less then these points will be ignored"""
MIN_PIXELS = 100
#REL_MAX_PIXELS = 0.25

class Visualizer(object):
    region_colors = [(50, 180, 50), (50, 50, 180)]
    point_colors = [(0, 255, 0), (0, 255, 0)]
    def __init__(self):
        self.current_frame = None
        self._nth_region = 0
        self._nth_point = 0

    def current_tag(self, tag):
        self.current_frame.tag = tag

    def next_region_color(self):
        color = self.region_colors[self._nth_region % 2]
        self._nth_region += 1
        return color

    def next_point_color(self):
        color = self.point_colors[self._nth_point % 2]
        self._nth_point += 1
        return color

    def show_region(self, region):
        self.current_frame.show_region(region, self.next_region_color())

    def show_points(self, points):
        for p in points:
            self.current_frame.show_point(p, self.next_point_color())

    def skip_frame(self):
        self.current_frame = None

    def new_frame(self, base_img, tag):
        if self.current_frame:
            self.current_frame.flush()
        self.current_frame = VisualizerFrame(base_img, tag)

visualizer = Visualizer()

def threshold(imghsv, huelowerb, hueupperb):
    return cv2.inRange(imghsv,
                       np.array([huelowerb, 0., 0.]),
                       np.array([hueupperb, 255., 255.]))

def in_range(coord, range):
    start, end = range
    return (all(c > s for c, s in zip(coord, start))
            and all(c < e for c, e in zip(coord, end)))

def tohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def hueavg(imghsv, range_coordinates):
    start, end = range_coordinates
    slice = imghsv[start[1]:end[1], start[0]:end[0], 0]
    return np.mean(slice)

class MarkerNotFoundException(Exception):
    pass

def estimate_hueaverages(observed_img, observed_index, conf):
    # Take hue average from the observed camera's images
    obsdhsv = tohsv(observed_img)
    hueaverages = [hueavg(obsdhsv, range)
                   for range in conf.color_reference_marker()[observed_index]]
    return hueaverages

def blob_center_opencv(threshed_img, ignore_range):
    return BlobCenterOpenCV().blob_center(threshed_img, ignore_range)

def blob_center_scipy(threshed_img, ignore_range):
    return BlobCenterScipy().blob_center(threshed_img, ignore_range)

def find_color_balls_given_hue(observer_img,
                               hueavg,
                               blob_center_func=blob_center_scipy,
                               ignore_range=np.zeros((2,2)),
                               tag='c'):
    threshed_img = threshold(tohsv(observer_img),
                             hueavg - HUE_TOL,
                             hueavg + HUE_TOL)
    if DEBUG:debug_img('%s_threshed' % tag, threshed_img)
    center = blob_center_func(threshed_img, ignore_range=ignore_range)
    return center

def find_colored_balls(images, observer_index, observed_index, conf,
                       blob_center_func=blob_center_scipy, tag='c'):
    observed_img = images[observed_index]
    hueaverages = estimate_hueaverages(observed_img, observed_index, conf)
    centers = [find_color_balls_given_hue(images[observer_index],
                                          havg,
                                          blob_center_func=blob_center_func,
                                          ignore_range=marker_pos, tag=tag)
               for havg, marker_pos in zip(hueaverages,
                                           conf.color_reference_marker()[observer_index])]
    return centers

class MarkerFinder(object):
    def __init__(self, images):
        self._images = images

    def blob_center(self, threshed_img, ignore_range):
        return self.blobCenter.blob_center(threshed_img, ignore_range)

    def find_markers(self, observer_index, observed_index):
        imges = self._images
        return find_colored_balls(images, observer_index, observed_index,
                                 self.blob_center)

class BlobCenterOpenCV(object):
    def blob_center(self, threshed, ignore_range):
        contours, hier = cv2.findContours(
            threshed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area_centroids = [(cv2.contourArea(c), np.mean(c, axis=0)[0])
                            for c in contours]
        filtered_centers = [(l, c) for l, c in area_centroids
                            if (l > MIN_CONTOUR_POINTS)
                            and not in_range(c, ignore_range)]
        if not len(filtered_centers):
            raise MarkerNotFoundException("Unable to find marker in the image")
        center = max(filtered_centers, key=lambda x:x[0])[1]
        return center

def mask(labeled):
    mask_ = np.zeros_like(labeled, dtype=np.bool)
    mask_[:, 0] = True
    mask_[:, -1] = True
    mask_[0, :] = True
    mask_[-1, :] = True
    return mask_

def on_edge(labeled, id):
    id_img = (labeled == id)
    return np.sum(id_img & mask(labeled))

class BlobCenterScipy(object):

    def blob_center(self, threshed, ignore_range):
        def center(labeled, id):
            center_row_col = np.array([np.average(lri)
                             for lri in np.where(labeled == id)])
            center_x_y = center_row_col[::-1]
            return center_x_y

        def area(labeled, id):
            # calculate areas
            return np.sum(labeled == id)


        labeled, n_labels = ndimage.label(threshed)
        if n_labels == 0:
            raise MarkerNotFoundException("Unable to find marker in the image")
        centroids_area = [(center(labeled, id), area(labeled, id))
                          for id in range(1, n_labels + 1)
                          if not on_edge(labeled, id)]

        filtered_centers = [(c, a) for c, a in centroids_area
                            if not in_range(c, ignore_range)
                            and a > MIN_PIXELS]
        if len(filtered_centers) == 0:
            raise MarkerNotFoundException("Unable to find marker in the image")

        center_with_max_area, area = max(filtered_centers,
                                         key=lambda x:-1 * x[1])
        return center_with_max_area

class MarkerFinderOpenCV(MarkerFinder):
    blobCenter = BlobCenterOpenCV()

class MarkerFinderScipy(MarkerFinder):
    blobCenter = BlobCenterScipy()

def find_markers(images, observer_index, observed_index):
    return MarkerFinderOpenCV(images).find_markers(observer_index,
                                                  observed_index)

class VisualizerFrame(object):
    def __init__(self, base_img, tag):
        self.vis = base_img.copy()
        self.tag = tag

    def show_region(self, region, color):
        rows, cols = region
        self.vis[rows, cols, :] = color

    def show_point(self, point, color):
        x, y = point
        self.vis[y:y+2, x:x+2, :] = color

    def flush(self):
        fname = "~/to/debug%s.png" % self.tag
        cv2.imwrite(os.path.expanduser(fname), self.vis)

####################################################
# Barcode detection
####################################################

def preprocess(img, tag='c'):
    #img = cv2.imread(fname)
    #debug_img('%s_orig' % tag, img)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_g.copy())
    img_gf = ndimage.gaussian_filter(img_eq, 1.0)
    return img_gf

def flatten_img(img):
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    return vis

def plot_lines(img, lines, window='c'):
    vis = flatten_img(img)
    for p1x, p1y, p2x, p2y in lines:
        cv2.line(vis, (p1x, p1y), (p2x, p2y), (255, 0, 0), thickness=1)

    if DEBUG:debug_img(window, vis)

def debug_img(tag, img):
    if img is None:
        raise RuntimeError("None")
    fname = "~/to/%s/%s.png" % (date.today().isoformat(), tag)
    fname = os.path.expanduser(fname)
    dirn = os.path.dirname(fname)
    if not os.path.exists(dirn):
        os.makedirs(dirn)
    scipy.misc.imsave(fname, img)

def plot_points(img, points, window='c'):
    vis = flatten_img(img)
    points = np.asarray(points, np.int)
    for pt in points:
        cv2.rectangle(vis, tuple(pt), tuple(pt), color=(255, 0, 0), thickness=1)
    if DEBUG:debug_img(window, vis)


class BlackNWhiteBarcodeDetection(object):
    def __init__(self, img, tag, canny_params):
        self.count = 0
        self._img = img
        self.tag = tag
        self.canny_params = canny_params

    @property
    def img(self):
        return self._img

    @img.setter
    def set_img(self, img):
        self.tag = ('d%03d' % self.count)
        self.count += 1
        self._img = img

    def plot_lines(self, vis, lines, window='c'):
        plot_lines(vis, lines, window)

    def debug_img(self, window, vis):
        debug_img(window, vis)

    def find_markers(self):
        img = preprocess(self.img)
        st, end = self.rotate_and_find(img, self.tag)
        if DEBUG: self.plot_lines(cv2.Canny(self.img, *self.canny_params),
                                  np.hstack((st[:, ::-1], end[:, ::-1])),
                                  window='%s_edge' % self.tag)
        if DEBUG: self.plot_lines(self.img, np.hstack((st[:, ::-1], end[:, ::-1])),
                                  window=self.tag)
        return np.median(st, axis=0), np.median(end, axis=0)


    def unsquarify(self, pts, squareim, rectim):
        dim, dim = squareim.shape
        pts = pts - dim/2. # shift the origin to center of the image
        pts[:, 0] += rectim.shape[0]/2.
        pts[:, 1] += rectim.shape[1]/2.
        return np.array(pts, dtype=np.int)

    def rotate_back(self, pts, squareim, deg):
        dim, dim = squareim.shape
        pts = pts - dim/2. # shift the origin to center of the image
        rad = np.pi * deg / 180
        rotation_mat = np.array(
            [[np.cos(-rad), -np.sin(-rad)],
             [np.sin(-rad), np.cos(-rad)]])
        pts = np.array(np.dot(rotation_mat, pts.T), dtype=np.int).T # rotate
        return pts + dim/2

    def original_coord(self, pts, deg, squareim, rectim):
        return self.unsquarify(self.rotate_back(pts, squareim, deg),
                               squareim, 
                               rectim)

    def grayaverage(self, img, xinds, yinds):
        grayavg = list()
        for i in range(len(yinds) - 1):
            ga = np.average(img[np.arange(xinds[i],xinds[i+1] + 1).reshape(1,-1),
                                np.arange(yinds[i], yinds[i+1]).reshape(-1, 1)])
            grayavg.append(ga)

        return np.array(grayavg)

    def sliding_windows(self, arr, windowsize):
        arr = arr.reshape(-1) # make 1d
        wins = np.lib.stride_tricks.as_strided(arr,
                                               shape=(arr.size, windowsize),
                                               strides=(arr.strides[0],
                                                        arr.strides[0]))
        return wins[:(1 - windowsize)]

    def find_bands(self, img, edge_im, min_separation=10, max_separation=200,
                    rel_var_threshold=0.05, bands=7, tag='c'):
        # edge index
        xinds, yinds = np.where(edge_im)
        grayavg = self.grayaverage(img, xinds, yinds)

        # separation between edges
        yseps = np.diff(yinds)
        # windows of 'bands' size
        ywins = self.sliding_windows(yseps, bands)
        xwins = self.sliding_windows(xinds[:-1], bands)
        graywins = self.sliding_windows(grayavg, bands)
        
        yptp = np.ptp(ywins, axis=1)
        xptp = np.ptp(xwins, axis=1)

        avg_sep = np.average(ywins, axis=1)
        rel_ptp = yptp / avg_sep
        same_row_min_max = ((xptp == 0)
                            & (avg_sep > min_separation)
                            & (avg_sep < max_separation)
                            & (graywins[:, 0] < 100)
                            & (graywins[:, -1] < 100)
                            & (rel_ptp < 0.3)
                           )
        inds, = np.where(same_row_min_max)
        #print("Gray scale values %s" % str(graywins[inds, 0]))
        return (np.vstack((xinds[inds], yinds[inds])).T, 
                np.vstack((xinds[inds + bands], yinds[inds + bands])).T)
                #min_ptp,
                #avg_sep[inds],
                #graywins[inds])

    def im_make_square(self, img):
        xlen, ylen = img.shape
        maxlen = np.linalg.norm(img.shape)
        newimg = np.zeros((maxlen, maxlen), dtype=img.dtype)
        xstart, ystart = int(maxlen/2 - xlen/2), int(maxlen/2 - ylen/2)
        newimg[xstart:xstart + xlen, ystart:ystart + ylen] = img
        return newimg

    def rotate_and_find(self, img_gf, tag='c'):
        squareim = self.im_make_square(img_gf)
        found_pts = list()
        for deg in np.linspace(0, 90, num=36):
            rotim = imrotate(squareim, deg)
            edge_rotim = cv2.Canny(rotim, *self.canny_params)
            bands_param = self.find_bands(rotim, edge_rotim,
                                     tag='%s_deg-%d' % (tag, deg))
            found_pts.append(tuple([deg] + list(bands_param)))

        deg, bands_st, bands_end = max(found_pts, key=lambda x:len(x[1]))
        #print("[%s] Min ptp : %f. Avg sep : %s. Gray: %s" % (tag, rel_ptp,
        #                                                     str(avg_sep),
        #                                                     str(firstgray)))
        if not len(bands_st):
            raise MarkerNotFoundException()
        if DEBUG:self.debug_img('%s_rot' % tag, cv2.Canny(
            imrotate(squareim, deg), *self.canny_params))
        return (self.original_coord(bands_st, deg, squareim, img_gf),
                self.original_coord(bands_end, deg, squareim, img_gf))

def find_barcode(img, tag, canny_params=(80, 240)):
    detector = BlackNWhiteBarcodeDetection(img, tag,
                                           canny_params=canny_params)
    return detector.find_markers()
