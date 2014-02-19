import argparse
import os
import itertools
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab

import ransac
import mutloc
from mutloc import filewriters, mayaviutils, usbcam, mark_images, undistortimgs, config
from mutloc.utils import absolute_path

DEBUG = True

from mutloc.localizer import * # shoot me

def fname2tag(f):
    f = str(f)
    basen = os.path.basename(f)
    return basen.replace(".png", "").replace("static", "")

def enforcedtype(lines):
    mappedlines = [ 
        (f0, f1,
         np.asarray(image0markers, np.float32),
         np.asarray(image1markers, np.float32),
         np.asarray(corrpoint, np.float32).reshape(-1, 2))
        for (f0, f1, image0markers, image1markers, corrpoint) in lines]
    return mappedlines

def readlines_numpy(markedimgs_path):
    def readimg(f):
        f = str(f)
        return cv2.imread(absolute_path(f, relfile=markedimgs_path))

    reader = filewriters.NumpyTxt(include_dtype=True)
    lines = reader.load(markedimgs_path)
    mappedlines = enforcedtype(lines)
    mappedlines = [ 
        (fname2tag(f0), readimg(f0), readimg(f1), m0, m1, tp)
         for (f0, f1, m0, m1, tp) in mappedlines]
    return mappedlines

def abspath(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def datadir(configfile):
    return os.path.dirname(configfile)

def results_file(configfile):
    return os.path.join(datadir(configfile), 'results.txt')

def results_writer():
    return filewriters.NumpyTxt(include_dtype=True)

def capture_data(configfile):
    conf = config.Config(open(configfile))
    usbcam.main(file_formats=[conf.abspath(f) for f in conf.img_file_format()])
    undistortimgs.main(configfile)
    mark_images.main(configfile, None)

def translation(T0, T1):
    T0inv = np.linalg.inv(T0)
    return np.dot(T1, T0inv)[:3, 3]

def run_localizer(configfile):
    config = mutloc.config.Config(open(configfile))
    markedimgs_path = absolute_path(config.marked_images(), relfile=configfile)
    mappedlines = readlines_numpy(markedimgs_path)
    loc = Localizer(config)
    #loc.marker_localizer = LabeledMarkers(mappedlines,
    #                                      lambda x:x[0],
    #                                      lambda x:(x[3], x[4]))
    #loc.marker_localizer = MarkerLocalizer()
    loc.marker_localizer = LabeledMarkers(mappedlines,
                                          lambda x:x[2],
                                          lambda x:(x[3], x[4]))
    loc.pointmatcher = LabeledTarget([(r[2], r[5]) for r in mappedlines])
    #loc.pointmatcher = SurfPointMatcher()

    output = list()
    target_loc3d_gt = np.array([-0.08, -0.45, -1.12])
    pmvs_dir = os.path.join(os.path.dirname(configfile), "mutloc-pmvs")
    K0, K1 = config.intrinsic_camera_matrix()
    pmvs_writer = filewriters.PMVSWriter(K1, pmvs_dir=pmvs_dir)
    for tag, img0, img1, image0markers, image1markers, corrpoint in mappedlines:
        try:
            target_loc_gt = loc.conf.target_pos()
            ret = loc.compute_point4d(img0, img1, tag)
            (tag0, T0, _), (tag1, T1, _) = loc.choose_observation_pair()
            output.append((tag0, tag1, T0, T1, ret))
            print("[%s-%s]Position of target: %s" % (tag0, tag1,str(ret)))
            print("[%s-%s]Translation of mobile cam: %s" % (tag0,
                                                         tag1,str(translation(T0, T1))))
        except mutloc.findmarkers.MarkerNotFoundException, e:
            print("Unable to find all four markers")
            continue
        except UnableToLocalizeTargetException, e:
            print("Target not localized because '%s'" % e.msg)
        if pmvs_writer:
            _, Tlatest, _ = loc.observation_queue[-1]
            pmvs_writer.add_image(len(loc.observation_queue) - 1,
                                  Tlatest,
                                  img1)

    dump_ordered_transforms(loc.conf, loc.observation_queue)
    if pmvs_writer:
        pmvs_writer.write_options()
    plot_rows(output)
    results_writer().dump(
        output,
        results_file(configfile))

def dump_ordered_transforms(conf, observation_queue):
    fname = conf.abspath("order_transforms.txt")
    o_queue_without_imgs = [(tag, T) for tag, T, img in observation_queue]
    filewriters.NumpyTxt().dump(o_queue_without_imgs, fname)

def arr_uniq(arr_list):
    uniq_list = list()
    for arr in arr_list:
        if not any([(arr == a).all() for a in uniq_list]):
            uniq_list.append(arr)
    return uniq_list

@mlab.show
def plot_rows(rows):
    uniq_transforms = arr_uniq([r[2] for r in rows] + [r[3] for r in rows])
    mayaviutils._plot_coordinate_transforms(
        *[np.linalg.inv(T) for T in uniq_transforms])
    target_points = np.vstack([r[4] for r in rows])
    labels = ['%s-%s' % (str(rows[i/3][0])[5:7], str(rows[i/3][1])[5:7])
              for i in range(3  * len(rows))]
    mayaviutils._labeled_points3d(target_points, labels)

def plot_results(configfile):
    conf = config.Config(open(configfile))
    lines = results_writer().load(results_file(configfile))
    rows = [(tag0, tag1,
            np.asarray(T0, np.float),
            np.asarray(T1, np.float),
            np.asarray(ret, np.float))
            for tag0, tag1, T0, T1, ret in lines]

    tag0, tag1, T0, T1, ret = lines[0]
    print("[%s]Position of mobile cam: %s" % (tag0, str(T0[:3, 3])))
    for tag0, tag1, T0, T1, ret in lines:
        print("[%s-%s]Translation in of mobile cam: %s" % (tag0,
                                                           tag1,str(np.linalg.norm(
                                                               translation(T0, T1)))))
    feature_points = [o[4] for o in rows]
    if all(fp.shape == feature_points[0].shape for fp in feature_points):
        data = np.array(feature_points)
        print data
        print("Variance:{0}".format(np.std(data, axis=0)))
        print("Range:{0} <=> {1}".format(np.min(data[:, :, 2], axis=0),
                                         np.max(data[:, :, 2], axis=0)))
    #plot_rows(rows)
    if conf.ground_truth() and conf.transforms_file():
        comparegroundtruth.main(conf.ground_truth(), conf.transforms_file())

def getargs():
    parser = argparse.ArgumentParser(description='''
                                     Test mutloc.mutualcam.compute_transoform() on images''')
    parser.add_argument('configfile', type=str, nargs='?',
                        help='location of config file. Example: data/bnwmarker20120831/calib.yml')
    parser.add_argument('--plot_only', action='store_true',
                        help='only plot the grah from the output file')
    parser.add_argument('--online', action='store_true',
                        help='get results everytime you save and mark an image')
    return parser.parse_args()

def main():
    args = getargs()
    if not args.plot_only:
        if args.online:
            capture_data(args.configfile)
        run_localizer(args.configfile)
    plot_results(args.configfile)

if __name__ == '__main__':
    main()
