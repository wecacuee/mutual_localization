"""
This script runs the experiment described in section V B of
MutualLocalization.pdf
"""
import os
import itertools
import pickle
import itertools
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import transformations as tf

import mutloc
from mutloc.utils import absolute_path, apply_transform
import mutloc.utils as utils
from mutloc.localizer import Localizer, ColoredBallLocalizer
import mutloc.filewriters as filewriters

def transformation_from_loc_quat(quat, loc):
    T = tf.quaternion_matrix(tf.quaternion_inverse(quat))
    T[:3, 3] = -1 * np.dot(T[:3, :3], loc)
    return T

def getargs():
    parser = argparse.ArgumentParser(description='''
                                     Test mutloc.mutualcam.compute_transoform()
                                     on images provided from blender''')
    parser.add_argument('configfile', type=str, nargs='?',
                        help='location of config file', 
                        default=absolute_path(
                            '../data/two-camera/20120903/calib.yml',
                            relfile=__file__))
    parser.add_argument('--justplot', action='store_true',
                        default=False,
                        help='Provide a regex to select images')
    return parser.parse_args()

def main():
    args = getargs()
    configfname = args.configfile
    config = mutloc.config.Config(open(configfname))
    index_file = absolute_path(config.image_index(), relfile=configfname)
    output_file = os.path.join(os.path.dirname(index_file),
                             'noise_vs_error_blender.txt')
    if not args.justplot:
        run_experiment(index_file, config, output_file)
    plot_graphs(output_file)

def run_experiment(index_file, config, output_file):
    loc = Localizer(config)
    loc.marker_localizer = ColoredBallLocalizer()
    loc.pointmatcher = None
    input_lines = []
    with open(index_file, 'r') as indexf:
        input_lines = list(indexf)
    output_lines = list()
    for n in [2, 4, 6, 8, 10]:
        e_lin, e_rot = error_with_noise(loc, index_file,  input_lines, noise_sigma=n)
        output_lines.append((n, e_lin, e_rot))
    writer = filewriters.NumpyTxt(include_dtype=True)
    writer.dump(output_lines, output_file)



def err_lin(T_gt, T_est):
    return np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3])

def err_rot_degree(T_gt, T_est):
    return np.arccos((np.trace(np.dot(T_gt[:3, :3].T, T_est[:3, :3])) - 1) /
                     2.) * 180 / np.pi

def error_with_noise(loc, image_index, input_lines, noise_sigma=1):
    e_lin = list()
    e_rot = list()
    for i, line in enumerate(input_lines):
        (_, _, depth, beta0, beta1, 
         trans_str, rot_str, 
         img0path, img1path) = line.strip().split("\t")
        tag=img0path
        image0 = cv2.imread(absolute_path(img0path, relfile=image_index))
        image1 = cv2.imread(absolute_path(img1path, relfile=image_index))
        trans4d = eval(trans_str)
        quat = eval(rot_str)
        T_gt = transformation_from_loc_quat(quat, trans4d[:3])
        try:
            image0markers, image1markers = \
                    loc.marker_localizer.localize_markers(
                        loc, image0, image1, tag='%s/' % loc.tag)
        except mutloc.findmarkers.MarkerNotFoundException, e:
            print("[%s]Unable to find all four markers" % tag)
            continue
        noise = np.random.normal(loc=0.0, scale=noise_sigma,
                                    size=(2, 2))
        # perturb the detected marker location with gaussian noise
        image0markers += noise
        noise = np.random.normal(loc=0.0, scale=noise_sigma,
                                    size=(2, 2))
        # perturb the detected marker location with gaussian noise
        image1markers += noise

        T_est = mutloc.compute_best_transform(image0markers, image1markers,
                                       loc.conf)
        e_lin.append(err_lin(T_gt, T_est))
        e_rot.append(err_rot_degree(T_gt, T_est))
    return np.mean(e_lin), np.mean(e_rot)

def plot_graphs(output_file):
    noise_vs_err = filewriters.NumpyTxt().load(output_file)
    noise_vs_err = np.array(noise_vs_err)
    utils.matplotlibusetex()
    ax = utils.newaxes('Noise (pixels)', 'Translation Error (m)',
                      'Translation error with noise')
    ax.plot(noise_vs_err[:, 0], noise_vs_err[:, 1])
    trans_file = 'media/blender-transerr-vs-noise.pdf'
    print('Writing graph to %s' % trans_file)
    ax.figure.savefig(trans_file)

    ax = utils.newaxes('Noise (pixels)', 'Rotation Error (degrees)',
                      'Rotation error with noise')
    ax.plot(noise_vs_err[:, 0], noise_vs_err[:, 2])
    rot_file = 'media/blender-roterr-vs-noise.pdf'
    print('Writing graph to %s' % rot_file)
    ax.figure.savefig(rot_file)

if __name__ == '__main__':
    main()


