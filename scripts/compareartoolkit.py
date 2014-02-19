import numpy as np
import transformations as tf
from mutloc import mayaviutils
from mutloc import filewriters
from mutloc import utils
import alignpointcloud
import mayavi.mlab as mlab
import matplotlib.mlab as matmlab
import matplotlib.cm as colormap

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mutloc.utils import errorbar_from_3d, matplotlibusetex, newaxes

def transform(R, t):
    Tr = tf.identity_matrix()
    Tr[:3, :3] = R
    return tf.concatenate_matrices(Tr, tf.translation_matrix(t))

@mlab.show
def plot(uniq_transforms, newTbundler):
    mayaviutils._plot_coordinate_transforms(*newTbundler, tag='b')
    mayaviutils._plot_coordinate_transforms(*uniq_transforms, tag='m')

def groundtruth(fname):
    return filewriters.NumpyTxtHeader().load(fname)

def transform(pos, eulxyz):
    ex, ey, ez = eulxyz * np.pi / 180
    return tf.concatenate_matrices(tf.translation_matrix(pos),
                                   tf.euler_matrix(ex, ey, ez, 'sxyz'))
def ground_truth_transforms(fname):
    return [transform(pos, eulxyz) for tag, pos, eulxyz in groundtruth(fname)]

def transform_from_quat(pos, quat):
    T = tf.concatenate_matrices(tf.translation_matrix(pos),
                                   tf.quaternion_matrix(quat))
    return T

def scale_and_translate(Tlistgroundtruth, Tlistartoolkit):
    ground_truth_pointcloud = np.asarray([utils.apply_transform(T, np.zeros(3))
                                          for T in Tlistgroundtruth])
    artoolkit_pointcloud = np.asarray([utils.apply_transform(T, np.zeros(3))
                                    for T in Tlistartoolkit])
    Tscale = tf.affine_matrix_from_points(artoolkit_pointcloud.T,
                                          ground_truth_pointcloud.T,
                                          shear=False,
                                          scale=True)
    #print("Scaling by {0}".format(tf.scale_from_matrix(Tscale)[0]))
    try:
        print("translation by {0}".format(tf.translation_from_matrix(Tscale)))
        print("rotation by {0}".format(tf.rotation_from_matrix(Tscale)[0]))
    except ValueError:
        pass
    Tlistartoolkit = [tf.concatenate_matrices(Tscale, T) for T in Tlistartoolkit]
    return Tlistartoolkit


def compute_errors_given_transforms(Tlistgroundtruth, Tlistartoolkit,
                                    scaling=False):
    ground_truth_pointcloud = np.asarray([utils.apply_transform(T, np.zeros(3))
                                          for T in Tlistgroundtruth])
    artoolkit_pointcloud = np.asarray([utils.apply_transform(T, np.zeros(3))
                                    for T in Tlistartoolkit])
    if scaling:
        Tlistartoolkit = scale_and_translate(Tlistgroundtruth, Tlistartoolkit)
    artoolkit_pointcloud = np.asarray([utils.apply_transform(T, np.zeros(3)) 
                                    for T in Tlistartoolkit])
    trans_err = matmlab.vector_lengths(
        ground_truth_pointcloud - artoolkit_pointcloud, axis=1)
    rot_err = np.asarray(
        [180 / np.pi * utils.angle_between_rotations(Tgt[:3, :3], Tml[:3,:3])
         for Tgt, Tml in zip(Tlistgroundtruth, Tlistartoolkit)])
    rot_err = np.abs(rot_err - np.mean(rot_err))
    return ground_truth_pointcloud, artoolkit_pointcloud, trans_err, rot_err

def error_bar_translation_artk_mutloc(gtpc, artk_trans_err, mutloc_trans_err,
                                      bundler_trans_err,
                                     file_template):
    ax1 = newaxes('X (meters)', 'Translation Error (meters)',
                  'Translation error along X-axis',
                  subplot=111)
    ax2 = newaxes('Z (meters)', 'Translation Error (meters)',
                  'Translation error along Z-axis',
                  subplot=111)
    ax1, ax2 = errorbar_from_3d(
        gtpc[:, 0],
        gtpc[:, 2],
        artk_trans_err,
        axes=[ax1, ax2],
        ylabel='Error (meters)',
        xlabels=['X (meters)', 'Z (meters)'],
        titles=['Translation error along X-axis', 
                'Translation error along Z-axis'],
        color='b',
        label='ARTookit')
    ax1, ax2 = errorbar_from_3d(
        gtpc[:, 0],
        gtpc[:, 2],
        mutloc_trans_err,
        axes=[ax1, ax2],
        ylabel='Error (meters)',
        xlabels=['X (meters)', 'Z (meters)'],
        titles=['Translation error along X-axis', 
                'Translation error along Z-axis'],
        color='r',
        label='Mutual Localization')
    ax1, ax2 = errorbar_from_3d(
        gtpc[:, 0],
        gtpc[:, 2],
        bundler_trans_err,
        axes=[ax1, ax2],
        ylabel='Error (meters)',
        xlabels=['X (meters)', 'Z (meters)'],
        titles=['Translation error along X-axis', 
                'Translation error along Z-axis'],
        color='g',
        label='Bundler')
    ax1.legend(loc='upper center')
    #ax2.legend(loc='upper left', fontsize='xx-small')
    #ax1.set_yscale('log')
    #ax2.set_yscale('log')
    print("Writing graph %s " % (file_template % 'X'))
    ax1.figure.savefig(file_template % 'X') 
    print("Writing graph %s " % (file_template % 'Z'))
    ax2.figure.savefig(file_template % 'Z')


def error_bar_rotation_artk_mutloc(gtpc, artk_rot_err, mutloc_rot_err,
                                   bundler_rot_err,
                                  file_template):
    ax1 = newaxes('X (meters)', 'Rotation Error (degrees)',
                  'Rotation error along X-axis',
                  subplot=111)
    ax2 = newaxes('Z (meters)', 'Rotation Error (degrees)',
                  'Rotation error along Z-axis',
                  subplot=111)
    ax1, ax2 = errorbar_from_3d(
        gtpc[:, 0],
        gtpc[:, 2],
        artk_rot_err,
        axes=[ax1, ax2],
        ylabel='Error (degrees)',
        xlabels=['X (meters)', 'Z (meters)'],
        titles=['Rotation error along X-axis', 
                'Rotation error along Z-axis'],
        color='b',
        label='ARToolkit')
    ax1, ax2 = errorbar_from_3d(
        gtpc[:, 0],
        gtpc[:, 2],
        mutloc_rot_err,
        axes=[ax1, ax2],
        ylabel='Error (degrees)',
        xlabels=['X (meters)', 'Z (meters)'],
        titles=['Rotation error along X-axis', 
                'Rotation error along Z-axis'],
        color='r',
        label='Mutual Localization')
    ax1, ax2 = errorbar_from_3d(
        gtpc[:, 0],
        gtpc[:, 2],
        bundler_rot_err,
        axes=[ax1, ax2],
        ylabel='Error (degrees)',
        xlabels=['X (meters)', 'Z (meters)'],
        titles=['Rotation error along X-axis', 
                'Rotation error along Z-axis'],
        color='g',
        label='Bundler')
    ax1.legend(loc='upper center')
    #ax2.legend(loc='bottom left', fontsize='xx-small')
    #ax1.set_yscale('log')
    #ax2.set_yscale('log')
    ax1.figure.savefig(file_template % 'X')
    ax2.figure.savefig(file_template % 'Z')

def plot_error_bars(Tlistgroundtruth, Tlistartoolkit, Tlistmutloc, Tlistbundler,
                    file_template=None, scaling=False):
    gtpc, artkpc, artk_trans_err, artk_rot_err = \
            compute_errors_given_transforms(Tlistgroundtruth,
                                            Tlistartoolkit, scaling)
    gtpc, mutlocpc, mutloc_trans_err, mutloc_rot_err = \
            compute_errors_given_transforms(Tlistgroundtruth,
                                            Tlistmutloc, scaling)
    gtpc, bundlerpc, bundler_trans_err, bundler_rot_err = \
            compute_errors_given_transforms(Tlistgroundtruth,
                                            Tlistbundler, scaling)
    print("Mean, Median artk translation error : {0}, {1}".format(
        np.mean(artk_trans_err), np.median(artk_trans_err)))
    print("Mean, Median mutloc translation error : {0}, {1}".format(
        np.mean(mutloc_trans_err), np.median(mutloc_trans_err)))
    print("Mean, Median bundler translation error : {0}, {1}".format(
        np.mean(bundler_trans_err), np.median(bundler_trans_err)))
    matplotlibusetex()
    
    error_bar_translation_artk_mutloc(gtpc, artk_trans_err, mutloc_trans_err,
                                      bundler_trans_err,
                                      file_template % "transerr")
    error_bar_rotation_artk_mutloc(gtpc, artk_rot_err, mutloc_rot_err,
                                   bundler_rot_err,
                                   file_template % "roterr")
    print("Mean, Median artk rotation error : {0}, {1}".format(
        np.mean(artk_rot_err), np.median(artk_rot_err)))
    print("Mean, Median mutloc rotation error : {0}, {1}".format(
        np.mean(mutloc_rot_err), np.median(mutloc_rot_err)))
    print("Mean, Median bundler rotation error : {0}, {1}".format(
        np.mean(bundler_rot_err), np.median(bundler_rot_err)))

    #plt.show()

def main(gtfile, artoolkitout, mutlocout):
    Tlistgroundtruth = ground_truth_transforms(gtfile)
    artk_lines = filewriters.NumpyTxt().load(artoolkitout)
    Tlistartoolkit = [transform_from_quat(trans, quat)
                   for tag, trans, quat in artk_lines]
    mutloc_lines = filewriters.NumpyTxt().load(mutlocout)
    Tlistmutloc = [T for tag, T in mutloc_lines]

    plot(Tlistartoolkit, Tlistgroundtruth)
    plot_error_bars(Tlistgroundtruth, Tlistartoolkit, Tlistmutloc, 
                    "media/localizationonly-artkvsmutloc-%s-errorbar-%%s.pdf")



if __name__ == '__main__':
    # Sample usage
    # python scripts/compareartoolkit.py data/localizationonly/tiled_ground_truth.txt data/localizationonly/artoolkitresults.txt data/localizationonly/ordered_transforms.txt
    import sys
    groundtruthfile = sys.argv[1]
    artoolkitout = sys.argv[2]
    mutlocout = sys.argv[3]
    main(groundtruthfile, artoolkitout, mutlocout)
