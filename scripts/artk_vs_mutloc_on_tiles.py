from mutloc.filewriters import NumpyTxt
import transformations as tf
import compareartoolkit as cmptk
import numpy as np
import matplotlib.pyplot as plt
import mutloc.mayaviutils as mayaviutils
import mutloc.filewriters as filewriters
import mutloc
import mayavi.mlab as mlab
import os

def transform_from_quat(pos, quat):
    return tf.concatenate_matrices(tf.translation_matrix(pos),
                                     tf.quaternion_matrix(quat))

def generate_ground_truth():
    origin = np.zeros(3)
    first_pos = np.array([0, 0, 0.86])
    Dx_Dz = np.array([[0.305, 0, 0], [0, 0, 0.305]]).T

    positions = list()

    positions.append(first_pos + np.dot(Dx_Dz, [0,  0])) # 0
    positions.append(first_pos + np.dot(Dx_Dz, [0,  0])) # 1
    positions.append(first_pos + np.dot(Dx_Dz, [0,  0])) # 2
    positions.append(first_pos + np.dot(Dx_Dz, [0,  0])) # 3
    positions.append(first_pos + np.dot(Dx_Dz, [1,  0])) # 4

    positions.append(first_pos + np.dot(Dx_Dz, [1,  1])) # 5
    positions.append(first_pos + np.dot(Dx_Dz, [0,  1])) # 6
    positions.append(first_pos + np.dot(Dx_Dz, [-1, 1])) # 7

    positions.append(first_pos + np.dot(Dx_Dz, [2,  2])) # 8
    positions.append(first_pos + np.dot(Dx_Dz, [1,  2])) # 9
    positions.append(first_pos + np.dot(Dx_Dz, [0,  2])) # 10
    positions.append(first_pos + np.dot(Dx_Dz, [-1, 2])) # 11
    positions.append(first_pos + np.dot(Dx_Dz, [-2, 2])) # 12

    positions.append(first_pos + np.dot(Dx_Dz, [2,  3])) # 13
    positions.append(first_pos + np.dot(Dx_Dz, [1,  3])) # 14
    positions.append(first_pos + np.dot(Dx_Dz, [0,  3])) # 15
    positions.append(first_pos + np.dot(Dx_Dz, [-1, 3])) # 16
    positions.append(first_pos + np.dot(Dx_Dz, [-2, 3])) # 17

    positions.append(first_pos + np.dot(Dx_Dz, [3,  4])) # 18
    positions.append(first_pos + np.dot(Dx_Dz, [2,  4])) # 19
    positions.append(first_pos + np.dot(Dx_Dz, [1,  4])) # 20
    positions.append(first_pos + np.dot(Dx_Dz, [0,  4])) # 21
    positions.append(first_pos + np.dot(Dx_Dz, [0,  4])) # 22
    positions.append(first_pos + np.dot(Dx_Dz, [-1, 4])) # 23
    positions.append(first_pos + np.dot(Dx_Dz, [-2, 4])) # 24
    positions.append(first_pos + np.dot(Dx_Dz, [-2, 4])) # 25
    positions.append(first_pos + np.dot(Dx_Dz, [-3, 4])) # 26

    positions.append(first_pos + np.dot(Dx_Dz, [3,  5])) # 27
    positions.append(first_pos + np.dot(Dx_Dz, [2,  5])) # 28
    positions.append(first_pos + np.dot(Dx_Dz, [2,  5])) # 29
    positions.append(first_pos + np.dot(Dx_Dz, [1,  5])) # 30
    positions.append(first_pos + np.dot(Dx_Dz, [0,  5])) # 31

    positions.append(first_pos + np.dot(Dx_Dz, [-1, 6])) # 32
    positions.append(first_pos + np.dot(Dx_Dz, [-2, 6])) # 33
    positions.append(first_pos + np.dot(Dx_Dz, [-3, 6])) # 34

    return [transform_from_quat(pos, np.array([0, 0, 1, 0])) for pos in
            positions]

@mlab.show
def mplot(Tlistgroundtruth, Tlistmutloc, Tlistartk):
    mayaviutils._plot_coordinate_transforms(*Tlistmutloc, tag='m') 
    mayaviutils._plot_coordinate_transforms(*Tlistgroundtruth, tag='g') 
    mayaviutils._plot_coordinate_transforms(*Tlistartk, tag='a') 

def transform(R, t):
    Tr = tf.identity_matrix()
    Tr[:3, :3] = R
    return tf.concatenate_matrices(Tr, tf.translation_matrix(t))

def main(out_file):
    bundleout = os.path.join(os.path.dirname(out_file),
                             'bundler-tmp/bundle/bundle.out')
    configfile = os.path.join(os.path.dirname(out_file), 'calib.yml')

    lines = NumpyTxt().load(out_file)

    Tlistgroundtruth = generate_ground_truth()
    outliers = []#[0, 2, 3, 21, 25, 29, 32, 33, 34]
    lines = [lines[i] for i in range(len(lines)) if i not in outliers]
    Tlistgroundtruth = [Tlistgroundtruth[i] for i in range(len(lines)) if i not in outliers]
    Tlistartk = [transform_from_quat(ta, qa)
               for tag, if0, d0, if1, d1, ta, qa, ti, qi, tm, qm in lines]
    Tlistartkinv = [tf.inverse_matrix(transform_from_quat(ti, qi))
               for tag, if0, d0, if1, d1, ta, qa, ti, qi, tm, qm in lines]
    cameraattrs = filewriters.BundlerReader().load(bundleout)
    Tlistbundler = [transform(ca['R'], ca['t']) 
                    for i, ca in enumerate(cameraattrs)
                    if i not in outliers]
    Tlistmutloc = [transform_from_quat(tm, qm)
               for tag, if0, d0, if1, d1, ta, qa, ti, qi, tm, qm in lines]
    conf = mutloc.config.Config(open(configfile))
    Tlistmutloc_comp = [mutloc.compute_best_transform(d0, d1, conf)
               for tag, if0, d0, if1, d1, ta, qa, ti, qi, tm, qm in lines]
    Tlistmutloc = Tlistmutloc_comp

    cmptk.plot_error_bars(Tlistgroundtruth, Tlistartk, Tlistmutloc,
                          Tlistbundler,
                          "media/tiles_turtlebots_artkvsmutloc_%s_errorbar_%%s.pdf",
                         scaling=True)
    cmptk.plot_error_bars(Tlistgroundtruth, Tlistartkinv, Tlistmutloc,
                          Tlistbundler,
                          "media/tiles_turtlebots_artkinv_vs_mutloc_%s_errorbar_%%s.pdf",
                         scaling=True)
    Tlistmutloc = cmptk.scale_and_translate(Tlistgroundtruth, Tlistmutloc)
    Tlistartk   = cmptk.scale_and_translate(Tlistgroundtruth, Tlistartk)
    mplot(Tlistgroundtruth, Tlistmutloc,Tlistartk)

if __name__ == '__main__':
    import sys
    try:
        out_file = sys.argv[1]
    except IndexError:
        print "Usage: python scripts/artk_vs_mutloc_on_tiles.py data/tiledexperiment_results/corrected_out.txt"
    main(out_file)
