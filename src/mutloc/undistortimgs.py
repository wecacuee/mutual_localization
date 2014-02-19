import cv2
import os

from mutloc import config 

def undistortimg(img_file, K, dist_coeff, dir):
    img = cv2.imread(img_file)
    undist = cv2.undistort(img, K, dist_coeff)
    basename = os.path.basename(img_file)

    fname = os.path.join(dir, "undistorted", basename)
    print("Writing image to {0}".format(fname))
    cv2.imwrite(fname, undist)

def main(configfile):
    conf = config.Config(open(configfile))
    dir = os.path.dirname(configfile)
    img_file_format = conf.img_file_format()
    dist_coeffs = conf.distortion_coefficients()
    camera_matrices = conf.intrinsic_camera_matrix()
    for i in range(100):
        img_files = [os.path.join(dir, format % i)
                     for format in img_file_format]
        if not all(os.path.exists(f) for f in img_files):
            break
        for f, K, dc in zip(img_files, camera_matrices, dist_coeffs):
            undistortimg(f, K, dc, dir)

if __name__ == '__main__':
    import sys
    configfile = sys.argv[1]
    main(configfile)
