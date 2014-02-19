#!/usr/bin/env python
import roslib; roslib.load_manifest('mutloc_ros')
import rospy
import os
import tf
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from mutloc import filewriters, mayaviutils
import tf.transformations as tfms

def transform_from_quat(pos, quat):
    return tfms.concatenate_matrices(tfms.translation_matrix(pos),
                                     tfms.quaternion_matrix(quat))

def append(writer, filename,  trans, quat):
    if os.path.exists(filename):
        lines = writer.load(filename)
        num = len(lines)
        num += 1
    else:
        lines = []
        num = 0

    lines.append(("%d" % num, np.asarray(trans), np.asarray(quat)))
    print("Writing lines to {0}".format(filename))
    writer.dump(lines, filename)
    Tlistartoolkit = [transform_from_quat(trans, quat)
                   for tag, trans, quat in lines]
    mayaviutils.plot_coordinate_transforms(*Tlistartoolkit)
    return len(lines)
    
class TfSaver(object):
    def __init__(self, resultsout):
        self.winname = "c"
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.onmouse)
        self.cvbridge = CvBridge()
        rospy.Subscriber("image_color", Image, self.callback)
        self.listener = tf.TransformListener()
        self.writer = filewriters.NumpyTxt()
        self.fname = resultsout

    def onmouse(self, event, x, y, flags, param):
        if flags & cv2.EVENT_FLAG_LBUTTON:
            print("mouse clicked")
            frame_id = self.last_image_message.header.frame_id
            try:
                (trans,rot) = self.listener.lookupTransform('/ar_marker',
                                                            frame_id,
                                                            rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException,
                    tf.ExtrapolationException), e:
                print e.msg
            num = append(self.writer, self.fname, trans, rot)
            imgname = os.path.join(os.path.dirname(self.fname), "img%04d.png"
                                  % num)
            print("Writing image to {0}".format(imgname))
            cv2.imwrite(imgname, self.lastcvimage)
            print trans,rot

    def callback(self, image_message):
        try:
            cv_image = np.array(self.cvbridge.imgmsg_to_cv(image_message,
                                                               desired_encoding="passthrough"))
        except CvBridgeError, e:
            print e

        self.lastcvimage = cv_image
        self.last_image_message = image_message
        cv2.imshow("c", cv_image)
        if (cv2.waitKey(30) > 0): return

def main(fname):
    rospy.init_node('savearpose', anonymous=True)
    tfsaver = TfSaver(fname)
    rospy.spin()

if __name__ == '__main__':
    import sys
    resultsout = sys.argv[1]
    main(resultsout)
