#!/usr/bin/env python
import roslib; roslib.load_manifest('mutloc_ros')
from sensor_msgs.msg import Image
from approxsync import ApproximateSynchronizer
from mutloc_ros.cv_bridge import CvBridge
from tvtk.api import tvtk
#from visualiser import (ImageRenderer, axes_actor, InteractorStyleModified,
#                        vtkmatrix_from_array)
import tf.transformations as tfms
from mutloc import filewriters, config, mayaviutils
from mutloc_ros.srv import simple
import rospy
import tf
import message_filters
import numpy as np
import cv2
import os

from mutloc_ros.detector import Detector
from mutloc_ros.localize_from_detections import Localizer


def transform_from_quat(pos, quat):
    return tfms.concatenate_matrices(tfms.translation_matrix(pos),
                                     tfms.quaternion_matrix(quat))

class Writer(filewriters.NumpyTxt):
    def append(self, filename, imgbot03, detect03, imgbot04, detect04, pose_artk, pose_artk_inv, pose_mutloc):
        if os.path.exists(filename):
            lines = self.load(filename)
            num = len(lines)
            num += 1
        else:
            lines = []
            num = 0
        directory = os.path.dirname(filename)
        imgbot03fname = "imgbot03/%d.png" % num
        imgbot03path = os.path.join(directory, imgbot03fname)
        filewriters.mkdir(imgbot03path)
        cv2.imwrite(imgbot03path, imgbot03)

        imgbot04fname = "imgbot04/%d.png" % num
        imgbot04path = os.path.join(directory, imgbot04fname)
        filewriters.mkdir(imgbot04path)
        cv2.imwrite(imgbot04path, imgbot04)

        print "added %d lines to file" % num
        lines.append(("%d" % num, 
                      np.asarray(imgbot03fname),
                      detect03,
                      np.asarray(imgbot04fname),
                      detect04,
                      np.zeros(3) if pose_artk is None else pose_artk[0],
                      np.array([0, 1, 0, 0]) if pose_artk is None else pose_artk[1],
                      np.zeros(3) if pose_artk_inv is None else pose_artk_inv[0],
                      np.array([0, 1, 0, 0]) if pose_artk_inv is None else pose_artk_inv[1],
                      pose_mutloc[0],
                      pose_mutloc[1]))
        self.dump(lines, filename)
        return len(lines)

def plot_transform_from_output(lines):
    Tlistartk = [transform_from_quat(ta, qa)
               for tag, if0, d0, if1, d1, ta, qa, ti, qi, tm, qm in lines]
    Tlistartkinv = [transform_from_quat(ti, qi)
               for tag, if0, d0, if1, d1, ta, qa, ti, qi, tm, qm in lines]
    Tlistmutloc = [transform_from_quat(tm, qm)
               for tag, if0, d0, if1, d1, ta, qa, ti, qi, tm, qm in lines]

    mayaviutils.plot_coordinate_transforms(*Tlistartoolkit)

class VisualizeAndSave(object):
    def __init__(self, conf, fname):
        self.bridge = CvBridge()
        self.output_file = fname
        self.last_imgbot03 = None
        self.last_imgbot04 = None
        self.last_pose_mutloc = None
        self.last_pose_artk = None
        self.listener = tf.TransformListener()
        self.bot03_detector = Detector(detect_vis_topic='/turtlebot03/detect_vis')
        self.bot04_detector = Detector(detect_vis_topic='/turtlebot04/detect_vis')
        self.localizer = Localizer(conf)
        self.writer = Writer()
        self.winname = "c"
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.on_opencv_mouse_click)

    def get_tf(self, frame_id, img_frame_id, img_stamp):
        try:
            self.listener.waitForTransform(frame_id, img_frame_id, img_stamp, rospy.Duration(5))
            (trans,rot) = self.listener.lookupTransform(frame_id,
                                                        img_frame_id,
                                                        img_stamp)
            return (trans, rot)
        except (Exception, tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException), e:
            rospy.logwarn(e)

    def on_vtk_mouse_click(self, vtkRenWinInt, vtkRenderer, event):
        pass

    def on_opencv_mouse_click(self, event, x, y, flags, param):
        if flags & cv2.EVENT_FLAG_LBUTTON:
            print("mouse clicked")
            self.on_service_call("")

    def on_service_call(self, string):
        self.last_pose_artk = self.get_tf('/turtlebot03/ar_marker',
                                          self.bot04_frame_id,
                                          self.bot04_stamp)
        self.last_pose_artk_inv = self.get_tf('/turtlebot04/ar_marker',
                self.bot03_frame_id, self.bot03_stamp)

        self.last_pose_mutloc =  self.get_pose_from_mutloc()
        if self.last_pose_mutloc is None:
            return
        self.writer.append(self.output_file, 
                           self.last_imgbot03,
                           self.last_detect03,
                           self.last_imgbot04,
                           self.last_detect04,
                           self.last_pose_artk,
                           self.last_pose_artk_inv,
                           self.last_pose_mutloc)

    def get_pose_from_mutloc(self):
        T = self.localizer.compute_localization(self.last_detect03, self.last_detect04)
        if T is None:
            return
        q = tf.transformations.quaternion_from_matrix(T)
        translation = np.dot(T[:3, :3].T, -T[:3, 3])
        self.localizer.publish(T, self.bot04_stamp, self.bot04_frame_id)
        return translation, q

    def update_imgbot03(self, img):
        cv2.imshow("dont click", img)

    def update_imgbot04(self, img):
        cv2.imshow(self.winname, img)
        if (cv2.waitKey(30) > 0): return

    def roscallback(self, bot03_imgmsg, bot04_imgmsg):
        self.bot03_frame_id = bot03_imgmsg.header.frame_id
        self.bot03_stamp = bot03_imgmsg.header.stamp

        self.bot04_frame_id = bot04_imgmsg.header.frame_id
        self.bot04_stamp = bot04_imgmsg.header.stamp

        self.last_imgbot03 = self.bot03_detector.get_cv_img(bot03_imgmsg)
        self.last_detect03, vis03 = self.bot03_detector.get_detection(self.last_imgbot03)
        if len(self.last_detect03) >= 1:
            self.update_imgbot03(vis03)
            self.bot03_detector.publish(bot03_imgmsg, self.last_detect03, vis03)
        else:
            rospy.logdebug("did not find any detections.")

        self.last_imgbot04 = self.bot04_detector.get_cv_img(bot04_imgmsg)
        self.last_detect04, vis04 = self.bot04_detector.get_detection(self.last_imgbot04)
        if len(self.last_detect04) >= 1:
            self.update_imgbot04(vis04)
            self.bot04_detector.publish(bot04_imgmsg, self.last_detect04, vis04)
        else:
            rospy.logdebug("did not find any detections.")
       
        self.last_pose_mutloc =  self.get_pose_from_mutloc()

def main(configfile, resultsout):
    rospy.init_node('tilesexperiment')
    conf = config.Config(open(configfile))
    tfsaver = VisualizeAndSave(conf, resultsout)
    tbot03 = message_filters.Subscriber(
        rospy.resolve_name("turtlebot03/camera/rgb/image_raw"), Image)
    tbot04 = message_filters.Subscriber(
        rospy.resolve_name("turtlebot04/camera/rgb/image_raw"), Image)
    ts = ApproximateSynchronizer(0.5, [tbot03, tbot04], 10)
    ts.registerCallback(tfsaver.roscallback)
    #rospy.Service('tilesexperiment_saver', 
    #              simple,
    #              tfsaver.on_mouse_click)
    rospy.spin()

if __name__ == '__main__':
    import sys
    configfile = sys.argv[1]
    resultsout = sys.argv[2]
    main(configfile, resultsout)
