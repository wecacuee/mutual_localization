#!/usr/bin/python
import roslib; roslib.load_manifest('mutloc_ros')
import rospy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import cv_bridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from approxsync import ApproximateSynchronizer
import numpy as np
import cv
from mutloc.blinkingtracker import BlinkingMarkerFinder
from mutloc_ros.cv_bridge import CvBridge


class Get3DPosition(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.marker_finder = BlinkingMarkerFinder()
        self.Kinv = None

    def pt3d_from_imgpt(self, lpt, lz):
        lpt3d = np.dot(self.Kinv, (lpt[0], lpt[1], 1))
        lpt3d *= lz/lpt3d[2] # set scale factor so that z coord is lz
        return lpt3d

    def set_camera_info(self, caminfo):
        self._caminfo = caminfo
        K = np.asarray(caminfo.K).reshape(3, 3)
        Kinv = np.linalg.inv(K)
        self.Kinv = Kinv

    def callback(self, img, depth):
        ## Get image as numpy array
        rgb = self.bridge.imgmsg_to_cv(img)
        img = np.asarray(rgb)

        self.marker_finder.findmarker(img)
        detections = self.marker_finder.detections()
        if len(detections) != 2:
            return
        vis = self.marker_finder.draw_detections(img)

        depth_16 = np.asarray(
            self.bridge.imgmsg_to_cv(depth, desired_encoding="passthrough"))
        depth = depth_16 / np.float32(1000.)
        lpt, rpt = [c.astype(np.int) for c, r, _ in detections]
        if np.max(depth_16) == 0:
            return
        lz = depth[lpt[1], lpt[0]]
        rz = depth[rpt[1], rpt[0]]
        if lz == 0 or rz == 0 or self.Kinv is None:
            return

        plt.clf()
        plt.imshow(vis)
        _, campt, _ = plt.ginput(3, timeout=-1)
        camz = depth[campt[0], campt[1]]

        print "Left marker", self.pt3d_from_imgpt(lpt, lz)
        print "Right marker", self.pt3d_from_imgpt(rpt, rz)

def main():
    rospy.init_node('markerconfiguration', log_level=rospy.DEBUG)
    loc = Get3DPosition()
    sub2 = rospy.Subscriber(
        rospy.resolve_name("camera/rgb/camera_info"),
        CameraInfo,
        loc.set_camera_info)

    sub0 = message_filters.Subscriber(
        rospy.resolve_name("camera/rgb/image_raw"), Image)
    sub1 = message_filters.Subscriber(
        rospy.resolve_name("camera/depth/image_raw"), Image)

    ts = ApproximateSynchronizer(0.2, [sub0, sub1], 10)
    ts.registerCallback(loc.callback)

    rospy.spin()

if __name__ == '__main__':
    main()
