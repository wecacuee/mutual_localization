#!/usr/bin/python
import roslib; roslib.load_manifest('mutloc_ros')
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv_bridge
import message_filters
import mutloc
import numpy as np
import tf
from approxsync import ApproximateSynchronizer
from mutloc.blinkingtracker import BlinkingMarkerFinder

class BarcodeMarkerFinder(object):
    def findmarker(self, img, tag):
        return mutloc.findmarkers.find_barcode(
                cvimg0, tag=tag, canny_params=(50,240))

class Localizer(object):
    def __init__(self, marker_finder=None):
        self.count = 0
        self._img1 = None
        self.br = tf.TransformBroadcaster()
        self.triangulation_queue = []
        if self.marker_finder is None:
            self.marker_finder = BlinkingMarkerFinder(object)

    def cv_bridge(self):
        return cv_bridge.CvBridge()

    def findmarker(self, img, tag):
        return self.marker_finder.findmarker(img, tag)


    def callback(self, image0, image1):
        cvmat0, cvmat1 = [self.cv_bridge().imgmsg_to_cv(im)
                          for im in image0, image1]
        cvimg0, cvimg1 = [np.asarray(mat) for mat in (cvmat0, cvmat1)]
        try:
            image0markers = self.findmarker(cvimg0, "img0/%04d" % self.count)
            image1markers = self.find_barcode(
                cvimg1, tag="img1/%04d" % self.count)
            self.count += 1
        except mutloc.findmarkers.MarkerNotFoundException:
            return

        Troots = mutloc.compute_transform_from_marker_pos(image0markers,
                                                          image1markers,
                                                          method='analytic')
        Tfiltered = mutloc.filter_by_distance(Troots)
        Tbest = Tfiltered[0]
        self.publish(Tbest, image0.header.frame_id, image1.header.frame_id)

    def triangulation(self, T1, cvimg0, cvimg1):
        if len(self.triangulation_queue):
            T0, cvimg0, cvimg1 = self.triangulation_queue[0]

        self.triangulation_queue.append(T1, cvimg0, cvimg1)
        cv2.imwrite('cvimg%04d_static.png' % self.count, cvimg0)
        cv2.imwrite('cvimg%04d_moving.png' % self.count)

    def publish(self, T, frame_id0, frame_id1):
        q = tf.transformations.quaternion_from_matrix(T)
        translation = np.dot(T[:3, :3].T, -T[:3, 3])
        rospy.loginfo("Translation: %s" % str(translation))
        rospy.loginfo("Broadcasting transform %s wrt %s" % (frame_id1,
                                                            frame_id0))
        self.br.sendTransform(translation,
                              tf.transformations.quaternion_inverse(q),
                              rospy.Time.now(),
                              frame_id1,
                              frame_id0)

def main():
    rospy.init_node('localizer',log_level=rospy.DEBUG )
    loc = Localizer()
    sub0 = message_filters.Subscriber(rospy.resolve_name("image0"), Image)
    sub1 = message_filters.Subscriber(rospy.resolve_name("image1"), Image)
    ts = ApproximateSynchronizer(0.5, [sub0, sub1], 10)
    ts.registerCallback(loc.callback)
    rospy.spin()

if __name__ == '__main__':
    main()
