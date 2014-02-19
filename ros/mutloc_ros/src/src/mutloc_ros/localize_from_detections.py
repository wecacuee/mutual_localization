#!/usr/bin/python
import roslib; roslib.load_manifest('mutloc_ros')
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PolygonStamped
import message_filters
from approxsync import ApproximateSynchronizer
import numpy as np
import mutloc
import mutloc.config as config
import sys
import tf
import os

class BarcodeMarkerFinder(object):
    def findmarker(self, img, tag):
        return mutloc.findmarkers.find_barcode(
                cvimg0, tag=tag, canny_params=(50,240))

class Localizer(object):
    def __init__(self, conf, marker_finder=None):
        self.count = 0
        self.conf = conf
        self.br = tf.TransformBroadcaster()

        # Dummy computation for initialization
        mutloc.compute_best_transform([[115, 75], [-85, 75]],
                                      [[115, 75], [-85, 75]],
                                      self.conf)

    def compute_localization(self, image0markers, image1markers):
        if len(image0markers)+len(image1markers) < 3:
            rospy.logdebug("Not enough detections. Found %d + %d" % (len(image0markers),len(image1markers)))
            return
        rospy.logdebug("Markers: {0} {1}".format(image0markers, image1markers))
        try:
            Tbest = mutloc.compute_best_transform(image0markers,
                                                   image1markers,
                                                   self.conf)
        except IndexError:
            rospy.logdebug("Unable to compute transform.")
            return
        return Tbest

    def callback(self, detect0, detect1):
        image0markers = np.array([[p.x, p.y] for p in detect0.polygon.points])
        image1markers = np.array([[p.x, p.y] for p in detect1.polygon.points])
        Tbest = self.compute_localization(image0markers, image1markers)
        if Tbest is None:
            return
        self.publish(Tbest, 
                     detect0.header.stamp + rospy.Duration(4),
                     detect1.header.frame_id,
                     detect0.header.frame_id)

    def publish(self, T, time, frame_id1, frame_id0):
        frame_id1 = "/turtlebot04" + frame_id1
        frame_id0 = "/turtlebot03" + frame_id0
        q = tf.transformations.quaternion_from_matrix(T)
        translation = np.dot(T[:3, :3].T, -T[:3, 3])
        rospy.logdebug("Translation: %s" % str(translation))
        rospy.logdebug("Broadcasting transform %s wrt %s at a delay of %f seconds"
                % ( frame_id1, frame_id0, (rospy.Time.now() - time).to_sec() ))
        self.br.sendTransform(translation,
                              tf.transformations.quaternion_inverse(q),
                              time,
                              frame_id1,
                              frame_id0)

def main(configfile):
    rospy.init_node('localizer_from_detections', log_level=rospy.DEBUG )
    conf = config.Config(open(configfile))
    loc = Localizer(conf)
    sub0 = message_filters.Subscriber(
        rospy.resolve_name("turtlebot03/marker_detections"), PolygonStamped)
    sub1 = message_filters.Subscriber(
        rospy.resolve_name("turtlebot04/marker_detections"), PolygonStamped)
    ts = ApproximateSynchronizer(0.75, [sub0, sub1], 10)
    ts.registerCallback(loc.callback)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv[1])
