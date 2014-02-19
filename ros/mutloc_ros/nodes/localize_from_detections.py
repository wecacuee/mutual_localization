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
from mutloc_ros.localize_from_detections import Localizer

class BarcodeMarkerFinder(object):
    def findmarker(self, img, tag):
        return mutloc.findmarkers.find_barcode(
                cvimg0, tag=tag, canny_params=(50,240))

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
