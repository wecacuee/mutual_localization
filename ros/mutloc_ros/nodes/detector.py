#!/usr/bin/python
import roslib; roslib.load_manifest('mutloc_ros')
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
import numpy as np
from mutloc.blinkingtracker import BlinkingMarkerFinder
from mutloc_ros.cv_bridge import CvBridge
import cv
from mutloc_ros.detector import Detector

def main():
    rospy.init_node('marker_detector', log_level=rospy.DEBUG)
    detector = Detector()
    resolved_topic  = rospy.resolve_name('camera/rgb/image_raw')
    rospy.logdebug("Listening to %s" % resolved_topic)
    rospy.Subscriber(resolved_topic, Image, detector.callback)
    rospy.spin()

if __name__ == '__main__':
    main()
