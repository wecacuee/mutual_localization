#!/usr/bin/python
import roslib; roslib.load_manifest('mutloc_ros')
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
import numpy as np
from mutloc.blinkingtracker import BlinkingMarkerFinder
from mutloc_ros.cvbridge import CvBridge
import cv

class Detector(object):
    def __init__(self, marker_detections_topic='marker_detections', detect_vis_topic='detect_vis'):
        self.bridge = CvBridge()
        self.marker_finder = BlinkingMarkerFinder()
        self.publisher = rospy.Publisher(marker_detections_topic, PolygonStamped)
        self.detect_vis = rospy.Publisher(detect_vis_topic, Image)

    def get_cv_img(self, imgmsg):
        rgb = self.bridge.imgmsg_to_cv(imgmsg)
        img = np.asarray(rgb)
        return img

    def get_detection(self, img):
        self.marker_finder.findmarker(img)
        detections = self.marker_finder.detections()
        vis = self.marker_finder.draw_detections(img)
        return np.array([[d.center[0], d.center[1]] for d in detections]), vis

    def callback(self, imgmsg):
        img = self.get_cv_img(imgmsg)
        detections, vis = self.get_detection(img)
        if len(detections) >= 1:
            self.publish(imgmsg, detections, vis)
        else:
            rospy.logdebug("did not find any detections.")

    def publish(self, imgmsg, detections, vis):
        header = Header(stamp=imgmsg.header.stamp,
                        frame_id=imgmsg.header.frame_id)
        points = [Point32(x, y, 0) for x, y in detections]

        polygon = Polygon(points=points)
        polygonstamp = PolygonStamped(header=header, polygon=polygon)
        self.publisher.publish(polygonstamp)

        im = cv.fromarray(vis)
        immsg = self.bridge.cv_to_imgmsg(im)
        self.detect_vis.publish(immsg)

def main():
    rospy.init_node('marker_detector', log_level=rospy.DEBUG)
    detector = Detector()
    resolved_topic  = rospy.resolve_name('camera/rgb/image_raw')
    rospy.logdebug("Listening to %s" % resolved_topic)
    rospy.Subscriber(resolved_topic, Image, detector.callback)
    rospy.spin()

if __name__ == '__main__':
    main()
