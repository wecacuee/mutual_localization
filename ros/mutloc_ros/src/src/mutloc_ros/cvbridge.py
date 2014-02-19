from __future__ import absolute_import
import cv_bridge
import cv

class CvBridge(cv_bridge.CvBridge):
    def encoding_as_cvtype(self, encoding):
        try:
            return cv_bridge.CvBridge.encoding_as_cvtype(self, encoding)
        except AttributeError:
            if encoding == "bayer_grbg8":
                return cv.CV_8UC1
            raise

    def imgmsg_to_cv(self, img, desired_encoding="passthrough"):
        assert desired_encoding in ["passthrough", "rgb8"]
        if img.encoding == "bayer_grbg8":
            # need to convert
            img_bayer = cv_bridge.CvBridge.imgmsg_to_cv(self, img, desired_encoding="mono8")
            rgb = cv.CreateMat(img_bayer.rows, img_bayer.cols, cv.CV_8UC3)
            cv.CvtColor(img_bayer, rgb, cv.CV_BayerGB2RGB)
        else:
            rgb = cv_bridge.CvBridge.imgmsg_to_cv(self, img,
                                            desired_encoding=desired_encoding)
        return rgb

