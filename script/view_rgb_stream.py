#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RGBImageViewer:
    def __init__(self):
        rospy.init_node("rgb_image_viewer", anonymous=True)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/camera/rgb", Image, self.callback)
        rospy.loginfo("Subscribed to /camera/rgb")
        # republish for RViz
        self.pub = rospy.Publisher("/camera/rgb_view", Image, queue_size=1)
        rospy.spin()

    def callback(self, msg):
        try:
            # republish raw Image for RViz
            self.pub.publish(msg)
            # optional OpenCV view
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("RGB Camera View", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", e)

if __name__ == "__main__":
    try:
        RGBImageViewer()
    except rospy.ROSInterruptException:
        pass
