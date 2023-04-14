import rclpy
from rclpy.node import Node
import rclpy
from rclpy import qos

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import math


class MineSweeper(Node):

    def __init__(self):
        super().__init__('minesweeper')
        self.bridge = CvBridge()

        self.camera_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos.qos_profile_sensor_data
        )

    def image_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        (numLabels, labels, stats, centroids) = self.detect_balls(img)

        cv2.imshow("Image", img)
        cv2.waitKey(3)

    def detect_balls(self, img):
        img = cv2.GaussianBlur(img, (15, 15), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        cv2.imshow("HSV", hsv)

        yellowLower = (25, 100, 100)
        yellowUpper = (80, 255, 255)
        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        return output

    def detect_lines(self, hsv):
        redLower1 = (160, 100, 100)
        redUpper1 = (180, 255, 255)

        redLower2 = (0, 100, 100)
        redUpper1 = (20, 255, 255)

        mask1 = cv2.inRange(hsv, redLower1, redUpper1)
        mask2 = cv2.inRange(hsv, redLower2, redUpper1)

        mask = mask1 + mask2

        cv2.imshow("Red Mask", mask)
        cv2.waitKey(3)


def main(args=None):
    rclpy.init(args=args)

    minesweeper = MineSweeper()

    rclpy.spin(minesweeper)

    minesweeper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
