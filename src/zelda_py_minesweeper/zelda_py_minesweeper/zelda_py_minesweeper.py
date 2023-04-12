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

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        yellowLower = (29, 100, 100)
        yellowUpper = (64, 255, 255)
        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        
        self.detect_lines(hsv)

        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        print(centroids)

        for x, y in centroids:
            if not math.isnan(x) and not math.isnan(y): #avoids nan's
                cv2.circle(cv_image, (x ,y), 2, (0, 255, 255), 2)

        # ret, thresh = cv2.threshold(hsv,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        cv2.imshow("Image", cv_image)
        cv2.waitKey(3)
        cv2.imshow("Mask", mask)
        cv2.waitKey(3)

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
