import rclpy
from rclpy.node import Node
import rclpy

from sensor_msgs.msg import Image
from rclpy import qos
import cv2
from cv_bridge import CvBridge

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
        #self.get_logger().info('I heard: "%s"' % msg.data)

        cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        yellowLower = (29, 100, 100)
        yellowUpper = (64, 255, 255)
        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        
        #ret, thresh = cv2.threshold(hsv,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        cv2.imshow("Image Window", mask)
        cv2.waitKey(3)


def main(args=None):
    rclpy.init(args=args)

    minesweeper = MineSweeper()

    rclpy.spin(minesweeper)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minesweeper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
