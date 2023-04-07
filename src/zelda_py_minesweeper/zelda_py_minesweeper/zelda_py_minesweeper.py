import rclpy
from rclpy.node import Node
import rclpy

from sensor_msgs.msg import Image
from rclpy import qos


class MineSweeper(Node):

    def __init__(self):
        super().__init__('minesweeper')

        self.camera_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos.qos_profile_sensor_data
        )

    def image_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


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
