import rclpy
from rclpy.node import Node
import rclpy


class MineSweeper(Node):

    def __init__(self):
        super().__init__('minesweeper')

        self.get_logger().info("Minesweeper node has been started")


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
