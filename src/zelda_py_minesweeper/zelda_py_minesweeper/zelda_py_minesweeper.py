import rclpy
from rclpy.node import Node
import rclpy
from rclpy import qos

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry
from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector
from cv_bridge import CvBridge

import cv2
import math
import numpy as np
import signal
import sys
from geometry_msgs.msg import Twist
TIMER_INTERVAL = 0.1
TURNING_POWER = .7
MAX_POWER = 1
STOP_INTERVAL = 2.5

# DO_MOVE = False
DO_MOVE = True

TARGET_POSITION = np.array([1.0, 1.0, 0.0])
ANGLE_TOLERANCE_DEG = 5.0
POSITION_TOLERANCE = 0.1


def to_np(p: Point):
    return np.array([p.x, p.y, p.z])


# Taken from: https://gist.github.com/salmagro/2e698ad4fbf9dae40244769c5ab74434
def euler_from_quaternion(quaternion: Quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def prop(error, k, max_power):
    return np.clip(error * k, -max_power, max_power)


class MineSweeper(Node):

    def __init__(self):
        super().__init__('minesweeper')
        self.bridge = CvBridge()

        # self.camera_subscription = self.create_subscription(
        #     Image,
        #     '/camera/color/image_raw',
        #     self.image_callback,
        #     qos.qos_profile_sensor_data
        # )
        # self.camera_subscription  # prevent unused variable warning

        self.move_publisher = self.create_publisher(Twist, 'zelda/cmd_vel', 10)
        self.move_timer = self.create_timer(
            TIMER_INTERVAL,
            self.move_timer_callback
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            'zelda/odom',
            self.odom_callback,
            qos.qos_profile_sensor_data
        )
        self.odom_subscription

        self.stop_timer = None

        self.move_state = "orient"
        self.slight_turn = 1.0
        self.forward_vel = 0.0

        # Odometry
        self.base_position = None
        self.base_orientation = None
        self.position = None
        self.orientation = None

    def odom_callback(self, odom: Odometry):
        raw_position = to_np(odom.pose.pose.position)
        raw_orientation = euler_from_quaternion(odom.pose.pose.orientation)

        if self.base_position is None:
            self.base_position = raw_position

        if self.base_orientation is None:
            self.base_orientation = raw_orientation

        self.position = raw_position - self.base_position
        self.orientation = raw_orientation - self.base_orientation

        print(f"Position {self.position}")

        desired_angle = math.atan2(
            TARGET_POSITION[1] - self.position[1],
            TARGET_POSITION[0] - self.position[0]
        )

        angle_diff = desired_angle - self.orientation[2]

        if self.move_state == "orient":
            if abs(angle_diff) < math.radians(ANGLE_TOLERANCE_DEG):
                self.move_state = "move"
                self.start_pos = self.position
                self.move_dist = np.linalg.norm(
                    TARGET_POSITION - self.position)
            else:
                self.slight_turn = prop(angle_diff, 1.0, 1.0)

        if self.move_state == "move":
            dist = np.linalg.norm(self.position - self.start_pos)
            if dist > self.move_dist:
                self.move_state = "stop"
                self.forward_vel = 0.0
            else:
                self.forward_vel = prop(self.move_dist - dist, 1.0, 0.2)

    def signal_handler(self, sig, frame):
        # signal handler to catch Ctrl+C and Ctrl+Z and close all cv2 windows
        cv2.destroyAllWindows()
        sys.exit(0)

    def image_callback(self, msg):
        signal.signal(signal.SIGINT, self.signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTSTP, self.signal_handler)  # Ctrl+Z
        # self.get_logger().info('I heard: "%s"' % msg.data)

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # self.detect_line_prob(img)

        (numLabels, labels, stats, centroids) = self.detect_balls(img)

        self.numLabels = numLabels
        maxIdx = 0
        maxY = 0
        for i in range(1, numLabels):
            x, y = centroids[i]
            if not math.isnan(x) and not math.isnan(y):  # avoids nan's
                x = int(x)
                y = int(y)
                cv2.circle(img, (x, y), 2, (255, 0, 0), 2)
                if maxY < y:
                    maxIdx = i
                    maxY = y

        if numLabels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            maxIdx = np.argmax(areas) + 1
            trackedCentroid = centroids[maxIdx]

            # draw the detected centroid on the image
            (x, y) = trackedCentroid
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)

        if (numLabels > 1):
            self.move_state = "forward"
        elif numLabels == 1:
            self.stop_timer = self.create_timer(
                STOP_INTERVAL,
                self.stop_timer_callback
            )

        x, y = centroids[maxIdx]
        power = (-x + img.shape[1]/2)/(img.shape[1]/2)
        self.slight_turn = min(
            max(power * TURNING_POWER, -MAX_POWER), MAX_POWER)

        # self.get_logger().info(f"x {x} y {y}, turn {power}")

        cv2.imshow("Image", img)

    def stop_timer_callback(self):
        if self.numLabels == 1:
            self.move_state = "searching"
        self.stop_timer.destroy()

    def move_timer_callback(self):
        twist = Twist()

        # self.get_logger().info(f"moving {self.move_state}")

        if self.move_state == "forward":
            twist.linear.x = .05
            twist.angular.z = self.slight_turn
        elif self.move_state == "searching":
            twist.angular.z = self.slight_turn
        elif self.move_state == "orient":
            twist.angular.z = self.slight_turn
        elif self.move_state == "move":
            twist.linear.x = self.forward_vel
            # twist.angular.z = self.slight_turn

        if DO_MOVE:
            self.move_publisher.publish(twist)

    def detect_balls(self, img):
        # img = cv2.GaussianBlur(img, (9, 9), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.detect_lines(hsv)

        cv2.imshow("HSV", hsv)

        yellowLower = (25, 50, 100)
        yellowUpper = (80, 255, 255)

        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow("ball image", mask)

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

    def detect_line_prob(self, img):
        img = cv2.GaussianBlur(img, (15, 15), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define range of red color in HSV
        lower_red = (0, 50, 50)
        upper_red = (15, 255, 255)
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = (170, 70, 70)
        upper_red = (180, 255, 255)
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2
        cv2.imshow('Lines', mask)

        edges = cv2.Canny(mask, threshold1=100, threshold2=150)
        cv2.imshow('Canny', edges)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180, threshold=70, minLineLength=65, maxLineGap=25)

        img_copy = img.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # display the image with detected lines
        cv2.imshow('Red lines', img_copy)
        cv2.waitKey(3)


def main(args=None):
    rclpy.init(args=args)

    minesweeper = MineSweeper()

    rclpy.spin(minesweeper)

    minesweeper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
