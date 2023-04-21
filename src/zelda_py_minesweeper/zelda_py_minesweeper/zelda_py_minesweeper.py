import rclpy
from rclpy.node import Node
import rclpy
from rclpy import qos

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import math
import numpy as np
import signal, sys
from geometry_msgs.msg import Twist
TIMER_INTERVAL = 0.1
TURNING_POWER = .7
MAX_POWER = 1
STOP_INTERVAL = 2.5

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

        self.move_publisher = self.create_publisher(Twist, 'zelda/cmd_vel', 10)
        self.move_timer = self.create_timer(
            TIMER_INTERVAL,
            self.move_timer_callback
        )

        self.stop_timer = None

        self.move_state = "searching"
        self.slight_turn = 1.0
        


    def signal_handler(self, sig, frame):
        # signal handler to catch Ctrl+C and Ctrl+Z and close all cv2 windows
        #cv2.destroyAllWindows()
        sys.exit(0)

    def image_callback(self, msg):
        signal.signal(signal.SIGINT, self.signal_handler) #Ctrl+C
        signal.signal(signal.SIGTSTP, self.signal_handler) #Ctrl+Z
        # self.get_logger().info('I heard: "%s"' % msg.data)

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #self.detect_line_prob(img)

        (numLabels, labels, stats, centroids) = self.detect_balls(img)

        self.numLabels = numLabels
        maxIdx = 0
        maxY = 0
        for i in range(1, numLabels):
            x, y = centroids[i]
            if not math.isnan(x) and not math.isnan(y): #avoids nan's
                x = int(x)
                y = int(y)
                cv2.circle(img, (x ,y), 2, (255, 0, 0), 2)
                if maxY < y:
                    maxIdx = i
                    maxY = y

        x,y = centroids[maxIdx]
        

        if (numLabels > 1):
            self.move_state = "forward"
        elif numLabels == 1:
            self.stop_timer = self.create_timer(
                STOP_INTERVAL,
                self.stop_timer_callback
            )

        power = (-x + img.shape[1]/2)/(img.shape[1]/2)
        self.slight_turn = min(max(power * TURNING_POWER, -MAX_POWER), MAX_POWER)

        self.get_logger().info(f"x {x} y {y}, turn {power}")

        cv2.imshow("Image", img)


    
    def stop_timer_callback(self):
        if self.numLabels == 1:
            self.move_state = "searching"
        self.stop_timer.destroy()



    def move_timer_callback(self):
        twist = Twist()

        self.get_logger().info(f"moving {self.move_state}")

        if self.move_state == "forward":
            twist.linear.x = .05
            twist.angular.z = self.slight_turn
        elif self.move_state == "searching":
            twist.angular.z = self.slight_turn

        self.move_publisher.publish(twist)


    def detect_balls(self, img):
        #img = cv2.GaussianBlur(img, (15, 15), 3)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.detect_lines(hsv)
        

        cv2.imshow("HSV", hsv)

        yellowLower = (25, 100, 100)
        yellowUpper = (80, 255, 255)
        
        
        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        #mask = cv2.erode(mask, None, iterations=2)
        #mask = cv2.dilate(mask, None, iterations=2)


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

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=70, minLineLength=65, maxLineGap=25)

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
