import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class Sim(Node):

    def __init__(self):
        super().__init__('sim')

        self.x = 0.0
        self.y = 0.0

        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.sub_odom_callback,
            10
        )

        self.timer = self.create_timer(
            0.2,
            self.timer_callback
        )


    def sub_odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y


    def timer_callback(self):
        print(f'x: {self.x}, y: {self.y}')


def main(args=None):
    rclpy.init(args=args)
    node = Sim()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()