import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose


class Recorder(Node):

    def __init__(self):
        super().__init__('recorder')

        # Open data.txt for writing
        self.f = open('data.txt', 'w')

        # Required 0.5 second timer
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Subscribe to turtle pose
        self.subscription = self.create_subscription(
            Pose,
            '/turtle1/pose',
            self.pose_callback,
            10
        )

    def timer_callback(self):
        pass

    def pose_callback(self, msg):
        self.f.write(f'{msg.x}\t{msg.y}\t{msg.theta}\n')


def main(args=None):
    rclpy.init(args=args)

    node = Recorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.f.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()