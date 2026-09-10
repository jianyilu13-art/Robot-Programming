import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity

class Logger(Node):

    def __init__(self):
        super().__init__('logger')

        set_logger_level(self.get_logger().name, LoggingSeverity.WARN)
        
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        self.i += 1
        self.get_logger().debug('this is debug')
        self.get_logger().info(f'{self.i}')
        self.get_logger().warn(f'warn {self.i}')
        self.get_logger().error('an error')


def main(args=None):
    rclpy.init(args=args)

    node = Logger()
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()