import rclpy
from rclpy.node import Node


class FakeNode(Node):

    def __init__(self):
        super().__init__('fake')


def main(args=None):
    rclpy.init(args=args)
    node = FakeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
