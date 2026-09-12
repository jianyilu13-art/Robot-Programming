import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import (
    QoSProfile,
    HistoryPolicy,
    DurabilityPolicy,
    ReliabilityPolicy
)

class PublishersNode(Node):

    sensor_pub = None
    latch_pub = None

    def __init__(self):
        super().__init__('publishers')

        self.timer = self.create_timer(
            0.05, self.timer_callback)
        self.i = 0

        self.sensor_pub = self.create_publisher(
        Header, '/sensor', qos_profile_sensor_data)


        qos_profile_latch = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.latch_pub = self.create_publisher(
            Header, '/latch', qos_profile_latch
        )

    def timer_callback(self):
        self.i += 1
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = str(self.i)

        print(f'Publish: {self.i}')
        
        if self.sensor_pub is not None:
            self.sensor_pub.publish(msg)

        if self.latch_pub is not None:
            self.latch_pub.publish(msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = PublishersNode()
    rclpy.spin(node) 
    rclpy.shutdown()
if __name__ == '__main__':
    main()