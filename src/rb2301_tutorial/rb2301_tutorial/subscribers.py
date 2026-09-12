import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import (
    QoSProfile,
    HistoryPolicy,
    DurabilityPolicy,
    ReliabilityPolicy
)

class SubscribersNode(Node):

    latch_msg = None
    sensor_msg = None
    shallow_msg = None

    def __init__(self):
        super().__init__('subscribers')

        # how long the timer_callback will 
        # block other callbacks for (sec)
        self.timer_wait = 0.0

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.sensor_sub = self.create_subscription(
        Header, '/sensor', self.sensor_sub_callback, qos_profile_sensor_data)


        qos_profile_latch = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.latch_sub = self.create_subscription(
            Header, '/latch', self.latch_sub_callback, qos_profile_latch
        )


        qos_profile_shallow = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.shallow_sub = self.create_subscription(
            Header, '/sensor',
            self.shallow_sub_callback,
            qos_profile_shallow
        )

        self.timer_wait = 1.0

    def latch_sub_callback(self, msg):
        self.latch_msg = msg
        print(f'Received       Latch: id({msg.frame_id})')
        
    def sensor_sub_callback(self, msg):
        self.sensor_msg = msg
        print(f'Received      Sensor: id({msg.frame_id})')

    def shallow_sub_callback(self, msg):
        self.shallow_msg = msg
        print(f'Received     Shallow: id({msg.frame_id})')

    def get_elapsed(self, msg):
        now = self.get_clock().now()
        then = Time.from_msg(msg.stamp)
        return float((now - then).nanoseconds) / 1e9
    
    def print_msg(self, name, msg):
        if msg is not None:
            id = msg.frame_id
            elapsed = self.get_elapsed(msg)
            print(f'{name}: id({id}) lifespan({elapsed} sec)')
        
    def timer_callback(self):
        print('---- Timer sees: ---')
        self.print_msg('      Latch', self.latch_msg)
        self.print_msg('     Sensor', self.sensor_msg)
        self.print_msg('    Shallow', self.shallow_msg)
        print('---- ')

        # mimic some complex calculations which  
        # prevents this callback from returning, 
        # blocking other callbacks.
        self.get_clock().sleep_for(
            Duration(seconds=self.timer_wait))

def main(args=None):
    rclpy.init(args=args)
    node = SubscribersNode()
    rclpy.spin(node) 
    rclpy.shutdown()
if __name__ == '__main__':
    main()
