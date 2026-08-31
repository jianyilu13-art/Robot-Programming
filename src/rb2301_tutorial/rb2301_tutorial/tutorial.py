# 1. IMPORT
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn


# NODE CLASS
class TutorialNode(Node):

    def __init__(self):
        super().__init__('tutorial')

        # 2. NODE PROPERTIES
        self.counter = 0
        self.pose_msg = None
        self.velocity = 1.0
        self.spawn_future = None
        self.spawn_requested = False

        # 3. NODE HANDLES
        self.pose_sub = self.create_subscription(
            Pose,
            '/turtle1/pose',
            self.pose_sub_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/turtle1/cmd_vel',
            10
        )

        self.spawn_cli = self.create_client(
            Spawn,
            '/spawn'
        )

        self.timer = self.create_timer(
            0.5,
            self.timer_callback
        )

    # 4. NODE CALLBACKS

    def pose_sub_callback(self, msg):
        self.pose_msg = msg

    def timer_callback(self):
        # Increment and print counter
        self.counter += 1
        print(self.counter)

        # Print turtle position if a pose has been received
        if self.pose_msg is not None:
            x = self.pose_msg.x
            y = self.pose_msg.y
            print(f'({x}, {y})')

        # Publish velocity
        velocity_msg = Twist()
        velocity_msg.linear.x = self.velocity
        self.cmd_vel_pub.publish(velocity_msg)

        # Alternate velocity between 1.0 and -1.0
        self.velocity *= -1.0

        # Send spawn request only once throughout the entire run.
        if not self.spawn_requested:
            if self.spawn_cli.service_is_ready():
                request = Spawn.Request()
                request.x = 1.0
                request.y = 1.0
                request.theta = 0.0
                request.name = ''

                self.spawn_future = self.spawn_cli.call_async(request)
                self.spawn_requested = True

        # Check whether spawn response has arrived
        if self.spawn_future is not None:
            if self.spawn_future.done():
                response = self.spawn_future.result()
                print(response.name)

                # Keep the request from being sent again.
                self.spawn_future = None



# 5. HOW TO USE


# MAIN BOILER PLATE
def main(args=None):
    rclpy.init(args=args)

    node = TutorialNode()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()

