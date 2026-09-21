import rclpy
from rclpy.node import Node

from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterType


class ParameterServices(Node):

    def __init__(self):
        super().__init__('parameter_services')

        self.get_params_cli = self.create_client(
            GetParameters,
            '/turtlesim/get_parameters'
        )

        self.set_params_cli = self.create_client(
            SetParameters,
            '/turtlesim/set_parameters'
        )

        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_params_future = None
        self.set_params_future = None


    def timer_callback(self):

        if self.get_params_future is None:
            request = GetParameters.Request()
            request.names = ['background_r', 'background_g']
            self.get_params_future = self.get_params_cli.call_async(request)


        if self.get_params_future.done():
            response = self.get_params_future.result()

            old_r = response.values[0].integer_value
            old_g = response.values[1].integer_value

            new_r = (old_r + 50) % 256
            new_g = (old_g - 40) % 256


            if self.set_params_future is None:
                request = SetParameters.Request()

                new_r_param = Parameter()
                new_r_param.name = 'background_r'
                new_r_param.value.type = ParameterType.PARAMETER_INTEGER
                new_r_param.value.integer_value = new_r

                new_g_param = Parameter()
                new_g_param.name = 'background_g'
                new_g_param.value.type = ParameterType.PARAMETER_INTEGER
                new_g_param.value.integer_value = new_g

                request.parameters = [new_r_param, new_g_param]

                self.set_params_future = self.set_params_cli.call_async(request)

                print(f'Setting Red({new_r:3d}) and Green({new_g:3d})')


        if self.set_params_future is not None and self.set_params_future.done():
            self.set_params_future.result()

            self.get_params_future = None
            self.set_params_future = None



def main(args=None):
    rclpy.init(args=args)
    node = ParameterServices()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()