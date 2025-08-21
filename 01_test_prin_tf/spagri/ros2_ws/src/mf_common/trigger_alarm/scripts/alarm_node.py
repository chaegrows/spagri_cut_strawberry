import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import Jetson.GPIO as GPIO

LED_PIN_LIGHT = 16  # Light
LED_PIN_ALARM = 18  # Alarm

class AlarmNode(Node):
    def __init__(self):
        super().__init__('alarm_node')
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(LED_PIN_LIGHT, GPIO.OUT)
        GPIO.setup(LED_PIN_ALARM, GPIO.OUT)

        self.sub = self.create_subscription(
            String,
            '/alarm/in',
            self.alarm_callback,
            10
        )

    def triggerAlarmSignal(self, on_off: str):
        if on_off == "light":
            GPIO.output(LED_PIN_LIGHT, GPIO.HIGH)
        elif on_off == "alarm":
            GPIO.output(LED_PIN_ALARM, GPIO.HIGH)
        elif on_off == "both_on":
            GPIO.output(LED_PIN_LIGHT, GPIO.HIGH)
            GPIO.output(LED_PIN_ALARM, GPIO.HIGH)
        elif on_off == "light_off":
            GPIO.output(LED_PIN_LIGHT, GPIO.LOW)
        elif on_off == "alarm_off":
            GPIO.output(LED_PIN_ALARM, GPIO.LOW)
        elif on_off == "both_off":
            GPIO.output(LED_PIN_LIGHT, GPIO.LOW)
            GPIO.output(LED_PIN_ALARM, GPIO.LOW)
        else:
            self.get_logger().error("Invalid input: %s" % on_off)

    def alarm_callback(self, msg: String):
        self.triggerAlarmSignal(msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = AlarmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
