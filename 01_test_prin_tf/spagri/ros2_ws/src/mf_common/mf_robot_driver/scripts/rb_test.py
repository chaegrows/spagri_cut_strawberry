import third_party.rbpodo as rb
import numpy as np

ROBOT_IP = "192.168.50.101"

def _main():
    try:
        robot = rb.Cobot(ROBOT_IP)
        rc = rb.ResponseCollector()

        print(robot.get_tcp_info(rc))

        print(robot.get_tfc_info(rc))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    _main()
