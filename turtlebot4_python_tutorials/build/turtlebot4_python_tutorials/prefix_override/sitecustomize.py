import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cuttlebot/Desktop/turtlebot4_python_tutorials/install/turtlebot4_python_tutorials'
