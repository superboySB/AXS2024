import atexit
import subprocess

import psutil
import rospy

from . import arm
from . import base
from . import camera
from . import gripper

# Global variable to keep track of the roscore subprocess
roscore_process = None
rosnode_inited = False


def is_roscore_running():
    """Check if roscore is running by looking for a process named 'roscore'."""
    for process in psutil.process_iter(['name']):
        if process.info['name'] == 'rosmaster':
            return True
    return False


def terminate_roscore():
    """Terminate the roscore process."""
    global roscore_process
    if roscore_process:
        roscore_process.terminate()
        roscore_process.wait()


def launch_roscore():
    """Launch roscore as a subprocess if it's not already running."""
    global roscore_process, rosnode_inited
    if not is_roscore_running():
        # Launch roscore in a non-blocking manner
        roscore_process = subprocess.Popen(['roscore'])
        # Register the terminate_roscore function to be called at exit
        atexit.register(terminate_roscore)
    if not rosnode_inited:
        rospy.init_node('ctrl_frontend')
        rosnode_inited = True


# Call launch_roscore when module is imported
launch_roscore()
