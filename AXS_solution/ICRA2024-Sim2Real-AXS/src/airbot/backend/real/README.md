```bash
docker pull discoverrobotics/airbot_play:lvg_v1.1.6
docker run -dit --network=host \
--name arm \
--privileged=true \
--device=/dev/ttyUSB0 \
--device=/dev/input/js0 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
discoverrobotics/airbot_play:lvg_v1.1.6 \
/bin/bash



# inside docker
source devel/setup.bash
roslaunch ros_interface airbot_joint.launch

```