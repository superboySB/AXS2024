# sudo docker rm -f baseline

sudo docker run -itd --name baseline --network=host \
    --gpus all \
    --privileged \
    --device=/dev/ttyUSB0 \
    --device=/dev/input/js0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v $HOME/Desktop/shared:/shared \
    jieyitsinghuawx/icra2024-sim2real-axs-baseline:v1.0.0 \
    /bin/bash
