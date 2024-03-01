# sudo docker rm -f baseline

sudo docker run -itd --name solution --network=host \
    --gpus all \
    --privileged \
    --device=/dev/ttyUSB0 \
    --device=/dev/input/js0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v $HOME/Desktop/shared:/shared \
    superboysb/axs2024:v1.0.0 \
    /bin/bash
