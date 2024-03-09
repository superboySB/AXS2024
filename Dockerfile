FROM jieyitsinghuawx/icra2024-sim2real-axs-baseline:v1.0.0

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

# TensorRT
WORKDIR /root/
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales git tmux gedit vim openmpi-bin openmpi-common libopenmpi-dev libgl1-mesa-glx tensorrt psmisc
RUN cd /usr/src/tensorrt/samples && make -j16

# Yolo-world
WORKDIR /root/Workspace/
RUN git clone https://github.com/superboySB/YOLOv8-TensorRT.git
RUN cd YOLOv8-TensorRT && \
    /root/miniconda3/bin/conda run -n baseline pip install --upgrade pip && \
    /root/miniconda3/bin/conda run -n baseline pip install -r requirements.txt && \
    /root/miniconda3/bin/conda run -n baseline pip install opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74 tensorrt \
    pyQt5 PySide2 pydot && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt
RUN cd YOLOv8-TensorRT && /root/miniconda3/bin/conda run -n baseline python test_yoloworld.py

# # EfficientViT + SAM
WORKDIR /root/Workspace/
RUN git clone https://github.com/superboySB/efficientvit.git
RUN cd efficientvit && \
    /root/miniconda3/bin/conda run -n baseline conda install -y mpi4py && \
    /root/miniconda3/bin/conda run -n baseline pip install -r requirements.txt && \
    mkdir -p assets/checkpoints/sam && cd assets/checkpoints/sam && \
    wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l2.pt && \
    wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt
RUN cd /root/Workspace/efficientvit && mkdir -p assets/export_models/sam/tensorrt/ && chmod -R 777 assets/export_models/sam/tensorrt/ && \
    /root/miniconda3/bin/conda run -n baseline python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
    /root/miniconda3/bin/conda run -n baseline python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
    /root/miniconda3/bin/conda run -n baseline python deployment/sam/onnx/export_encoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_encoder.onnx && \ 
    /root/miniconda3/bin/conda run -n baseline python deployment/sam/onnx/export_decoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_decoder.onnx --return-single-mask

COPY AXS_solution.py /root/Workspace/AXS_baseline/ICRA2024-Sim2Real-AXS/src/airbot/example/AXS_solution.py