```shell
# Install CUDA!

git clone --recursive https://github.com/TB5z035/LM-Inference.git

conda create -n baseline python=3.9 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
conda activate baseline

source /opt/ros/noetic/setup.bash
pip install torch
pushd thirdparty/graspnet/knn && python setup.py install && popd
pushd thirdparty/graspnet/pointnet2 && python setup.py install && popd
pushd thirdparty/graspnet && pip install -e . && popd
pushd thirdparty/segment-anything-fast && pip install -e . && popd
pushd thirdparty/yolo-v7 && pip install -e . && popd
pushd thirdparty/orocos_kinematics_dynamics/orocos_kdl && mkdir -p build && cd build && cmake .. && make -j8 && sudo make -j8 install && popd
pushd thirdparty/orocos_kinematics_dynamics/python_orocos_kdl/pybind11 && mkdir -p build && cd build && cmake .. && make -j8 && sudo make -j8 install && popd
pushd thirdparty/orocos_kinematics_dynamics/python_orocos_kdl && mkdir -p build && cd build && cmake .. && make -j8 && cp devel/lib/python3/dist-packages/PyKDL.so $CONDA_PREFIX/lib/python3.9/site-packages/ && popd

python setup.py install[dev,server,ros]


```
