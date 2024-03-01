# ICRA AgileX Sim2Real Challenge
BIT-LINC方案

## What's News!!!
The simulation stage of the competition has begun. Please submit your client-side Docker image to the team's Docker Hub, following GitHub's instructions. Additionally, submit your Docker Hub account and operation instructions of your client docker to the organizing committee via “Mail” section of the website before 23:59 on March 10. The organizing committee will evaluate the competition based on the code submitted by the team. Teams must also submit a technical report. The template for the report can be downloaded from the Data section of the website and should be sent to sim2real@air.tsinghua.edu.cn before 11:59 pm on March 14. The results of the simulation phase will be determined by a "test result: technical report" ratio of 7:3. Final scores will be published on March 15 via the website.

## 安装过程
下拉代码
```sh
git clone https://github.com/superboySB/AXS2024 && cd AXS2024 && sudo chmod a+x scripts/*
```
构建两个containers，一个是omni gibson的，一个是baseline方法的，他们会自动拉image。
```sh
./scripts/run_omni.sh
./scripts/run_baseline.sh
```
运行两个containers
```sh
./scripts/exec_env.sh
./scripts/exec_baseline.sh
```
ROS相关的话题可以先参考[官方教程](docs/sim2real-install-guide.md)，回头再学习一下,基本如下：

### Core (ROS Master)

This part serves as the communication pivot in ROS systems.

#### server

In Phase 1, the task is executed in a simulated scenario. Therefore, we built a real-world test scenario with isaac-sim.

The interface between the server and the client is defined by the ROS topic, which can be found here [ros topic](./Platform_introduction.md).

#### Client

In this repo, we provide a baseline method for this task. A functional image can be obtained by building the image or pulling .jieyitsinghuawx/icra2024-sim2real-axs-baseline:v0.0.2


The baseline image (and potentially the images you built) is based on the base image . Basic functional units (e.g. keyboard control) are included in this base image. Please refer to this repo for further information on the base image .jieyitsinghuawx/icra2024-sim2real-axs-baseline:v0.0.2

## 运行仿真
进入OmniGibsom仿真对应的containers，在保证`DISPLAY`正常的情况下启动仿真
```sh
conda activate omnigibson && cd /omnigibson-src

roscore &
python -m omnigibson.AXS_env --ik
```
启动ros TF publish
```sh
roslaunch airbot_play_launch robot_state_publisher.launch robot_description_path:=/root/OmniGibson-Airbot/omnigibson/data/assets/models/airbot_play_with_rm2/airbot_with_texture/urdf_obj/AIRBOT_V3_v2-3.urdf

roslaunch airbot_play_launch static_transform_publisher.launch
```
启动IK service
```sh
roslaunch airbot_play_launch airbot_play_moveit.launch use_rviz:=true target_moveit_config:=airbot_play_v2_1_config use_basic:=true
```
[Optional] 尝试键盘控制（issue:我发现3/4对应关节没有反应，导致机械臂没办法探出去）
```sh
python /root/OmniGibson-Airbot/teleop_twist_keyboard_AXS.py
```

## 运行baseline
进入算法对应另一个containers，在保证`DISPLAY`正常的情况下在多个consle依次启动`hdl localization`节点、`base control`节点和`main baseline service`节点
```sh
cd ~/Workspace

roslaunch hdl_localization hdl_localization.launch

conda activate baseline && python /root/robot_tools/examples/ros_base_control.py

conda activate baseline && python /root/Workspace/AXS_baseline/ICRA2024-Sim2Real-AXS/src/airbot/example/AXS_baseline.py
```


## 运行自研算法
ongoing

## 提交结果镜像
Submitting images requires registering for the ICRA2024-Sim2Real-RM challenges.

Players create a personal docker hub account,create repository in Repositories, and save your Repository Name.

If players already have an account,they can directly enter their account password to log in:
```sh
docker login
```
After logging in, use the following command to view the image ID that needs to be submitted：
```sh
docker images
```
Then change the name of the image that needs to be submitted：
```sh
docker tag {image_id} {username}/{repository_name}:{image_version}
```
Submit to dockerhub:
```sh
docker push {username}/{repository_name}:{image_version}
```



