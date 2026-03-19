<div align="center">
  <h1>Multi-Resolution Voxelized Map-Based Stereo Visual-Inertial Odometry</h1>
</div>

## 📜 Introduction

This project implements the method in *Multi-Resolution Voxelized Map-Based Stereo Visual-Inertial Odometry*, which introduces a map-based stereo VIO framework designed for edge-cloud scenarios with limited transmission bandwidth.

The core idea is to build a **multi-resolution voxelized prior map** where each voxel stores a single representative 3D point. During online odometry, 2D image features are associated with prior 3D map points through a **depth-adaptive cone/ray indexing strategy** and **3D-DDA traversal**, enabling efficient data association with low communication and computation cost.

Built on an MSCKF-style estimator, the system integrates prior-map constraints and visual-inertial observations to improve accuracy and robustness on both indoor and outdoor datasets.

Key ideas:

- Multi-resolution voxelized prior map with one-point-per-voxel representation
- Depth-aware resolution selection and ray-based 2D-3D association
- 3D-DDA voxel traversal for efficient map indexing
- Stereo + IMU tightly-coupled state estimation with map constraints

---
Please kindly star ⭐️ this project if it helps you. We take great efforts to develop and maintain it 😁.

##  Developers:
The codes of this repo are contributed by:[Shuyi Pan](https://github.com/PANshuyi), [Zikang Yuan](https://github.com/ZikangYuan).

## 🛠️ Installation

### 1. Requirements

> GCC >= 7.5.0
>
> Cmake >= 3.16.0
> 
> [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.3.4
>
> [OpenCV](https://github.com/opencv/opencv) == 4.2.0 for Ubuntu 20.04
> 
> [PCL](https://pointclouds.org/downloads/) == 1.10 for Ubuntu 20.04
>
> [Ceres](http://ceres-solver.org/installation.html) >= 1.14
>
> [ROS](http://wiki.ros.org/ROS/Installation)


##### Have Tested On:

| OS    | GCC  | Cmake | Eigen3 | OpenCV | PCL | Ceres |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Ubuntu 20.04 | 10.5.0  | 3.16.3 | 3.3.7 | 4.2.0 | 1.10.0 | 1.14.0 |

### 2. Create ROS workspace

```bash
mkdir -p ~/MR-Voxel-SVIO/src
cd MR-Voxel-SVIO/src
```

### 3. Clone the directory and build

```bash
git clone https://github.com/PANshuyi/MR-Voxel-SVIO.git
cd ..
catkin_make
```

## 🚀 Run on Public Datasets

Noted:

A. **Please create a folder named "output" before running.** When **MR-Voxel-SVIO** is running, the estimated pose is recorded in real time in the **pose.txt** located in the **output folder**.

B. Our ground-truth pose files are provided in the `GT-Pose` folders, prior maps are provided in the [*prior_voxel_lidar_map*](https://pan.baidu.com/s/16Kbg_5e2BA3yA2XTLViHGA?pwd=1234 ). You may also download data from the official dataset websites [*KAIST*](https://sites.google.com/view/complex-urban-dataset) and [*Euroc_MAV*](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset). However, for KAIST, the ground-truth poses need to be transformed into the IMU coordinate frame.

C. Before running, update the launch file to use the ground-truth pose file and prior map corresponding to the target sequence.

###  1. Run on [*EuRoC_MAV*](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)

Please go to the workspace of **MR-Voxel-SVIO** and type:

```bash
cd MR-Voxel-SVIO
source devel/setup.bash
roslaunch mr_voxel_svio vio_euroc.launch
```

Then open the terminal in the path of the bag file, and type:

```bash
rosbag play SEQUENCE_NAME.bag --clock -d 1.0
```

###  2. Run on [*KAIST*](https://sites.google.com/view/complex-urban-dataset)

Please go to the workspace of **MR-Voxel-SVIO** and type:

```bash
cd MR-Voxel-SVIO
source devel/setup.bash
roslaunch mr_voxel_svio vio_kaist.launch
```

Then open the terminal in the path of the bag file, and type:

```bash
rosbag play SEQUENCE_NAME.bag --clock -d 1.0
```

For the KAIST dataset, the extrinsic parameters of sequences *urban38* and *urban39* differ from other sequences. When processing *urban38* or *urban39*, please use **kaist2.yaml**; for all other sequences, please use **kaist.yaml**.

## 🤓 Acknowledgments

We would like to express our gratitude to the following projects, which have provided significant support and inspiration for our work:
- [Voxel-SVIO](https://github.com/ZikangYuan/voxel_svio): A MSCKF-based stereo visual-inertial odometry framework with voxel-based map management. We especially thank the project for its clear system design and open-source implementation.
- [Open-VINs](https://github.com/rpng/open_vins): An open source platform for visual-inertial navigation research, the implementation of our MSCKF is based on it.
- [DSO](https://github.com/JakobEngel/dso): A monocular direct sparse visual odometry, our feature extraction and matching is inspired from it.
- [VINs-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono): A robust and versatile monocular visual-inertial state estimator, our keyframe selection and marginalization is based on it.
- [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion): An optimization-based multi-sensor state estimator extending VINS-Mono, supporting monocular/stereo visual-inertial setups and GPS fusion examples.
- [GMMLoc](https://github.com/HKUSTGZ-IADC/gmmloc): A Gaussian Mixture Model based localization framework for map-based localization.

