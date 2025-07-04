# 标定说明
1. 在使用前确保camera,lidar,imu三个文件夹中的`bag`文件已经被删除,因为编号是接续的,防止编号错位导致数据对应错误标定失败;
2. 在终端输入`source ~/Desktop/proj/ros_ws/devel/setup.bash`确保工作空间被正确建立;
3. 找好一个位置,在终端输入`realsense-viewer`打开realsense-viewer查看标定版是否在相机视野中;
4. 在终端输入`roslaunch livox_ros_driver2 rviz_MID360.launch`打开可视化工具检查标定版角点是否在点云中;
5. 连接好所有的设备,在终端中输入`roslaunch sensors sensors.launch`启动同步数据传输并创建同步数据话题;
6. 此时在终端中输入`rostopic list`应该可以看到`/synced/camera`,`/synced/lidar`和`/synced/imu`三个话题;
    - 注意: camera的数据格式为`sensor_msgs::Image`,lidar的数据格式为`livox_ros_driver2::CustomMsg`,imu的数据格式为`sensor_msgs::Imu`;其中点云并不是标准的`sensor_msgs::PointCloud2`格式,这里只有自定义的格式可以供后续处理,初始的lidar和camera频率为30Hz,imu频率为100Hz,同步后为10Hz.
7. 在终端输入`./record.sh`可以开始录制rosbag,10s后自动结束;
8. 重复步骤3-7,录制10个以上rosbag,保证雷达距离标定版3m以上,选择不同的角度和距离摆放标定版;
9. 得到全部的rosbag之后,需要转化成图片和pcd文件,运行
```bash
roslaunch camera_lidar_calibration imgTransfer.launch
roslaunch camera_lidar_calibration pcdTransfer.launch
```
10. 角点选取,外参计算,标定结果验证参考[官方教程](https://github.com/Livox-SDK/livox_camera_lidar_calibration/blob/master/doc_resources/README_cn.md#%E6%AD%A5%E9%AA%A44-%E6%A0%87%E5%AE%9A%E6%95%B0%E6%8D%AE%E8%8E%B7%E5%8F%96).
11. 在IMU-CAM标定时需要注意: 激励IMU的各个轴,运动需要包括俯仰,偏航和翻滚以及上下,左右和前后,可以包含随机不规则运动

# 标定数据
$$
    lidar2cam = \begin{bmatrix}
    -0.999368021973186 & -0.0234552496419775 & -0.0267096971460617 & -0.0419713959345447 \\
    0.0250930787360151 & 0.0667040370989892 & -0.997457221556015 & 0.0182503859117656 \\
    0.0251772527681233 & -0.997497079042806 & -0.0660733171872038 & -0.0780677112711236\\
    0 & 0 & 0 &	1
    \end{bmatrix}
$$

$$
    lidar2imu = \begin{bmatrix}
    1 & 0 & 0 & 0.011 \\
    0 & 1 & 0 & 0.02329 \\
    0 & 0 & 1 & -0.04412 \\
    0 & 0 & 0 & 1
    \end{bmatrix}
$$

$$ 
    cam2ego = \begin{bmatrix}
    -0.99972652 & -0.0190869 &  -0.01351239 & -0.00110909 \\
    0.00389408 &  0.43386909 & -0.90096751 & -0.04466106 \\
    0.02305929 & -0.90077373 & -0.4336761 & 0.05583883 \\
    0 & 0 & 0 & 1
    \end{bmatrix}
$$

# 实验方案
1. 找到一片适合实验的空地,应有沙子,土壤和石头等符合火星地貌的语义特征;
2. 将小车停放在一片较为平坦空旷的区域,连接线路,开启车载电脑,相机和雷达等传感器;
3. 调试无误后打开各传感器话题,打开FAST-LIVO建图工具;
4. mat_out.txt格式:<时间> <姿态角 roll pitch yaw（角度）> <位置 x y z> <速度 vx vy vz> <gyro bias> <acc bias> <时间尺度参数> <特征点数量>
   imu.txt格式:<时间戳相对值> <平均角速度 x y z> <平均加速度 x y z>;
5. 让小车以一恒定速度在场地内行驶,尽可能地行驶遍场地中的每个位置;
6. 将全部位置,速度,加速度等相关参数进行坐标转换,均转换到底盘坐标系下,记录imu得到的数据,将imu时间戳与相应位置时间戳对齐,找到每个位置对应的IMU数据,相机拍摄到的图片以及雷达获取的点云;
7. 将数据打包并转换成期望的格式进行下一步处理.