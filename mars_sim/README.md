# 火星地形生成方法
## Get Start
### Install
执行下面的命令创建所需环境
```bash 
conda create -n mars_sim python=3.10
conda activate mars_sim
conda install -c conda-forge gdal numpy pillow libgdal-jp2openjpeg
pip install -r mars_sim/requirements.txt
sudo apt install ros-noetic-jackal-desktop ros-noetic-jackal-simulator ros-noetic-teleop-twist-keyboard ros-noetic-velodyne ros-noetic-velodyne-gazebo-plugins
```

### Create a Mars environment
1. 在[HiRISE](https://www.uahirise.org/dtm/)网站找到想要的DTM，下载以.IMG为后缀的高程图和以.JP2为后缀的纹理图，通常一个DTM目录中只有一个高程图文件，但有多个纹理图，高程图文件名示例：DTEEC_083964_1980_083529_1980_A01.IMG，纹理图文件名示例：ESP_083964_1980_RED_C_01_ORTHO.JP2，高程图是由两个轨道卫星的相机合成的，文件名中“083964_1980”和“083529_1980”分别代表左右两个相机，我们通常选择下载左相机的以“RED_C_01_ORTHO”为后缀的纹理图，**并将它们放在同一个目录下**，可以利用`gdal`工具查看下载文件的信息，例如
```bash
gdalinfo DTEEC_083964_1980_083529_1980_A01.IMG
```
2. 下载完成后，为了能够将地图可视化，便于后续选取ROI裁剪，先将纹理图进行格式转换，执行
```bash
gdal_translate -ot Byte -of PNG <file_name>.JP2 texture.png
```
可以看到所在目录下有已经转换好的png纹理图

3. 纹理图的拍摄区域一般是和高程图对应的（即使有误差也误差不大，可以用`gdalinfo `查看），所以我们只对纹理图选取ROI，由于将高程图和纹理图导入gazebo要求图片大小必须为$(2^n+1)\times (2^n+1)$，所以我们一般选择的区域一般为$129\times 129$或$257\times 257$，找到选取区域左上角的像素点位置(x,y)，注意选取的区域中不能包含黑的的nodata区域，执行
```bash
cd mars_sim
python mars_terrain_generator.py --name mars_terrain --texture <path_to_texure_file> --heightmap <path_to_heightmap_file> --x <x> --y <y> --size <size>
```
可以看到已经生成已经生成名称为`mars_terrain`的目录了，并且可以在gazebo的默认模型目录下`~/.gazebo/models/`找到！

### Compile and Run
下面编译ros包
```bash
cd ros_ws
source /opt/ros/noetic/setup.bash
catkin build
```
编译成功之后就可以直接运行了！
```bash
source devel/setup.bash
roslaunch mars_simulation start_simulation.launch
```
现在导入场景的车只有base，为了能够采集数据，需要挂载一些传感器
```bash
cd /opt/ros/noetic/share/jackal_description
echo -e "JACKAL_LASER_3D=1\nJACKAL_FLEA3=1" > urdf/configs/custom
```
现在挂载了一个前置的相机和一个激光雷达，为了避免读取雷达点云时卡死，将`urdf/accessories/vlp16_mount.urdf.xacro`中的所有`VLP-16`函数中的`gpu`参数设置为`true`，`samples`和`hz`酌情设置
```xml
<!-- <xacro:VLP-16 parent="${prefix}_vlp16_plate" topic="${topic}"> -->
<!-- set gpu as true -->
<xacro:VLP-16 parent="${parent_link}" topic="${topic}" gpu="true" hz="30" samples="1000">
```
然后回到我们的`ros_ws`，将`launch/start_simulation.launch`中加载机器人部分的`config`参数设置为`custom`，下面重新launch就可以用传感器采集数据啦！打开rviz订阅`/front/raw_image`和`/mid/points`就可以看到相应的图像和点云了
