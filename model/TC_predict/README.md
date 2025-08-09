# 如何将图像和点云用PointPillar的方法编码
## 1. 点云扩展特征

   除了 (x,y,z)，你可以在点的特征向量里加入：
    - RGB颜色
    - 该点对应图像语义分割的语义类别或概率（通过点的投影坐标对应UNet输出取值）

   例如，每个点的特征是 [x,y,z,r,g,b,s1,s2,...,sN]，其中s*是语义概率或one-hot。

## 2. 点云Pillar化

   按照 PointPillars 的方法，将点云划分成固定大小的pillar（XY方向的柱子），每个pillar最多保留固定数量的点。

## 3. Pillar Feature Net (PFN)

   使用MLP对pillar内点特征编码，生成固定维度pillar特征。

## 4. BEV伪图像

   将所有pillar特征按位置映射到BEV二维网格，形成多通道特征图。

## 5. 后续处理

   用2D CNN做后续的预测或代价学习。