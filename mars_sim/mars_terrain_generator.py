#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mars Terrain Generator for Gazebo Simulation
自动化生成Gazebo火星场景仿真脚本

使用方法:
python mars_terrain_generator.py --texture ESP_083964_1980_RED_C_01_ORTHO.JP2 --heightmap DTEEC_083964_1980_083529_1980_A01.IMG --x 1000 --y 1000 --size 257
"""

import os
import sys
import argparse
import shutil
import numpy as np
from osgeo import gdal, gdalconst

class MarsTerrainGenerator:
    def __init__(self, texture_file, heightmap_file, x, y, size, model_name="mars_terrain"):
        self.model_name = model_name
        self.texture_file = texture_file
        self.heightmap_file = heightmap_file
        self.x = x
        self.y = y
        self.size = size
        self.root_dir = os.path.dirname(texture_file) if texture_file else os.getcwd()
        self.output_dir = os.path.join(self.root_dir, model_name)
        self.gazebo_models_path = os.path.expanduser("~/.gazebo/models")
        # 启用GDAL异常
        gdal.UseExceptions()
        self.generate(texture_file, heightmap_file, x, y, size)
        
    def validate_size(self, size):
        """验证尺寸是否为2^n+1格式"""
        if size <= 0:
            return False
        # 检查是否为2^n+1格式
        n = 0
        while (2**n + 1) <= size:
            if (2**n + 1) == size:
                return True
            n += 1
        return False
    
    def get_file_info(self, filepath):
        """获取文件信息"""
        print(f"📋 获取文件信息: {filepath}")
        try:
            ds = gdal.Open(filepath, gdalconst.GA_ReadOnly)
            if ds is None:
                raise Exception(f"无法打开文件: {filepath}")
            
            # 获取基本信息
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            bands = ds.RasterCount
            
            # 获取地理变换信息
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            
            print(f"   尺寸: {cols} x {rows}")
            print(f"   波段数: {bands}")
            print(f"   像素分辨率: {geotransform[1]:.6f} x {abs(geotransform[5]):.6f}")
            
            if projection:
                print(f"   投影: {projection[:100]}...")
            
            ds = None  # 关闭数据集
            print("✓ 文件信息获取成功")
            
        except Exception as e:
            print(f"✗ 获取文件信息失败: {e}")
            sys.exit(1)
    
    def convert_texture_to_png(self, jp2_file):
        """将JP2纹理图转换为PNG格式"""
        print(f"🔄 转换纹理图格式: {jp2_file}")
        output_file = os.path.dirname(jp2_file) + "/texture_temp.png"
        
        try:
            # 打开源文件
            src_ds = gdal.Open(jp2_file, gdalconst.GA_ReadOnly)
            if src_ds is None:
                raise Exception(f"无法打开JP2文件: {jp2_file}")
            
            # 创建PNG驱动
            png_driver = gdal.GetDriverByName('PNG')
            if png_driver is None:
                raise Exception("PNG驱动不可用")
            
            # 创建输出数据集，转换为Byte类型
            dst_ds = png_driver.CreateCopy(output_file, src_ds, options=['WORLDFILE=NO'])
            
            # 如果数据类型不是Byte，需要进行缩放
            if src_ds.GetRasterBand(1).DataType != gdalconst.GDT_Byte:
                print("   执行数据类型转换...")
                # 使用gdal.Translate进行类型转换
                translate_options = gdal.TranslateOptions(
                    outputType=gdalconst.GDT_Byte,
                    scaleParams=[[0, 255]]
                )
                dst_ds = gdal.Translate(output_file, src_ds, options=translate_options)
            
            # 关闭数据集
            src_ds = None
            dst_ds = None
            
            print(f"✓ 纹理图转换完成: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"✗ 纹理图转换失败: {e}")
            sys.exit(1)
    
    def crop_files(self, texture_file, heightmap_file, x, y, size):
        """裁剪纹理图和高程图"""
        print(f"✂️  裁剪文件 - 位置:({x},{y}) 尺寸:{size}x{size}")
        
        texture_output = os.path.dirname(texture_file) + "/texture_cropped.png"
        heightmap_output = os.path.dirname(heightmap_file) + "/heightmap_cropped.IMG"
        
        try:
            # 裁剪纹理图
            print("   裁剪纹理图...")
            src_ds = gdal.Open(texture_file, gdalconst.GA_ReadOnly)
            if src_ds is None:
                raise Exception(f"无法打开纹理文件: {texture_file}")
            
            # 检查裁剪范围是否有效
            if x + size > src_ds.RasterXSize or y + size > src_ds.RasterYSize:
                raise Exception(f"裁剪范围超出图像边界: ({x},{y},{size},{size}) vs ({src_ds.RasterXSize},{src_ds.RasterYSize})")
            
            translate_options = gdal.TranslateOptions(
                srcWin=[x, y, size, size],
                outputType=gdalconst.GDT_Byte
            )
            gdal.Translate(texture_output, src_ds, options=translate_options)
            src_ds = None
            print(f"✓ 纹理图裁剪完成: {texture_output}")
            
            # 裁剪高程图
            print("   裁剪高程图...")
            src_ds = gdal.Open(heightmap_file, gdalconst.GA_ReadOnly)
            if src_ds is None:
                raise Exception(f"无法打开高程图文件: {heightmap_file}")
            
            # 检查裁剪范围是否有效
            if x + size > src_ds.RasterXSize or y + size > src_ds.RasterYSize:
                raise Exception(f"裁剪范围超出图像边界: ({x},{y},{size},{size}) vs ({src_ds.RasterXSize},{src_ds.RasterYSize})")
            
            translate_options = gdal.TranslateOptions(
                srcWin=[x, y, size, size]
            )
            gdal.Translate(heightmap_output, src_ds, options=translate_options)
            src_ds = None
            print(f"✓ 高程图裁剪完成: {heightmap_output}")
            
            return texture_output, heightmap_output
            
        except Exception as e:
            print(f"✗ 文件裁剪失败: {e}")
            sys.exit(1)
    
    def analyze_heightmap(self, heightmap_path):
        """分析高程图获取尺寸和高程范围"""
        print(f"📊 分析高程图: {heightmap_path}")
        
        try:
            ds = gdal.Open(heightmap_path, gdalconst.GA_ReadOnly)
            if ds is None:
                raise Exception(f"无法打开高程图文件: {heightmap_path}")
            
            # 获取地理变换信息
            gt = ds.GetGeoTransform()
            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            
            # 读取数据数组
            arr = band.ReadAsArray()
            
            # 处理无效数据
            if nodata in arr:
                raise ValueError(f"高程图数据中包含无效值: {nodata}")
            else:
                valid_data = arr.flatten()
            
            # 计算统计信息
            min_elev = float(np.min(valid_data))
            max_elev = float(np.max(valid_data))
            elev_range = max_elev - min_elev
            
            # 获取像素尺寸
            x_size = ds.RasterXSize
            y_size = ds.RasterYSize
            
            # 计算实际地理尺寸
            pixel_width = abs(gt[1])
            pixel_height = abs(gt[5])
            
            real_x_length = pixel_width * x_size
            real_y_length = pixel_height * y_size
            
            ds = None  # 关闭数据集
            
            print(f"✓ 高程分析完成:")
            print(f"   最小高程: {min_elev:.2f}")
            print(f"   最大高程: {max_elev:.2f}")
            print(f"   高程范围: {elev_range:.2f}")
            print(f"   实际尺寸: {real_x_length:.2f} x {real_y_length:.2f}")
            
            return {
                'min_elev': min_elev,
                'max_elev': max_elev,
                'elev_range': elev_range,
                'real_x_length': real_x_length,
                'real_y_length': real_y_length
            }
            
        except Exception as e:
            print(f"✗ 高程分析失败: {e}")
            sys.exit(1)
    
    def normalize_heightmap(self, heightmap_file, min_elev, max_elev):
        """归一化高程图并转换为PNG"""
        print(f"🔧 归一化高程图: {min_elev:.2f} - {max_elev:.2f}")
        
        output_file = os.path.dirname(heightmap_file) + "/heightmap.png"
        
        try:
            src_ds = gdal.Open(heightmap_file, gdalconst.GA_ReadOnly)
            if src_ds is None:
                raise Exception(f"无法打开高程图文件: {heightmap_file}")
            
            # 使用gdal.Translate进行缩放和格式转换
            translate_options = gdal.TranslateOptions(
                outputType=gdalconst.GDT_Byte,
                scaleParams=[[min_elev, max_elev, 0, 255]],
                format='PNG'
            )
            
            dst_ds = gdal.Translate(output_file, src_ds, options=translate_options)
            
            # 关闭数据集
            src_ds = None
            dst_ds = None
            
            print(f"✓ 高程图归一化完成: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"✗ 高程图归一化失败: {e}")
            sys.exit(1)
    
    def create_directory_structure(self):
        """创建目录结构"""
        print(f"📁 创建目录结构: {self.output_dir}")
        
        # 删除已存在的目录
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        # 创建新目录结构
        materials_dir = os.path.join(self.output_dir, "materials", "textures")
        os.makedirs(materials_dir, exist_ok=True)
        
        print(f"✓ 目录创建完成: {materials_dir}")
        return materials_dir
    
    def copy_textures(self, materials_dir, texture_file, heightmap_file):
        """复制纹理文件到目标目录"""
        print("📋 复制纹理文件")
        
        # 重命名并复制文件
        texture_dest = os.path.join(materials_dir, "texture.png")
        heightmap_dest = os.path.join(materials_dir, "heightmap.png")
        
        shutil.copy2(texture_file, texture_dest)
        shutil.copy2(heightmap_file, heightmap_dest)
        
        print(f"✓ 文件复制完成:")
        print(f"   纹理图: {texture_dest}")
        print(f"   高程图: {heightmap_dest}")
    
    def create_model_config(self):
        """创建model.config文件"""
        print("📝 创建model.config文件")
        
        config_content = f'''<?xml version="1.0"?>
<model>
  <name>{self.model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Author Name</name>
    <email>address@email.com</email>
  </author>
  <description>Mars terrain heightmap model</description>
</model>'''
        
        config_path = os.path.join(self.output_dir, "model.config")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✓ model.config创建完成: {config_path}")
    
    def create_model_sdf(self, terrain_info):
        """创建model.sdf文件"""
        print("📝 创建model.sdf文件")
        
        sdf_content = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{self.model_name}">
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <collision name="collision">
        <geometry>
          <heightmap>
            <uri>model://{self.model_name}/materials/textures/heightmap.png</uri>
            <size>{terrain_info['real_x_length']:.2f} {terrain_info['real_y_length']:.2f} {terrain_info['elev_range']:.2f}</size>
            <pos>0 0 0</pos>
          </heightmap>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <heightmap>
            <uri>model://{self.model_name}/materials/textures/heightmap.png</uri>
            <size>{terrain_info['real_x_length']:.2f} {terrain_info['real_y_length']:.2f} {terrain_info['elev_range']:.2f}</size>
            <pos>0 0 0</pos>
            <texture>
              <diffuse>model://{self.model_name}/materials/textures/texture.png</diffuse>
              <size>{terrain_info['real_x_length']:.2f}</size>
            </texture>
          </heightmap>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>'''
        
        sdf_path = os.path.join(self.output_dir, "model.sdf")
        with open(sdf_path, 'w', encoding='utf-8') as f:
            f.write(sdf_content)
        
        print(f"✓ model.sdf创建完成: {sdf_path}")
        print(f"   地形尺寸: {terrain_info['real_x_length']:.2f} x {terrain_info['real_y_length']:.2f} x {terrain_info['elev_range']:.2f}")
    
    def install_to_gazebo(self):
        """安装模型到Gazebo"""
        print(f"🚀 安装模型到Gazebo: {self.gazebo_models_path}")
        
        # 创建Gazebo模型目录
        os.makedirs(self.gazebo_models_path, exist_ok=True)
        
        # 复制整个模型目录
        dest_path = os.path.join(self.gazebo_models_path, self.model_name)
        if os.path.exists(dest_path):
            print(f"   覆盖已存在的模型: {dest_path}")
            shutil.rmtree(dest_path)
        
        shutil.copytree(self.output_dir, dest_path)
        print(f"✓ 模型安装完成: {dest_path}")
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        print("🧹 清理临时文件")
        temp_files = [
            "texture_temp.png",
            "texture_cropped.png", 
            "heightmap_cropped.IMG",
            "heightmap.png"
        ]
        temp_files.extend([file_name + '.aux.xml' for file_name in temp_files])
        temp_files = [os.path.join(self.root_dir, file) for file in temp_files]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"   删除: {temp_file}")
    
    def validate_crop_region(self, filepath, x, y, size):
        """验证裁剪区域是否有效"""
        try:
            ds = gdal.Open(filepath, gdalconst.GA_ReadOnly)
            if ds is None:
                return False, f"无法打开文件: {filepath}"
            
            width = ds.RasterXSize
            height = ds.RasterYSize
            
            if x < 0 or y < 0:
                return False, f"裁剪起始坐标不能为负数: ({x}, {y})"
            
            if x + size > width or y + size > height:
                return False, f"裁剪区域超出图像边界: 裁剪区域({x}, {y}, {size}, {size}) vs 图像尺寸({width}, {height})"
            
            # 检查裁剪区域是否包含nodata值
            band = ds.GetRasterBand(1)
            if band is None:
                return False, f"文件 {filepath} 不包含有效波段"
            nodata = band.GetNoDataValue()
            if nodata is not None:
                # 读取裁剪区域的数据
                crop_data = band.ReadAsArray(x, y, size, size)
                if nodata in crop_data:
                    return False, f"裁剪区域包含无效数据(nodata={nodata})，请选择其他区域"
            
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def generate(self, texture_file, heightmap_file, x, y, size):
        """主要生成流程"""
        print(f"🚀 开始生成火星地形模型: {self.model_name}")
        print("=" * 50)
        
        # 验证输入文件
        if not os.path.exists(texture_file):
            print(f"✗ 纹理文件不存在: {texture_file}")
            sys.exit(1)
        
        if not os.path.exists(heightmap_file):
            print(f"✗ 高程图文件不存在: {heightmap_file}")
            sys.exit(1)
        
        # 验证尺寸
        if not self.validate_size(size):
            print(f"✗ 无效的尺寸: {size}. 必须为2^n+1格式 (如129, 257, 513等)")
            sys.exit(1)
        
        # 验证裁剪区域
        valid, error_msg = self.validate_crop_region(texture_file, x, y, size)
        if not valid:
            print(f"✗ 纹理图裁剪区域无效: {error_msg}")
            sys.exit(1)
        
        valid, error_msg = self.validate_crop_region(heightmap_file, x, y, size)
        if not valid:
            print(f"✗ 高程图裁剪区域无效: {error_msg}")
            sys.exit(1)
        
        try:
            # 1. 获取文件信息
            self.get_file_info(texture_file)
            self.get_file_info(heightmap_file)
            
            # 2. 转换纹理图格式
            temp_texture = self.convert_texture_to_png(texture_file)
            
            # 3. 裁剪文件
            cropped_texture, cropped_heightmap = self.crop_files(
                temp_texture, heightmap_file, x, y, size)
            
            # 4. 分析高程图
            terrain_info = self.analyze_heightmap(cropped_heightmap)
            
            # 5. 归一化高程图
            final_heightmap = self.normalize_heightmap(
                cropped_heightmap, terrain_info['min_elev'], terrain_info['max_elev'])
            
            # 6. 创建目录结构
            materials_dir = self.create_directory_structure()
            
            # 7. 复制纹理文件
            self.copy_textures(materials_dir, cropped_texture, final_heightmap)
            
            # 8. 创建配置文件
            self.create_model_config()
            self.create_model_sdf(terrain_info)
            
            # 9. 安装到Gazebo
            self.install_to_gazebo()
            
            # 10. 清理临时文件
            self.cleanup_temp_files()
            
            print("=" * 50)
            print(f"🎉 火星地形模型生成完成: {self.model_name}")
            print(f"📁 本地模型目录: {self.output_dir}")
            print(f"🚀 Gazebo模型路径: {os.path.join(self.gazebo_models_path, self.model_name)}")
            print(f"💡 在Gazebo中搜索 '{self.model_name}' 即可使用该模型")
            
        except KeyboardInterrupt:
            print("\n⏹️  用户中断操作")
            self.cleanup_temp_files()
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ 生成过程中出现错误: {e}")
            self.cleanup_temp_files()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="自动化生成Gazebo火星场景仿真模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            示例用法:
            python mars_terrain_generator.py --name "mars_crater_region" --texture ESP_083964_1980_RED_C_01_ORTHO.JP2 --heightmap DTEEC_083964_1980_083529_1980_A01.IMG --x 1000 --y 1000 --size 257

            # 或使用默认名称
            python mars_terrain_generator.py --texture ESP_083964_1980_RED_C_01_ORTHO.JP2 --heightmap DTEEC_083964_1980_083529_1980_A01.IMG --x 1000 --y 1000 --size 257
        """
    )
    
    parser.add_argument('--name', '-n', type=str, default='mars_terrain',
                       help='模型名称 (默认: mars_terrain)')
    parser.add_argument('--texture', required=True,
                       help='纹理图文件路径 (.JP2格式)')
    parser.add_argument('--heightmap', required=True,
                       help='高程图文件路径 (.IMG格式)')
    parser.add_argument('--x', type=int, required=True,
                       help='裁剪区域左上角X坐标')
    parser.add_argument('--y', type=int, required=True,
                       help='裁剪区域左上角Y坐标')
    parser.add_argument('--size', type=int, required=True,
                       help='裁剪区域尺寸 (必须为2^n+1格式，如129, 257, 513)')
    
    args = parser.parse_args()
    
    generator = MarsTerrainGenerator(args.texture, args.heightmap, args.x, args.y, args.size, model_name="mars_terrain",)
    

if __name__ == "__main__":
    main()